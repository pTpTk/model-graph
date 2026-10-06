"""Record a module-level execution graph using storage-free shape propagation."""

from dataclasses import dataclass, field
import inspect
import weakref

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass(eq=False)
class TensorReference:
    name: str
    shape: tuple
    dtype: torch.dtype
    producer: object = None

    def __str__(self):
        return f"{self.name} Tensor(shape={self.shape}, dtype={self.dtype})"


@dataclass
class Step:
    number: int
    label: str
    path: str
    inputs: dict = field(default_factory=dict)
    outputs: object = None
    children: list = field(default_factory=list)
    operations: dict = field(default_factory=dict)
    tensor_count: int = 0
    module_type: str = ""


def format_value(value):
    if isinstance(value, TensorReference):
        return str(value)
    if isinstance(value, tuple):
        parts = ", ".join(format_value(item) for item in value)
        return f"({parts}{',' if len(value) == 1 else ''})"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key!r}: {format_value(item)}" for key, item in value.items()) + "}"
    return repr(value)


def module_label(module, path):
    name = type(module).__name__
    lower = name.lower()
    if path.rsplit(".", 1)[-1] == "lm_head":
        return "OutputProjection"
    if isinstance(module, torch.nn.Linear) or name == "Conv1D":
        return "Linear"
    if isinstance(module, torch.nn.Embedding):
        return "Embedding"
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        return name
    if "gateddeltanet" in lower or "linearattention" in lower:
        return "LinearAttention"
    if "attention" in lower:
        return "Attention"
    if "rotary" in lower:
        return "RotaryEmbedding"
    if "norm" in lower:
        return "Normalization"
    if "mlp" in lower or "feedforward" in lower:
        return "MLP"
    if "decoderlayer" in lower or lower.endswith("block"):
        return "TransformerBlock"
    return name


def operation_label(func):
    """Use mathematical names, without backend overload or kernel names."""
    name = func._schema.name.split("::")[-1]
    labels = {
        "add": "Add", "sub": "Subtract", "mul": "Multiply", "div": "Divide",
        "mm": "MatMul", "bmm": "MatMul", "matmul": "MatMul", "addmm": "Linear",
        "linear": "Linear", "convolution": "Convolution", "conv1d": "Conv1d",
        "_softmax": "Softmax", "softmax": "Softmax", "embedding": "Embedding",
        "view": "Reshape", "_unsafe_view": "Reshape", "reshape": "Reshape",
        "_to_copy": "Cast", "alias": "Alias", "detach": "Detach",
        "copy_": "Copy", "slice": "Slice", "select": "Select",
        "native_layer_norm": "LayerNorm", "rsqrt": "ReciprocalSqrt",
    }
    if "scaled_dot_product" in name:
        return "ScaledDotProductAttention"
    return labels.get(name, "".join(part.capitalize() for part in name.strip("_").split("_")))


class ModuleGraph(TorchDispatchMode):
    """Keep module calls and group the functional work between those calls.

    Hooks preserve executed call order and nesting. Dispatch observes dependencies
    between modules, including residual additions and functional attention. The
    work inside leaf modules is represented by the module itself. A functional
    region retains only tensors used outside it, so recurrent chunk internals do
    not become thousands of separate graph steps.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.roots = []
        self.stack = []
        self.region = None
        self.references = {}
        self.placeholders = []
        self.hooks = []
        self.step_count = 0
        self.constant_count = 0
        self.modules = dict(model.named_modules())
        self.result = None

    def new_step(self, label, path):
        self.step_count += 1
        step = Step(self.step_count, label, path)
        parent = self.stack[-1][0] if self.stack else None
        (parent.children if parent is not None else self.roots).append(step)
        return step

    def remember(self, tensor, reference):
        key = id(tensor)

        def discard(ref):
            current = self.references.get(key)
            if current is not None and current[0] is ref:
                self.references.pop(key)

        self.references[key] = (weakref.ref(tensor, discard), reference)

    def reference(self, tensor, producer=None):
        entry = self.references.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            return entry[1]
        if producer is None:
            name = f"constant_{self.constant_count}"
            self.constant_count += 1
        else:
            name = f"%s{producer.number}_{producer.tensor_count}"
            producer.tensor_count += 1
        reference = TensorReference(name, tuple(tensor.shape), tensor.dtype, producer)
        self.remember(tensor, reference)
        return reference

    def snapshot(self, value, producer=None, consumer=None):
        if isinstance(value, torch.Tensor):
            reference = self.reference(value, producer)
            source = reference.producer
            if source is not None and source.label == "Functional" and source is not consumer:
                source.outputs.setdefault(reference.name, reference)
            return reference
        if isinstance(value, tuple):
            return tuple(self.snapshot(item, producer, consumer) for item in value)
        if isinstance(value, list):
            return [self.snapshot(item, producer, consumer) for item in value]
        if isinstance(value, dict):
            return {key: self.snapshot(item, producer, consumer) for key, item in value.items()}
        if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
            return value
        # Optional cache objects are disabled for the CLI; avoid their large reprs.
        return f"<{type(value).__name__}>"

    def add_placeholder(self, name, kind, tensor, target=None):
        reference = TensorReference(f"%{name}", tuple(tensor.shape), tensor.dtype)
        self.remember(tensor, reference)
        self.placeholders.append((reference, kind, target))

    def flush(self):
        self.region = None

    def enter_module(self, path, module, args, kwargs):
        if self.stack and self.stack[-1][2]:
            self.stack.append((None, path, True))
            return
        self.flush()
        step = self.new_step(module_label(module, path), path)
        step.module_type = type(module).__name__
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
            arguments = dict(bound.arguments)
        except (TypeError, ValueError):
            arguments = {"args": args, **kwargs}
        step.inputs = self.snapshot(arguments, consumer=step)
        for name, parameter in module.named_parameters(recurse=False):
            step.inputs[f"parameter:{name}"] = self.snapshot(parameter, consumer=step)
        atomic = not any(module.children())
        self.stack.append((step, path, atomic))

    def leave_module(self, module, args, kwargs, output):
        step, path, atomic = self.stack[-1]
        if step is not None:
            self.flush()
            step.outputs = self.snapshot(output, producer=step, consumer=step)
        self.stack.pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Device/layout queries made by FakeTensor are bookkeeping, not model ops.
        if func._schema.name.startswith("prim::"):
            return func(*args, **kwargs)
        if self.stack and self.stack[-1][2]:
            return func(*args, **kwargs)
        if self.region is None:
            path = self.stack[-1][1] if self.stack else "<root>"
            self.region = self.new_step("Functional", path)
            self.region.outputs = {}
        step = self.region
        # Only cross-region tensors are dependencies; internal chunk intermediates
        # are intentionally represented by this one functional step.
        def inputs(value):
            if isinstance(value, torch.Tensor):
                ref = self.snapshot(value, consumer=step)
                if ref.producer is not step:
                    step.inputs.setdefault(ref.name, ref)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    inputs(item)
            elif isinstance(value, dict):
                for item in value.values():
                    inputs(item)
        inputs(args)
        inputs(kwargs)
        label = operation_label(func)
        step.operations[label] = step.operations.get(label, 0) + 1
        output = func(*args, **kwargs)
        self.snapshot(output, producer=step, consumer=step)
        return output

    def trace(self, inputs):
        for name, parameter in self.model.named_parameters():
            self.add_placeholder("p_" + name.replace(".", "_"), "parameter", parameter, name)
        for name, buffer in self.model.named_buffers():
            self.add_placeholder("b_" + name.replace(".", "_"), "buffer", buffer, name)
        for path, module in self.model.named_modules():
            self.hooks.append(module.register_forward_pre_hook(
                lambda mod, args, kwargs, path=path: self.enter_module(path or "<root>", mod, args, kwargs),
                with_kwargs=True,
            ))
            self.hooks.append(module.register_forward_hook(self.leave_module, with_kwargs=True))
        try:
            with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False) as fake_mode:
                fake_inputs = {
                    name: fake_mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                    for name, value in inputs.items()
                }
                for name, value in fake_inputs.items():
                    if isinstance(value, torch.Tensor):
                        self.add_placeholder(name, "user_input", value)
                with self:
                    output = self.model(**fake_inputs)
                    self.flush()
                    self.result = self.snapshot(output)
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
        return self

    def print(self, stream):
        for ref, kind, target in self.placeholders:
            label = kind + (f" {target}" if target is not None else "")
            print(f"{ref.name} = {label} -> Tensor(shape={ref.shape}, dtype={ref.dtype})", file=stream)

        structural = {
            "Reshape", "Unsqueeze", "Squeeze", "Transpose", "Permute", "Expand",
            "Slice", "Select", "SplitWithSizes", "Split", "Clone", "Copy", "Alias",
            "Cast", "Detach", "ConstantPadNd", "Zeros", "ZerosLike", "Ones",
            "Empty", "EmptyLike", "Arange", "NewOnes", "ScalarTensor",
        }

        def emit(step, depth):
            if step.label == "Functional" and not step.outputs:
                return
            indent = "  " * depth
            if step.label == "Functional":
                compute = [name for name in step.operations if name not in structural]
                names = compute or list(step.operations)
                operations = ", ".join(names)
                label = "Functional"
                if names == ["Add"]:
                    module = self.modules.get(step.path)
                    label = "ResidualAdd" if module is not None and module_label(module, step.path) == "TransformerBlock" else "Add"
                elif names == ["Multiply"]:
                    label = "ElementwiseMultiply"
                elif "MatMul" in names and "linear_attn" in step.path:
                    label = "LinearAttentionCore"
                elif "MatMul" in names and ("attn" in step.path or "attention" in step.path):
                    label = "AttentionCore"
                elif not compute:
                    label = "TensorTransform"
                details = f" ops=[{operations}]"
                arguments = ", ".join(format_value(value) for value in step.inputs.values())
                output = format_value(tuple(step.outputs.values()))
            else:
                label = step.label
                details = f" type={step.module_type}"
                arguments = ", ".join(f"{key}={format_value(value)}" for key, value in step.inputs.items())
                output = format_value(step.outputs)
            print(f"{indent}[step {step.number}] {label} [{step.path}]{details}"
                  f"({arguments}) -> {output}", file=stream)
            for child in step.children:
                emit(child, depth + 1)
        for step in self.roots:
            emit(step, 0)
        print(f"return {format_value(self.result)}", file=stream)
