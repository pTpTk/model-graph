"""Record a module-level execution graph using storage-free shape propagation."""

from dataclasses import dataclass, field
import inspect
import weakref

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_description(shape, dtype, show_floating_dtype=True):
    """Omit floating dtype annotations when they cannot describe the model."""
    suffix = f", dtype={dtype}" if show_floating_dtype or not dtype.is_floating_point else ""
    return f"Tensor(shape={shape}{suffix})"


@dataclass(eq=False)
class TensorReference:
    name: str
    shape: tuple
    dtype: torch.dtype
    producer: object = None
    show_floating_dtype: bool = True

    def __str__(self):
        return f"{self.name} {tensor_description(self.shape, self.dtype, self.show_floating_dtype)}"


@dataclass
class CacheReference:
    name: str
    cache_type: str

    def __str__(self):
        return f"{self.name} {self.cache_type}"


def cache_tensors(cache, prefix="past_key_values"):
    """Walk actual initialized cache tensors, including KV and recurrent states."""
    seen = set()

    def walk(value, path, inspect_object=False):
        if isinstance(value, torch.Tensor):
            yield path, value
            return
        if id(value) in seen:
            return
        if isinstance(value, dict):
            seen.add(id(value))
            for key, item in value.items():
                yield from walk(item, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            seen.add(id(value))
            for i, item in enumerate(value):
                yield from walk(item, f"{path}.{i}", inspect_object=path.endswith("layers"))
        elif inspect_object and hasattr(value, "__dict__"):
            seen.add(id(value))
            for key, item in vars(value).items():
                if isinstance(item, (torch.Tensor, tuple, list, dict)):
                    yield from walk(item, f"{path}.{key}")

    yield from walk(cache, prefix, inspect_object=True)


def output_cache(output):
    cache = getattr(output, "past_key_values", None)
    if cache is None and isinstance(output, dict):
        cache = output.get("past_key_values")
    if cache is None and isinstance(output, tuple) and len(output) > 1:
        cache = output[1]
    if cache is None:
        raise ValueError("The model did not return past_key_values; decode requires a cache-capable model")
    return cache


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
    shape_contract: str = ""


def format_value(value):
    if isinstance(value, (TensorReference, CacheReference)):
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
    if getattr(module, "graph_operation", None):
        return module.graph_operation
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
    if "router" in lower or name == "KimiMoEGate":
        return "Router"
    if name in INDEXER_SHAPE_CLASSES:
        return "SparseAttentionIndexer"
    if name in EXPERT_SHAPE_CLASSES:
        return "ExpertDispatch"
    if lower.endswith("moe") or "sparsemoeblock" in lower:
        return "MoE"
    if "gateddeltanet" in lower or "linearattention" in lower or "deltaattention" in lower:
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


EXPERT_SHAPE_CLASSES = {
    "DeepseekV3Experts", "DeepseekV32Experts", "DeepseekV4Experts",
    "Glm4MoeExperts", "GlmMoeDsaExperts", "Glm5NextTextExperts",
    "KimiLinearExperts", "Qwen3_5MoeExperts", "Qwen4ExpTextExperts",
    "MiniMaxM2Experts", "NemotronHExperts", "GptOssExperts",
    "Step3p7TextExperts", "MiniMaxM3SparseExperts",
    "KimiK3Experts",
    "Qwen3NextExperts", "Qwen3MoeExperts",
}

INDEXER_SHAPE_CLASSES = {"Qwen4ExpTextQSAIndexer", "Glm5NextTextIndexer"}


def indexer_shape_forward(module, hidden_states, attention_mask, past_key_values):
    """Reviewed fixed boundaries; selected positions require tensor values.

    Qwen returns an additive/bool mask matching its causal mask and caches raw
    keys. GLM returns a padded fixed-width index list and caches packed keys,
    compression gates, and a validity channel. Preserve those native layouts.
    """
    batch, query = hidden_states.shape[:2]
    if type(module).__name__ == "Qwen4ExpTextQSAIndexer":
        state_width = module.index_head_dim
        result = torch.empty_like(attention_mask)
    else:
        state_width = 2 * module.head_dim + 1
        result_width = module.index_topk
        if module.index_kpool_always_select_tail:
            result_width += module.index_kpool - 1
        result = torch.empty((batch, query, result_width), device=hidden_states.device, dtype=torch.int32)
    state = hidden_states.new_empty((batch, query, state_width))
    if past_key_values is not None:
        past_key_values.update_indexer(state, module.layer_idx)
    return result


class ModuleGraph(TorchDispatchMode):
    """Keep module calls and group the functional work between those calls.

    Hooks preserve executed call order and nesting. Dispatch observes dependencies
    between modules, including residual additions and functional attention. The
    work inside leaf modules is represented by the module itself. A functional
    region retains only tensors used outside it, so recurrent chunk internals do
    not become thousands of separate graph steps.
    """

    def __init__(self, model, show_floating_dtype=True):
        super().__init__()
        self.model = model
        self.show_floating_dtype = show_floating_dtype
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
        self.cache = None
        self.cache_input_refs = {}
        self.cache_output_refs = {}
        self.shape_contracts = []

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
        reference = TensorReference(name, tuple(tensor.shape), tensor.dtype, producer, self.show_floating_dtype)
        self.remember(tensor, reference)
        return reference

    def snapshot(self, value, producer=None, consumer=None):
        if self.cache is not None and value is self.cache:
            name = "%past_key_values" if consumer is not None and producer is None else "%updated_past_key_values"
            return CacheReference(name, type(value).__name__)
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
        if isinstance(value, torch.dtype) and value.is_floating_point and not self.show_floating_dtype:
            return "<unspecified floating dtype>"
        if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
            return value
        # Optional cache objects are disabled for the CLI; avoid their large reprs.
        return f"<{type(value).__name__}>"

    def add_placeholder(self, name, kind, tensor, target=None):
        reference = TensorReference(f"%{name}", tuple(tensor.shape), tensor.dtype, show_floating_dtype=self.show_floating_dtype)
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
        if step.module_type in EXPERT_SHAPE_CLASSES:
            expert_kind = "GatedExpertMLP" if hasattr(module, "gate_up_proj") else "ExpertMLP"
            step.shape_contract = f"GatherTokens -> {expert_kind} -> RoutingWeightMultiply -> ScatterAdd; tokens_per_expert=data-dependent"
        elif step.module_type in INDEXER_SHAPE_CLASSES:
            step.shape_contract = "QueryKeyProjection -> KeyNormalization -> BlockPool -> MatMul -> TopK -> IndexOrMask; selected_positions=data-dependent; native indexer cache layout preserved"
            if step.module_type == "Qwen4ExpTextQSAIndexer":
                step.shape_contract = step.shape_contract.replace("KeyNormalization", "QKNormalization -> RoPE")
        if getattr(module, "graph_shape_contract", None):
            step.shape_contract = module.graph_shape_contract
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
            arguments = dict(bound.arguments)
        except (TypeError, ValueError):
            arguments = {"args": args, **kwargs}
        step.inputs = self.snapshot(arguments, consumer=step)
        for name, parameter in module.named_parameters(recurse=step.module_type in INDEXER_SHAPE_CLASSES):
            step.inputs[f"parameter:{name}"] = self.snapshot(parameter, consumer=step)
        atomic = not any(module.children()) or step.module_type in EXPERT_SHAPE_CLASSES | INDEXER_SHAPE_CLASSES
        self.stack.append((step, path, atomic))

    def leave_module(self, module, args, kwargs, output):
        step, path, atomic = self.stack[-1]
        if step is not None:
            self.flush()
            step.outputs = self.snapshot(output, producer=step, consumer=step)
            if step.module_type in INDEXER_SHAPE_CLASSES and self.cache is not None:
                prefix = f"past_key_values.layers.{module.layer_idx}."
                updates = {key: value for key, value in cache_tensors(self.cache)
                           if key.startswith(prefix) and "indexer" in key}
                if updates:
                    step.outputs = {"result": step.outputs,
                                    "cache_updates": self.snapshot(updates, producer=step, consumer=step)}
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
        # Cache writes must create a new graph value even when the tensor object
        # is mutated in place. Input cache references retain their original shapes.
        for index, argument in enumerate(func._schema.arguments):
            alias = argument.alias_info
            if alias is not None and alias.is_write:
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                if isinstance(value, torch.Tensor):
                    entry = self.references.get(id(value))
                    if entry is not None:
                        self.references.pop(id(value))
                    self.reference(value, producer=step)
        self.snapshot(output, producer=step, consumer=step)
        return output

    def trace(self, inputs, warmup_inputs=None):
        for name, parameter in self.model.named_parameters():
            self.add_placeholder("p_" + name.replace(".", "_"), "parameter", parameter, name)
        for name, buffer in self.model.named_buffers():
            self.add_placeholder("b_" + name.replace(".", "_"), "buffer", buffer, name)
        patched = []
        for path, module in self.model.named_modules():
            if type(module).__name__ not in EXPERT_SHAPE_CLASSES | INDEXER_SHAPE_CLASSES:
                continue
            original = module.forward
            had_instance_forward = "forward" in vars(module)
            if type(module).__name__ in INDEXER_SHAPE_CLASSES:
                signature = inspect.signature(original)
                def shape_forward(*args, _module=module, _signature=signature, **kwargs):
                    arguments = _signature.bind(*args, **kwargs).arguments
                    return indexer_shape_forward(_module, arguments["hidden_states"],
                                                 arguments["attention_mask"], arguments.get("past_key_values"))
            else:
                def shape_forward(hidden_states, *args, **kwargs):
                    # The reviewed native expert interfaces scatter their contributions
                    # back into this same hidden-state shape. Routing values and the
                    # per-expert token count cannot be determined from meta tensors.
                    return torch.empty_like(hidden_states)
            shape_forward.__signature__ = inspect.signature(original)
            module.forward = shape_forward
            patched.append((module, original, had_instance_forward))
            self.shape_contracts.append(path)
        try:
            with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False) as fake_mode:
                def fake_arguments(arguments):
                    return {
                        name: fake_mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                        for name, value in arguments.items()
                    }

                fake_inputs = fake_arguments(inputs)
                if warmup_inputs is not None:
                    # No hooks or recording mode during cache preparation: only the
                    # subsequent one-token call appears in the decode graph.
                    self.cache = output_cache(self.model(**fake_arguments(warmup_inputs)))
                    fake_inputs["past_key_values"] = self.cache
                    entries = list(cache_tensors(self.cache))
                    if not entries:
                        raise ValueError("Prefill did not initialize any cache tensors")
                    for path, value in entries:
                        name = path.replace(".", "_")
                        self.add_placeholder(name, "cache_input", value, path)
                        self.cache_input_refs[path] = self.reference(value)
                for name, value in fake_inputs.items():
                    if isinstance(value, torch.Tensor):
                        self.add_placeholder(name, "user_input", value)
                for path, module in self.model.named_modules():
                    self.hooks.append(module.register_forward_pre_hook(
                        lambda mod, args, kwargs, path=path: self.enter_module(path or "<root>", mod, args, kwargs),
                        with_kwargs=True,
                    ))
                    self.hooks.append(module.register_forward_hook(self.leave_module, with_kwargs=True))
                with self:
                    output = self.model(**fake_inputs)
                    self.flush()
                    self.result = self.snapshot(output)
                    if self.cache is not None:
                        self.cache_output_refs = self.snapshot(dict(cache_tensors(output_cache(output))))
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            for module, original, had_instance_forward in patched:
                if had_instance_forward:
                    module.forward = original
                else:
                    del module.forward
        return self

    def print(self, stream):
        if self.shape_contracts:
            print("# Expert dispatch and sparse indexers use reviewed native boundary shape contracts where required; data-dependent token counts and selected positions are not inferred.", file=stream)
        for ref, kind, target in self.placeholders:
            label = kind + (f" {target}" if target is not None else "")
            print(f"{ref.name} = {label} -> {tensor_description(ref.shape, ref.dtype, self.show_floating_dtype)}", file=stream)

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
                if step.shape_contract:
                    details += f" shape_contract=[{step.shape_contract}]"
                arguments = ", ".join(f"{key}={format_value(value)}" for key, value in step.inputs.items())
                output = format_value(step.outputs)
            print(f"{indent}[step {step.number}] {label} [{step.path}]{details}"
                  f"({arguments}) -> {output}", file=stream)
            for child in step.children:
                emit(child, depth + 1)
        if self.cache_input_refs:
            print("%past_key_values = cache_state {" + ", ".join(
                f"{key!r}: {value.name}" for key, value in self.cache_input_refs.items()
            ) + "}", file=stream)
        for step in self.roots:
            emit(step, 0)
        if self.cache_output_refs:
            print(f"%updated_past_key_values = cache_state {format_value(self.cache_output_refs)}", file=stream)
        print(f"return {format_value(self.result)}", file=stream)
