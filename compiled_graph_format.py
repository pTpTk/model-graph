"""Render compiler FX operations in the module graph style with execution stages."""

from dataclasses import dataclass, field
import re

import torch

from model_ops import CacheReference, TensorReference, module_label, operation_name_label, output_cache, tensor_description


@dataclass
class ModuleRegion:
    number: int
    path: str
    stages: list
    children: list = field(default_factory=list)
    nodes: list = field(default_factory=list)


@dataclass
class ValueReference:
    name: str
    description: str

    def __str__(self):
        return f"{self.name} {self.description}"


class CompiledGraphFormatter:
    def __init__(self, program):
        self.program = program
        self.modules = dict(program.model.named_modules())
        self.refs = {}
        self.steps = {}
        self.stages = {}
        self.covered_stages = {}
        self.names = set()
        self.placeholders = []
        self.cache_inputs = {}
        self.identity_refs = {}
        self.roots = []
        self.scope_paths = {}
        self.step_count = 0
        self.cache = output_cache(program.result) if program.cache_outputs else None
        self.build()

    def unique_name(self, name):
        base = re.sub(r"\W", "_", name)
        candidate, suffix = base, 1
        while candidate in self.names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.names.add(candidate)
        return "%" + candidate

    def reference(self, name, value):
        if isinstance(value, torch.Tensor):
            return TensorReference(name, tuple(value.shape), value.dtype,
                                   show_floating_dtype=self.program.show_floating_dtype)
        return ValueReference(name, "?" if value is None else repr(value))

    def node_value(self, node):
        return node.meta.get("val", node.meta.get("example_value"))

    def result_references(self, node, number):
        count = 0

        def snapshot(value):
            nonlocal count
            if isinstance(value, (tuple, list)):
                values = [snapshot(item) for item in value]
                return tuple(values) if isinstance(value, tuple) else values
            if isinstance(value, dict):
                return {key: snapshot(item) for key, item in value.items()}
            ref = self.reference(f"%s{number}_{count}", value)
            count += 1
            return ref
        return snapshot(self.node_value(node))

    def scopes(self, node):
        if node in self.scope_paths:
            return self.scope_paths[node]
        paths = [""]
        for source, _ in node.meta.get("nn_module_stack", {}).values():
            path = re.sub(r"^L\[['\"]self['\"]\]\.?", "", source)
            if path not in self.modules:
                matches = [name for name in self.modules if name and source.endswith("." + name)]
                path = max(matches, key=len) if matches else ""
            if path and path not in paths:
                paths.append(path)
        self.scope_paths[node] = paths
        return paths

    def scope_stage(self, path):
        if not path:
            return "Inference"
        module = self.modules[path]
        label = module_label(module, path)
        name = path.rsplit(".", 1)[-1]
        if label == "TransformerBlock":
            return "Layer" + name
        if name in ("input_layernorm", "input_layer_norm"):
            return "AttentionNormalization"
        if name in ("post_attention_layernorm", "post_attention_layer_norm"):
            return "MLPNormalization"
        if name == "qkv_proj":
            return "QKVProjection"
        if name in ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj"):
            return {"q_proj": "QueryProjection", "k_proj": "KeyProjection", "v_proj": "ValueProjection",
                    "o_proj": "AttentionOutputProjection", "out_proj": "AttentionOutputProjection"}[name]
        if name in ("router", "gate") and (label == "Router" or "mlp" in path or "moe" in path):
            return "Routing"
        if label == "RotaryEmbedding":
            return "PositionalEncoding"
        if label in ("Normalization", "FusedAddRMSNorm") and path in ("model.norm", "transformer.ln_f"):
            return "FinalNormalization"
        if label == "NativeAttention":
            return "AttentionCore"
        if label == "MLP" and hasattr(module, "experts"):
            return "MoE"
        if name == "model" or name == "transformer":
            return "Decoder" if hasattr(self.program.model, "lm_head") else "Model"
        return label

    def operation_stage(self, node, paths):
        trace = node.meta.get("stack_trace", "")
        path = paths[-1]
        label = self.op_label(node)
        if label == "FusedAddRMSNorm":
            return ["ResidualAdd", "Normalization"] if node.args[2] is not None else ["Normalization"]
        if label == "RoPEAndKVCache":
            return ["RotaryPositionEncoding", "KVCacheUpdate"]
        if label == "NativeAttention":
            return ["AttentionMask", "AttentionCore"]
        if label == "TopKSoftmax":
            return ["RoutingTopK", "RoutingSoftmax"]
        if self.scope_stage(path) == "QKVProjection":
            return ["QueryProjection", "KeyProjection", "ValueProjection"]
        if label == "Split" and "self_attn" in path and any(
            self.scopes(item)[-1].endswith("qkv_proj") for item in node.all_input_nodes
        ):
            return "QKVSplit"
        if label == "Select" and isinstance(node.args[0], torch.fx.Node) and node.args[0].target == "split":
            if node.args[1] in (0, 1, 2) and "self_attn" in path:
                return ["QKVSplit", ("QueryProjection", "KeyProjection", "ValueProjection")[node.args[1]]]
        if "masking_utils.py" in trace:
            return "AttentionMask"
        if "cache_utils.py" in trace:
            return "KVCacheUpdate"
        if "apply_rotary" in trace:
            return "RotaryPositionEncoding"
        if "eager_attention_forward" in trace or "scaled_dot_product" in self.target_name(node):
            return "AttentionCore"
        if "Routing" in [self.scope_stage(item) for item in paths]:
            return "Routing"
        if label == "ExpertDispatch":
            return "ExpertDispatch"
        if "Attention" in [self.scope_stage(item) for item in paths]:
            for name, stage in (("q_proj", "QueryProjection"), ("k_proj", "KeyProjection"),
                                ("v_proj", "ValueProjection"), ("o_proj", "AttentionOutputProjection")):
                if f"self.{name}(" in trace:
                    return stage
            if "attn_output" in trace and label in ("Reshape", "Contiguous", "Transpose"):
                return "AttentionOutputPreparation"
        module = self.modules[path]
        if label == "Add" and module_label(module, path) == "TransformerBlock":
            producers = [item for item in node.all_input_nodes if item.op != "placeholder"]
            if any(any("mlp" in scope or "moe" in scope for scope in self.scopes(item)) for item in producers):
                return "MLPResidual"
            return "AttentionResidual"
        if path == "" and "self.lm_head(" in trace:
            return "OutputPreparation"
        return self.scope_stage(path) if path else "Functional"

    @staticmethod
    def target_name(node):
        target = node.target
        if callable(target) and not isinstance(target, torch._ops.OpOverload):
            return f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', type(target).__name__)}"
        return str(target)

    def op_label(self, node):
        if node.op == "get_attr":
            return "Constant"
        target = node.target
        name = target._schema.name.split("::")[-1] if isinstance(target, torch._ops.OpOverload) else getattr(target, "__name__", str(target))
        return operation_name_label(name)

    def build(self):
        active = []
        for node in self.program.graph_module.graph.nodes:
            if node.op == "placeholder":
                kind, _, target = self.program.labels.get(node.name, "input " + node.name).partition(" ")
                prefix = {"parameter": "p_", "buffer": "b_"}.get(kind, "")
                ref = self.reference(self.unique_name(prefix + target), self.node_value(node))
                self.refs[node] = ref
                self.placeholders.append((ref, kind, target))
                if kind == "cache_input":
                    self.cache_inputs[target] = ref
                continue
            if node.op == "output":
                continue
            paths = self.scopes(node)
            shared = 0
            while shared < min(len(paths), len(active)) and paths[shared] == active[shared].path:
                shared += 1
            active = active[:shared]
            for path in paths[shared:]:
                self.step_count += 1
                region = ModuleRegion(self.step_count, path, [self.scope_stage(item.path) for item in active] + [self.scope_stage(path)])
                (active[-1].children if active else self.roots).append(region)
                active.append(region)
            self.step_count += 1
            self.steps[node] = self.step_count
            self.refs[node] = self.result_references(node, self.step_count)
            active[-1].children.append(node)
            for region in active:
                region.nodes.append(node)
            stages = [self.scope_stage(path) for path in paths]
            stage = self.operation_stage(node, paths)
            if isinstance(stage, list):
                self.covered_stages[node] = stage
            elif stage != stages[-1]:
                stages.append(stage)
            self.stages[node] = stages
        self.identity_refs.update({key: self.refs[node] for key, node in self.program.input_nodes.items()})
        self.identity_refs.update({key: self.refs[node] for key, node in self.program.output_nodes.items()})
        for path, value in self.program.cache_inputs.items():
            if path not in self.cache_inputs:
                ref = self.reference(self.unique_name(path), value)
                self.cache_inputs[path] = ref
                self.placeholders.append((ref, "cache_input", path))
                self.identity_refs.setdefault(id(value), ref)

    def format(self, value):
        if self.cache is not None and value is self.cache:
            return str(CacheReference("%updated_past_key_values", type(value).__name__))
        if isinstance(value, torch.fx.Node):
            return self.format(self.refs[value])
        if isinstance(value, (TensorReference, ValueReference, CacheReference)):
            return str(value)
        if isinstance(value, torch.Tensor):
            if id(value) in self.identity_refs:
                return self.format(self.identity_refs[id(value)])
            return tensor_description(tuple(value.shape), value.dtype, self.program.show_floating_dtype)
        if isinstance(value, (tuple, list)):
            parts = ", ".join(self.format(item) for item in value)
            return f"({parts}{',' if len(value) == 1 else ''})" if isinstance(value, tuple) else f"[{parts}]"
        if isinstance(value, dict):
            return "{" + ", ".join(f"{key!r}: {self.format(item)}" for key, item in value.items()) + "}"
        if isinstance(value, torch.dtype) and value.is_floating_point and not self.program.show_floating_dtype:
            return "unspecified_floating_dtype"
        if value is None or isinstance(value, (str, int, float, bool, slice, torch.dtype, torch.device)):
            return repr(value)
        return f"<{type(value).__name__}>"

    def print(self, stream):
        print(f"# Captured FX: nodes={len(list(self.program.graph_module.graph.nodes))}, operations={len(self.steps)}.", file=stream)
        print("# Numbered compiler operations; nested module steps summarize their children.", file=stream)
        print("# stages=[...] lists enclosing model execution stages, then the operation's specific stage.", file=stream)
        if self.covered_stages:
            print("# covers=[...] lists multiple stages implemented by one operation; entries are not nested.", file=stream)
        for ref, kind, target in self.placeholders:
            print(f"{ref.name} = {kind} {target} -> {str(ref).split(' ', 1)[1]}", file=stream)
        if self.cache_inputs:
            print("%past_key_values = cache_state " + self.format(self.cache_inputs), file=stream)

        def emit(item, depth):
            indent = "  " * depth
            if isinstance(item, ModuleRegion):
                members = set(item.nodes)
                inputs = dict.fromkeys(source for node in item.nodes for source in node.all_input_nodes if source not in members)
                # Composite summaries leave descendant weights on their own module steps.
                inputs = [source for source in inputs if source.op != "placeholder"
                          or self.program.labels.get(source.name, "").split(" ", 1)[0] not in ("parameter", "buffer")
                          or self.program.labels[source.name].split(" ", 1)[1].rsplit(".", 1)[0] == item.path]
                outputs = [node for node in item.nodes if any(user not in members for user in node.users)]
                module = self.modules[item.path]
                label = module_label(module, item.path)
                print(f"{indent}[step {item.number}] {label} [{item.path or '<root>'}] type={type(module).__name__}"
                      f" stages=[{' > '.join(item.stages)}]"
                      f"({', '.join(self.format(source) for source in inputs)}) -> {self.format(tuple(outputs))}", file=stream)
                for child in item.children:
                    emit(child, depth + 1)
                return
            node = item
            path = self.scopes(node)[-1]
            label = self.op_label(node)
            if self.stages[node][-1] in ("AttentionResidual", "MLPResidual"):
                label = "ResidualAdd"
            arguments = [self.format(arg) for arg in node.args]
            arguments.extend(f"{key}={self.format(value)}" for key, value in node.kwargs.items())
            covers = f" covers=[{', '.join(self.covered_stages[node])}]" if node in self.covered_stages else ""
            print(f"{indent}[step {self.steps[node]}] {label} [{path or '<root>'}]"
                  f" stages=[{' > '.join(self.stages[node])}]{covers} op={self.target_name(node)}"
                  f"({', '.join(arguments)}) -> {self.format(self.refs[node])}", file=stream)
        for region in self.roots:
            emit(region, 0)
        if self.program.cache_outputs:
            print("%updated_past_key_values = cache_state " + self.format(self.program.cache_outputs), file=stream)
        print("return " + self.format(self.program.result), file=stream)
