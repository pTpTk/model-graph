"""Capture complete torch.compile FX graphs without weight or activation storage."""

from contextlib import contextmanager

import torch
from torch.utils._pytree import tree_flatten
from torch._subclasses import fake_tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from model_ops import cache_tensors, model_shape_contracts, output_cache


@contextmanager
def allow_meta_constants():
    """Let Dynamo fakeify scalar meta constants created inside model code."""
    tls = getattr(fake_tensor, "fake_tensor_tls", None)
    if tls is None or not hasattr(tls, "allow_non_fake_inputs_override"):
        yield
        return
    previous = tls.allow_non_fake_inputs_override
    try:
        tls.allow_non_fake_inputs_override = True
        yield
    finally:
        tls.allow_non_fake_inputs_override = previous


@torch.library.custom_op("model_graph::expert_dispatch", mutates_args=())
def expert_dispatch(hidden_states: torch.Tensor, routing: list[torch.Tensor],
                    weights: list[torch.Tensor]) -> torch.Tensor:
    """Opaque reviewed expert boundary; numeric execution is unsupported."""
    if hidden_states.device.type != "meta":
        raise ValueError("Expert dispatch contract supports shape tensors only")
    return torch.empty_like(hidden_states)


@expert_dispatch.register_fake
def expert_dispatch_fake(hidden_states, routing, weights):
    return torch.empty_like(hidden_states)


def compiled_expert_forward(module, hidden_states, args, kwargs):
    routing = [value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
    return expert_dispatch(hidden_states, routing, list(module.parameters()))


class CompiledGraph:
    def __init__(self, model, show_floating_dtype=True):
        self.model = model
        self.show_floating_dtype = show_floating_dtype
        self.graph_module = None
        self.labels = {}
        self.cache_inputs = {}
        self.cache_outputs = {}
        self.shape_contracts = []
        self.input_nodes = {}
        self.output_nodes = {}
        self.result = None

    def trace(self, inputs, warmup_inputs=None):
        sources = {id(value): ("parameter", name) for name, value in self.model.named_parameters()}
        sources.update({id(value): ("buffer", name) for name, value in self.model.named_buffers()})

        def capture(graph_module, example_inputs):
            if self.graph_module is not None:
                raise ValueError("Expected a single complete compiler FX graph")
            self.graph_module = graph_module
            placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
            for node, value in zip(placeholders, example_inputs):
                kind, name = sources.get(id(value), ("input", node.name))
                self.labels[node.name] = f"{kind} {name}"
                self.input_nodes[id(value)] = node

            def run(*args):
                result = graph_module.forward(*args)
                output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
                nodes, _ = tree_flatten(output_node.args[0])
                values, _ = tree_flatten(result)
                self.output_nodes.update({id(value): node for value, node in zip(values, nodes)
                                          if isinstance(value, torch.Tensor) and isinstance(node, torch.fx.Node)})
                return result
            return run

        with model_shape_contracts(self.model, compiled_expert_forward) as paths, torch.no_grad(), FakeTensorMode(
            allow_non_fake_inputs=True, allow_fallback_kernels=False
        ) as fake_mode:
            self.shape_contracts = list(paths)
            fake_inputs = {name: fake_mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                           for name, value in inputs.items()}
            if warmup_inputs is not None:
                warmup = {name: fake_mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                          for name, value in warmup_inputs.items()}
                cache = output_cache(self.model(**warmup))
                self.cache_inputs = dict(cache_tensors(cache))
                if not self.cache_inputs:
                    raise ValueError("Prefill did not initialize any cache tensors")
                fake_inputs["past_key_values"] = cache
                sources.update({id(value): ("cache_input", name) for name, value in self.cache_inputs.items()})
            sources.update({id(value): ("user_input", name) for name, value in fake_inputs.items()
                            if isinstance(value, torch.Tensor)})
            compiled = torch.compile(self.model, backend=capture, fullgraph=True, dynamic=False)
            with allow_meta_constants():
                output = compiled(**fake_inputs)
            if warmup_inputs is not None:
                self.cache_outputs = dict(cache_tensors(output_cache(output)))
            self.result = output
        if self.graph_module is None:
            raise ValueError("torch.compile did not capture any model operations")
        return self

    def print(self, stream):
        from compiled_graph_format import CompiledGraphFormatter

        contracts = {path: module.graph_shape_contract for path, module in self.model.named_modules()
                     if getattr(module, "graph_shape_contract", None)}
        for path, contract in contracts.items():
            print(f"# shape_contract [{path}]: {contract}", file=stream)
        if self.shape_contracts:
            print("# Reviewed shape contracts: " + ", ".join(self.shape_contracts), file=stream)
            print("# model_graph.expert_dispatch preserves routing and expert weight dependencies; "
                  "tokens_per_expert=data-dependent; expert kernels are not traced.", file=stream)
        CompiledGraphFormatter(self).print(stream)
