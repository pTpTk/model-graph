#!/usr/bin/env python3
"""Print a Hugging Face model's execution graph without loading weights."""

import argparse
from contextlib import contextmanager
import inspect
import sys
import threading
import time

import torch
import transformers

from model_ops import ModuleGraph, tensor_description


@contextmanager
def progress(label, stream=None, interval=30):
    """Report a slow stage on stderr without polluting the graph on stdout."""
    if stream is None:
        stream = sys.stderr
    started = time.monotonic()
    stopped = threading.Event()

    def report(message):
        print(f"model-graph: {message}", file=stream, flush=True)

    def heartbeat():
        while not stopped.wait(interval):
            report(f"{label}: still running ({time.monotonic() - started:.0f}s elapsed)")

    report(label)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
    report(f"{label}: done ({time.monotonic() - started:.1f}s)")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def has_quantization_metadata(config):
    """Detect recipes on the parent config or any nested model configuration."""
    def contains_recipe(metadata):
        if not isinstance(metadata, dict):
            return False
        if any(metadata.get(key) for key in (
            "quantization_config", "compression_config", "load_in_4bit", "load_in_8bit",
        )):
            return True
        return any(contains_recipe(value) for value in metadata.values() if isinstance(value, dict))

    return contains_recipe(config.to_dict())


def resolve_dtype(config, requested="auto"):
    """Resolve a floating dtype from explicit input or declared config metadata."""
    if requested != "auto":
        return getattr(torch, requested), "override"
    candidates = [("config", config)]
    text_config = config.get_text_config()
    if text_config is not config:
        candidates.append(("text_config", text_config))
    for source, candidate in candidates:
        metadata = candidate.to_dict()
        value = metadata.get("dtype") or metadata.get("torch_dtype")
        if value is None:
            continue
        dtype = value if isinstance(value, torch.dtype) else getattr(torch, str(value).removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"Unsupported floating dtype {value!r} declared in {source}")
        return dtype, source
    return None, None


def describe(value, show_floating_dtype=True):
    """Format graph references and tensors without printing tensor contents."""
    if isinstance(value, torch.fx.Node):
        return f"%{value.name}"
    if isinstance(value, torch.Tensor):
        return tensor_description(tuple(value.shape), value.dtype, show_floating_dtype)
    if isinstance(value, tuple):
        parts = ", ".join(describe(item, show_floating_dtype) for item in value)
        return f"({parts}{',' if len(value) == 1 else ''})"
    if isinstance(value, list):
        return "[" + ", ".join(describe(item, show_floating_dtype) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key!r}: {describe(item, show_floating_dtype)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, torch.dtype) and value.is_floating_point and not show_floating_dtype:
        return "unspecified_floating_dtype"
    return repr(value)


def print_graph(program, stream, show_floating_dtype=True):
    labels = {}
    constants = {}
    for spec in program.graph_signature.input_specs:
        if hasattr(spec.arg, "name"):
            label = spec.kind.name.lower()
            if spec.target is not None:
                label += f" {spec.target}"
            labels[spec.arg.name] = label
            if hasattr(spec.arg, "value"):
                constants[spec.arg.name] = spec.arg.value
    print_fx_graph(program.graph_module, stream, show_floating_dtype, labels, constants)


def print_fx_graph(graph_module, stream, show_floating_dtype=True, labels=None, constants=None):
    """Print export or compiler FX nodes, including their shape metadata."""
    labels = labels or {}
    constants = constants or {}

    for node in graph_module.graph.nodes:
        metadata = node.meta
        value = metadata.get("val", metadata.get("example_value", constants.get(node.name)))
        result = (
            describe(value, show_floating_dtype)
            if "val" in metadata or "example_value" in metadata or node.name in constants
            else "?"
        )

        if node.op == "placeholder":
            label = labels.get(node.name, "input")
            line = f"%{node.name} = {label} -> {result}"

        elif node.op == "output":
            line = f"return {describe(node.args[0], show_floating_dtype)}"

        else:
            arguments = [describe(arg, show_floating_dtype) for arg in node.args]
            arguments.extend(
                f"{key}={describe(value, show_floating_dtype)}"
                for key, value in node.kwargs.items()
            )
            target = node.target
            if callable(target) and not isinstance(target, torch._ops.OpOverload):
                target = f"{target.__module__}.{target.__name__}"
            line = (
                f"%{node.name} = {target}"
                f"({', '.join(arguments)}) -> {result}"
            )

        print(line, file=stream)


def load_model_config(args):
    """Use native config support, with explicit text/architecture fallbacks."""
    options = dict(revision=args.revision, local_files_only=args.local_files_only,
                   trust_remote_code=args.trust_remote_code)
    try:
        return transformers.AutoConfig.from_pretrained(args.model, **options), None
    except ValueError as original_error:
        raw, _ = transformers.PretrainedConfig.get_config_dict(args.model, **options)
        if raw.get("model_type") in ("kimi_k3", "kimi_k3_text_graph"):
            from kimi_k3_graph import KimiK3TextConfig
            config = KimiK3TextConfig.from_dict(raw.get("text_config", raw))
            if raw.get("quantization_config"):
                config.quantization_config = raw["quantization_config"]
            return config, "Reviewed local Kimi K3 text adapter; vision operations excluded."
        # Some repositories still declare remote AutoConfig code even after the
        # same model_type has gained native Transformers support.
        try:
            values = dict(raw)
            model_type = values.pop("model_type")
            config = transformers.AutoConfig.for_model(model_type, **values)
        except (ValueError, KeyError):
            pass
        else:
            return config, f"Native configuration for declared model_type={model_type}."
        text = raw.get("text_config")
        if isinstance(text, dict) and text.get("model_type"):
            values = dict(text)
            model_type = values.pop("model_type")
            try:
                config = transformers.AutoConfig.for_model(model_type, **values)
            except (ValueError, KeyError):
                pass
            else:
                if raw.get("quantization_config"):
                    config.quantization_config = raw["quantization_config"]
                return config, f"Text-only submodel from text_config ({model_type}); vision operations excluded."
        for architecture in raw.get("architectures", []):
            model_class = getattr(transformers, architecture, None)
            config_class = getattr(model_class, "config_class", None)
            if config_class is not None:
                config = config_class.from_dict(raw)
                return config, f"Native declared architecture {architecture}; source model_type={raw.get('model_type')}."
        raise original_error


class NativeTextCausalLM(torch.nn.Module):
    """Text-only forward using a native text model and its ordinary output head."""

    def __init__(self, config, text_model_class, dtype_options):
        super().__init__()
        self.config = config
        self.model = text_model_class._from_config(
            config, attn_implementation="eager", **dtype_options
        )
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False,
            dtype=next(self.model.parameters()).dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, position_ids=None, past_key_values=None,
                use_cache=False, return_dict=True):
        output = self.model(input_ids=input_ids, position_ids=position_ids,
                            past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(output.last_hidden_state)
        if return_dict:
            return transformers.modeling_outputs.CausalLMOutputWithPast(
                logits=logits, past_key_values=output.past_key_values
            )
        return (logits, output.past_key_values) if use_cache else (logits,)


def construct_model(config, factory, args, dtype_options):
    """Select native text support when a multimodal parent has no causal mapping."""
    text = config.get_text_config()
    if args.compile_backend == "cuda-serving":
        if args.task != "causal-lm" or text.model_type != "gpt_oss":
            raise ValueError("--compile-backend cuda-serving currently supports gpt-oss causal-LM configs")
        from cuda_serving_graph import CudaServingForCausalLM
        model = CudaServingForCausalLM(text, dtype_options.get("dtype", dtype_options.get("torch_dtype")))
        return model, "Config-only CUDA serving adapter: packed QKV, fused residual/RMSNorm and native operation shape contracts; logical KV dimensions; no measured kernel launches or quantized storage layouts."
    if type(config).__name__ == "KimiK3TextConfig":
        if args.task != "causal-lm" or args.format != "modules":
            raise ValueError("Kimi K3 adapter requires --task causal-lm and --format modules")
        from kimi_k3_graph import KimiK3ForCausalLM
        model = KimiK3ForCausalLM(config, dtype=dtype_options.get("dtype", dtype_options.get("torch_dtype")))
        return model, "Kimi K3: SiTU, latent MoE, attention residuals, no-position MLA and output gating; KDA uses an explicit shape contract. Expert weights packed across expert axis."
    if text.model_type == "kimi_linear" and any((
        getattr(text, "hidden_act", None) == "situ",
        getattr(text, "attn_res_block_size", None),
        getattr(text, "routed_expert_hidden_size", None),
        getattr(text, "mla_use_nope", False),
        getattr(text, "mla_use_output_gate", False),
    )):
        raise ValueError(
            "Kimi K3 requires SiTU, attention residuals, latent MoE, and modified MLA; "
            "the installed native KimiLinear implementation does not implement these features. "
            "Its config cannot be traced as ordinary Kimi Linear."
        )
    options = dict(**dtype_options, attn_implementation="eager",
                   trust_remote_code=args.trust_remote_code)
    if args.task != "causal-lm" or args.trust_remote_code or type(config) in factory._model_mapping:
        return factory.from_config(config, **options), None
    if type(text) in factory._model_mapping:
        if getattr(config, "quantization_config", None):
            text.quantization_config = config.quantization_config
        return factory.from_config(text, **options), "Native text-only causal model; vision operations excluded."
    native_text_classes = {
        "glm5_next_text": ("glm5_next", "Glm5NextTextModel"),
    }
    if text.model_type in native_text_classes:
        package, name = native_text_classes[text.model_type]
        module = __import__(f"transformers.models.{package}.modeling_{package}", fromlist=[name])
        model = NativeTextCausalLM(text, getattr(module, name), dtype_options)
        return model, f"Native {name} with output projection; text-only, vision and auxiliary MTP operations excluded."
    return factory.from_config(config, **options), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID or local directory")
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--seq-len", type=positive_int, default=128,
                        help="prompt length for prefill; cached context length before decode")
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill",
                        help="full prompt or one token against a populated cache")
    parser.add_argument(
        "--task",
        choices=("causal-lm", "masked-lm", "encoder", "seq2seq-lm"),
        default="causal-lm",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="shape-recording dtype: configuration default or explicit override; "
             "quantized models omit floating dtype annotations",
    )
    parser.add_argument(
        "--format", choices=("modules", "aten"), default="modules",
        help="model modules with dimensions (default), or detailed torch.export operations",
    )
    parser.add_argument("--compile", action="store_true",
                        help="capture a full Transformers torch.compile FX graph before backend "
                             "lowering; shape-only, not CUDA kernel launches")
    parser.add_argument("--compile-backend", choices=("fx", "cuda-serving"), default="fx",
                        help="Transformers FX (default), or config-only gpt-oss CUDA serving operations")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow execution of custom code from the model repository",
    )
    args = parser.parse_args()
    if args.compile_backend != "fx" and not args.compile:
        parser.error("--compile-backend requires --compile")
    if args.compile and args.format != "modules":
        parser.error("--compile selects compiler FX output and cannot be combined with --format aten")
    if args.phase == "decode" and (args.task != "causal-lm" or args.format != "modules"):
        parser.error("decode currently requires --task causal-lm and --format modules")

    factories = {
        "causal-lm": transformers.AutoModelForCausalLM,
        "masked-lm": transformers.AutoModelForMaskedLM,
        "encoder": transformers.AutoModel,
        "seq2seq-lm": transformers.AutoModelForSeq2SeqLM,
    }

    try:
        with progress(f"Loading configuration for {args.model}"):
            config, config_note = load_model_config(args)

        resolved_dtype, dtype_source = resolve_dtype(config, args.dtype)
        quantized = has_quantization_metadata(config)
        show_floating_dtype = resolved_dtype is not None and not quantized
        text_config = config.get_text_config()
        linear_layers = (getattr(text_config, "layer_types", None) or []).count("linear_attention")
        if linear_layers and args.format == "aten":
            print(
                f"model-graph: {linear_layers} linear-attention layers; "
                f"export may expand sequence-dependent loops at seq_len={args.seq_len} "
                "and take several minutes or longer.",
                file=sys.stderr, flush=True,
            )

        # Eager attention exposes attention operations in the graph.
        # Meta tensors retain shape and dtype without allocating storage.
        with torch.device("meta"):
            with progress("Constructing model on meta device (no weight storage)"):
                dtype_options = {}
                if resolved_dtype is not None:
                    # v4 uses torch_dtype; v5 renamed the argument to dtype.
                    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
                    dtype_options[key] = resolved_dtype
                model, model_note = construct_model(config, factories[args.task], args, dtype_options)
                model.eval()

            query_len = args.seq_len if args.phase == "prefill" else 1
            position_start = 0 if args.phase == "prefill" else args.seq_len
            inputs = {
                "input_ids": torch.empty((args.batch, query_len), dtype=torch.long)
            }
            parameters = inspect.signature(model.forward).parameters

            if "position_ids" in parameters:
                inputs["position_ids"] = torch.arange(
                    position_start, position_start + query_len, dtype=torch.long
                ).unsqueeze(0).expand(args.batch, -1)

            if args.task == "seq2seq-lm":
                inputs["decoder_input_ids"] = torch.empty(
                    (args.batch, args.seq_len), dtype=torch.long
                )

            if "use_cache" in parameters:
                inputs["use_cache"] = args.phase == "decode"
            if "return_dict" in parameters:
                inputs["return_dict"] = False

            warmup_inputs = None
            if args.phase == "decode":
                warmup_inputs = dict(inputs)
                warmup_inputs["input_ids"] = torch.empty((args.batch, args.seq_len), dtype=torch.long)
                warmup_inputs["use_cache"] = True
                if "return_dict" in parameters:
                    warmup_inputs["return_dict"] = True
                if "position_ids" in parameters:
                    warmup_inputs["position_ids"] = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0).expand(args.batch, -1)
                if "cache_position" in parameters:
                    warmup_inputs["cache_position"] = torch.arange(args.seq_len, dtype=torch.long)
                    inputs["cache_position"] = torch.arange(args.seq_len, args.seq_len + 1, dtype=torch.long)

            if args.compile:
                from compiled_graph import CompiledGraph
                with progress("Capturing torch.compile FX graph with shape-only tensors"):
                    program = CompiledGraph(model, show_floating_dtype).trace(inputs, warmup_inputs)
            elif args.format == "modules":
                stage = "Preparing cache shapes and recording one-token decode" if args.phase == "decode" else "Recording model operations with shape-only tensors"
                with progress(stage):
                    program = ModuleGraph(model, show_floating_dtype=show_floating_dtype).trace(inputs, warmup_inputs)
            else:
                with progress("Exporting execution graph"), torch.no_grad():
                    program = torch.export.export(
                        model, args=(), kwargs=inputs, strict=False
                    )

    except Exception as error:
        parser.exit(
            1,
            f"model-graph: {type(error).__name__}: {error}\n"
            "This model must support shape propagation with meta/fake tensors; "
            "data-dependent operations can prevent tracing.\n",
        )

    dtype_summary = (
        f", dtype={str(resolved_dtype).removeprefix('torch.')} (source={dtype_source})"
        if show_floating_dtype else ""
    )
    print(
        f"# {args.model}: task={args.task}, batch={args.batch}, "
        f"seq_len={args.seq_len}, phase={args.phase}{dtype_summary}"
    )
    if config_note:
        print(f"# {config_note}")
    if model_note:
        print(f"# {model_note}")
    if quantized:
        print("# Quantization metadata present: floating dtype annotations omitted.")
        print("# Config-only shape recording does not establish checkpoint or quantized runtime dtypes.")
        print("# Model operations shown before quantization; quantization-specific kernels and scale tensors are not represented.")
    elif not show_floating_dtype:
        print("# Configuration does not declare a floating dtype; floating dtype annotations omitted.")
    if args.phase == "decode":
        print(f"# Decode: query_tokens=1, past_tokens={args.seq_len}, total_tokens={args.seq_len + 1}; caching enabled.")
        print("# Cache prepared by shape-only prefill; prefill operations are excluded from this graph.")
    else:
        print(f"# Prefill: query_tokens={args.seq_len}, past_tokens=0; cache output disabled.")
    print("# Inference forward pass; all tokens valid.")
    if args.compile:
        print("# torch.compile FX graph; fullgraph=True, dynamic=False; no backend kernel lowering.")
        if args.compile_backend == "cuda-serving":
            print("# CUDA serving operation contracts: packed QKV and fused native stages; independent of host device.")
            print("# Native contracts may cover multiple CUDA launches; this is not a measured kernel graph.")
            print("# KV cache shapes describe logical active tokens, not an allocated paged cache pool.")
        else:
            print("# Source runtime: Hugging Face Transformers; graph level: Dynamo FX before backend lowering.")
            print("# Numbered operation steps are FX nodes, not CUDA kernel launches; CUDA fusion is not captured.")
        print("# Shapes only: tensor values are not computed on CPU or GPU.")
        program.print(sys.stdout)
    elif args.format == "modules":
        print("# Model operations; nested steps belong to their parent module.")
        print("# Shapes only: tensor values are not computed on CPU or GPU.")
        program.print(sys.stdout)
    else:
        print_graph(program, sys.stdout, show_floating_dtype)


if __name__ == "__main__":
    main()
