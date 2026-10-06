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

from model_ops import ModuleGraph


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


def describe(value):
    """Format graph references and tensors without printing tensor contents."""
    if isinstance(value, torch.fx.Node):
        return f"%{value.name}"
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, tuple):
        parts = ", ".join(describe(item) for item in value)
        return f"({parts}{',' if len(value) == 1 else ''})"
    if isinstance(value, list):
        return "[" + ", ".join(describe(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key!r}: {describe(item)}" for key, item in value.items()
        ) + "}"
    return repr(value)


def print_graph(program, stream):
    specs = {
        spec.arg.name: spec
        for spec in program.graph_signature.input_specs
        if hasattr(spec.arg, "name")
    }

    for node in program.graph_module.graph.nodes:
        metadata = node.meta
        result = (
            describe(metadata["val"])
            if "val" in metadata
            else "?"
        )

        if node.op == "placeholder":
            spec = specs.get(node.name)
            label = "input"
            if spec is not None:
                label = spec.kind.name.lower()
                if spec.target is not None:
                    label += f" {spec.target}"
                if "val" not in metadata and hasattr(spec.arg, "value"):
                    result = describe(spec.arg.value)
            line = f"%{node.name} = {label} -> {result}"

        elif node.op == "output":
            line = f"return {describe(node.args[0])}"

        else:
            arguments = [describe(arg) for arg in node.args]
            arguments.extend(
                f"{key}={describe(value)}"
                for key, value in node.kwargs.items()
            )
            line = (
                f"%{node.name} = {node.target}"
                f"({', '.join(arguments)}) -> {result}"
            )

        print(line, file=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID or local directory")
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--seq-len", type=positive_int, default=128)
    parser.add_argument(
        "--task",
        choices=("causal-lm", "masked-lm", "encoder", "seq2seq-lm"),
        default="causal-lm",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--format", choices=("modules", "aten"), default="modules",
        help="model modules with dimensions (default), or detailed torch.export operations",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow execution of custom code from the model repository",
    )
    args = parser.parse_args()

    factories = {
        "causal-lm": transformers.AutoModelForCausalLM,
        "masked-lm": transformers.AutoModelForMaskedLM,
        "encoder": transformers.AutoModel,
        "seq2seq-lm": transformers.AutoModelForSeq2SeqLM,
    }

    try:
        with progress(f"Loading configuration for {args.model}"):
            config = transformers.AutoConfig.from_pretrained(
                args.model,
                revision=args.revision,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
            )

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
                model = factories[args.task].from_config(
                    config,
                    torch_dtype=getattr(torch, args.dtype),
                    attn_implementation="eager",
                    trust_remote_code=args.trust_remote_code,
                ).eval()

            inputs = {
                "input_ids": torch.empty(
                    (args.batch, args.seq_len), dtype=torch.long
                )
            }
            parameters = inspect.signature(model.forward).parameters

            if "position_ids" in parameters:
                inputs["position_ids"] = torch.arange(
                    args.seq_len, dtype=torch.long
                ).unsqueeze(0).expand(args.batch, -1)

            if args.task == "seq2seq-lm":
                inputs["decoder_input_ids"] = torch.empty(
                    (args.batch, args.seq_len), dtype=torch.long
                )

            if "use_cache" in parameters:
                inputs["use_cache"] = False
            if "return_dict" in parameters:
                inputs["return_dict"] = False

            if args.format == "modules":
                with progress("Recording model operations with shape-only tensors"):
                    program = ModuleGraph(model).trace(inputs)
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

    print(
        f"# {args.model}: task={args.task}, batch={args.batch}, "
        f"seq_len={args.seq_len}, dtype={args.dtype}"
    )
    print("# Inference forward pass; all tokens valid; caching disabled.")
    if args.format == "modules":
        print("# Model operations; nested steps belong to their parent module.")
        print("# Shapes only: tensor values are not computed on CPU or GPU.")
        program.print(sys.stdout)
    else:
        print_graph(program, sys.stdout)


if __name__ == "__main__":
    main()