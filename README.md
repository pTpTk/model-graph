# model-graph

Print a Hugging Face model's execution graph as named model operations, with
input, weight parameter, buffer, and output tensor dimensions and dtypes.
Only the configuration is downloaded. Weights and activations have no storage:
meta/fake tensors propagate shapes without computing tensor values on CPU or GPU.
Python model control flow and shape bookkeeping still use the CPU.

```sh
python -m pip install -e .
model-graph username/modelname --batch 32 --seq-len 8192
```

In this workspace, activate the installed environment first:

```sh
source .venv/bin/activate
model-graph Qwen/Qwen3.8-27B --batch 32 --seq-len 8192 > graph.txt
```

Replace `username/modelname` with a real Hugging Face model ID, or use a local
configuration directory. The default task is `causal-lm`; `--task` also accepts
`encoder`, `masked-lm`, and `seq2seq-lm`. Sequence-to-sequence models use the
specified sequence length for both encoder and decoder inputs.

`--dtype auto` is the default. The tool reads `dtype` or the legacy `torch_dtype`
from the model configuration, including a nested text configuration. For example,
Qwen/Qwen3.8-27B declares `bfloat16`; running on a CPU host does not change this.
`--dtype float32`, `--dtype float16`, and `--dtype bfloat16` provide explicit overrides.
For models without quantization metadata, tensor dtypes come from shape
propagation, so explicit casts can still produce float32 intermediates or buffers.
If the configuration does not declare a floating dtype, floating dtype annotations
are omitted; known integer and boolean dtypes remain visible.

For models with quantization or compression metadata, floating dtype annotations
are always omitted, including with an explicit `--dtype` override. For example,
Qwen/Qwen3.8-27B-fp8 declares a bfloat16 default alongside an FP8 recipe; that default
does not describe the FP8 weights. Shape recording constructs ordinary modules and
does not load the quantized checkpoint or instantiate quantization-specific kernels
and scale tensors. Its inferred dtypes cannot establish quantized checkpoint or
runtime dtypes. `--dtype` only controls shape recording in this case. Graph output
states this limitation. Quantization metadata is checked on parent and nested configs.

Dtype annotations for models without quantization describe config-based construction
and its casts, not independently verified checkpoint storage or runtime autocast settings.

`--phase prefill` is the default and records the full prompt with cache output
disabled. `--phase decode` records one new token against a populated cache:

```sh
model-graph username/modelname --batch 32 --seq-len 8192 --phase prefill > prefill.txt
model-graph username/modelname --batch 32 --seq-len 8192 --phase decode > decode.txt
```

For decode, `--seq-len` is the cached context length **before** the new token.
Input IDs and positions have shape `(32, 1)`. Standard full-attention KV inputs
hold 8192 positions and updated KV holds 8193. Sliding, compressed and recurrent
layers retain their native layouts. Cache tensors are listed as graph inputs and
updated outputs. An unrecorded shape-only prefill initializes the cache, so decode
generation may still take time; no prefill operations appear in the decode graph.
Decode currently supports the causal-LM task and module output format.

Reviewed native packed expert modules use a shape contract for expert dispatch.
The graph includes routing and expert parameter dimensions, but marks per-expert
token allocation as data-dependent. The contract returns the exact aggregate
boundary shape without evaluating routing decisions. This applies to both phases.
The GLM5 Next and Qwen4 Exp sparse indexers also use reviewed boundary contracts
for their fixed-width index/mask outputs and native cache layouts. Their selected
token positions require tensor values and are explicitly marked data-dependent.

The model collection is organized under [results/README.md](results/README.md),
with separate `prefill/` and `decode/` graphs, pinned configs, metadata and logs.
Use `python generate_results.py` to generate or retry the catalog.
Kimi K3 uses a local text adapter derived from its reviewed official source. It
retains SiTU, attention residuals, latent MoE, no-position MLA and output gating.
Its Kimi Delta recurrence uses a declared shape contract with the native recurrent
state dimensions; the graph omits fused backend kernels. Expert weights are packed
across the expert axis. Source revision and hashes are in
[results/kimi-k3-source.json](results/kimi-k3-source.json). This adapter supports
all-valid causal inference in module format and is intended only for shape graphs.
The collection index lists any remaining unsupported architectures.

The default `--format modules` output lists numbered steps with the executed
module's path and class: embedding, attention (including linear attention),
projections, normalization, MLP, activation, and output projection. Indentation
shows parent/child module calls. Composite steps summarize their nested steps;
they are not extra independent computations. Tensor references such as `%s9_0`
connect producers to consumers, and each reference includes its shape and dtype.
Repeated calls to a shared module receive distinct step numbers.

Functional work between module calls, including residual addition, elementwise
gating, and attention cores, is grouped into steps with mathematical operation
names and boundary tensor dimensions. Structural transformations appear when
needed to connect modules. These groups omit internal intermediate tensors and
do not list every chunk-loop iteration or backend kernel. Some modules invoke
functional operations directly, so a registered child module is only listed as
a module step if its `forward` actually runs; its parameters still appear in the
graph and the containing functional step.

`--format aten` retains the detailed `torch.export` operator graph when needed:

```sh
model-graph username/modelname --batch 32 --seq-len 8192 --format aten
```

Both output formats use eager attention and assume all tokens are valid.
Prefill disables cache output; decode enables caching.
Stage messages and elapsed-time updates every 30 seconds go to stderr; the graph
alone goes to stdout. Default module recording avoids export and graph lowering,
but still follows Python sequence-chunk loops for shape propagation. Hybrid
models can therefore take longer at large sequence lengths. The optional ATen
export can take several minutes or longer and consume much more graph memory.
Elapsed-time updates indicate a stage is still running, not a completion percentage.

A model must support shape propagation with meta/fake tensors. Data-dependent
operations or missing shape implementations can prevent recording; numeric CPU
fallback kernels are disabled in module mode. Sequence lengths must fit the
model architecture. This tool describes operations and shapes, not numerical
outputs or backend performance.

Run offline integration and graph-dependency tests with:

```sh
python -m unittest -v test_model_graph
```
