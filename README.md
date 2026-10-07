# model-graph

Print a Hugging Face model's execution graph as named model operations, with
input, weight parameter, buffer, and output tensor dimensions and dtypes.
Only the configuration is downloaded. Weights and activations have no storage:
meta/fake tensors propagate shapes without computing tensor values on CPU or GPU.
Python model control flow and shape bookkeeping still use the CPU.

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
model-graph username/modelname --batch 32 --seq-len 8192
```

Python **3.10 or newer** is required. Runtime dependencies are PyTorch (`torch>=2.6`)
and Hugging Face Transformers (`transformers>=4.45,<6`), listed in
[requirements.txt](requirements.txt) and package metadata. Newer catalog architectures
require a Transformers release that implements them; the recorded serving run used
Torch 2.14.1 and Transformers 5.19.0. Tests use the standard-library `unittest` runner.

If using a virtual environment, activate it first:

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

Add `--compile` to capture the complete FX graph that `torch.compile` sends to
its backend, including tensor dimensions and dtypes:

```sh
model-graph openai/gpt-oss-120b --batch 32 --seq-len 8192 --phase decode --compile > compiled-decode.txt
# Use the pinned local config for an offline run:
model-graph results/configs/openai--gpt-oss-120b --batch 32 --seq-len 8192 --phase decode --compile --local-files-only > compiled-decode.txt
```

This flag prints compiler FX operations in the same numbered, indented style as
the original module graph, with short tensor references and dimensions on every
dependency. Module steps summarize their nested compiler operations. Every step
has `stages=[...]`, listing its enclosing model execution stages. Operation steps
also name their specific stage, including attention mask preparation, normalization,
Q/K/V projections, rotary position encoding, KV cache updates, attention core,
residuals, routing, expert dispatch, final normalization and output projection.
Stage attribution uses the compiler's module stack and source function metadata;
operations outside a named stage are labeled `Functional` under their model scope.

`--compile` cannot be combined with `--format aten`.
Capture uses `fullgraph=True` and `dynamic=False`: graph breaks
fail with a diagnostic instead of producing a partial graph. The backend records
FX and propagates shapes without lowering or executing CPU/GPU kernels. Decode
captures only the new token; cache preparation stays outside compilation. Input
and updated cache states use the same tensor references as the operation steps.
Reviewed expert dispatch boundaries appear as opaque `model_graph.expert_dispatch`
operations with routing and weight dependencies; per-expert token counts remain
data-dependent. Quantized configurations still omit floating dtype annotations.
Models must also support full-graph Dynamo tracing for this option.
The retained gpt-oss serving decode graph and run metadata are linked in the
[serving results index](#cuda-serving-decode-estimates).

`--compile` describes the Hugging Face Transformers implementation at the Dynamo
FX level. Its numbered operations are not CUDA kernel launches. For example,
Transformers gpt-oss calls separate Q, K and V linear modules, while SGLang's
gpt-oss implementation calls one packed `QKVParallelLinear` and splits its output.
Changing tensor devices or grouping the three displayed steps would not reproduce
that serving runtime's kernel graph. CUDA kernel attribution requires the actual
serving implementation and its selected quantization, attention/MoE backends,
tensor-parallel layout, GPU, and compilation settings. A kernel can cover multiple
model stages, and a model operation can launch multiple kernels. Matching a CUDA
profile requires backend-generated code and/or a CUDA trace from that same run.
The shape-only expert contracts here also omit the serving runtime's MoE kernels.

For a host-independent gpt-oss serving operation graph, use:

```sh
model-graph results/configs/openai--gpt-oss-120b --batch 32 --seq-len 8192 --phase decode --compile --compile-backend cuda-serving --local-files-only > cuda-serving-decode.txt
```

This mode constructs a reviewed config-only serving adapter with one packed QKV
projection per layer. At tensor-parallel size one, gpt-oss-120b projects 2880
features to 5120, then splits Q/K/V into widths 4096/512/512. It uses the serving
residual stream with fused add/RMSNorm and explicit native RoPE/cache, attention,
top-k/softmax and expert boundaries. No installed serving engine, target GPU or
historical benchmark version is required. The adapter follows the common serving
structure documented in the SGLang gpt-oss implementation.

`stages=[...]` identifies the enclosing stages and `covers=[...]` identifies
multiple stages belonging to the same operation. For example, the packed
projection covers QueryProjection, KeyProjection and ValueProjection; fused norm
covers ResidualAdd and Normalization. These stages are not separate kernel steps.
Native operations are declared shape contracts: the graph does not execute or
lower their CUDA implementations, infer backend kernel names, or measure launch
counts. Logical KV spans all active tokens; a sliding layer selects its attention
window inside the native boundary. Physical paged pool capacity, native cache
packing, quantization scales and data-dependent expert dispatch are not inferred.
This mode currently supports gpt-oss causal-LM prefill and decode. The retained
[decode graph](results/compiled/cuda-serving/decode/openai--gpt-oss-120b.txt),
[run metadata](results/compiled/cuda-serving/gpt-oss-decode.json) and
[stderr log](results/compiled/cuda-serving/logs/openai--gpt-oss-120b.decode.log)
document the config, source hashes, contracts and validation counts.

Reviewed native packed expert modules use a shape contract for expert dispatch.
The graph includes routing and expert parameter dimensions, but marks per-expert
token allocation as data-dependent. The contract returns the exact aggregate
boundary shape without evaluating routing decisions. This applies to both phases.
The GLM5 Next and Qwen4 Exp sparse indexers also use reviewed boundary contracts
for their fixed-width index/mask outputs and native cache layouts. Their selected
token positions require tensor values and are explicitly marked data-dependent.

The model collection is listed in the [results index](#model-execution-graphs),
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

All output modes use eager attention and assume all tokens are valid.
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
python -m unittest -v test_model_graph test_serving_results
```

The [CUDA serving decode estimates](#cuda-serving-decode-estimates) cover all 22 models
with successful decode traces. Run `python generate_serving_results.py` to regenerate
the estimates and index. DeepSeek V4.1 remains unsupported.

<!-- model-results:start -->
## Model execution graphs

Text-only inference graphs at batch **32** and prompt/cached-context length **8192**.

- `prefill/`: all 8192 prompt tokens, no preceding cache; cache output disabled.
- `decode/`: one new token after a shape-only 8192-token prefill; caching enabled.
  Standard full-attention KV grows to 8193 positions. Sliding/compressed and recurrent caches follow their native layouts.
- `configs/`: original configuration JSON, pinned to the revision in `catalog.json`.
- `manifest.json`: per-phase commands, source revision, tool/output hashes, timing and status.
- `logs/`: stderr diagnostics, including unsupported models and tracing failures.

MoE expert dispatch and GLM5 Next / Qwen4 Exp sparse indexers use reviewed native boundary shape contracts.
Routing, shared experts, projections and other fixed-shape operations remain visible.
Per-expert token counts and sparse-indexer selected positions require tensor values and are explicitly labeled data-dependent.
Floating dtype annotations are omitted for quantized configurations.

Kimi K3 uses the reviewed local text adapter in `kimi_k3_graph.py`, preserving SiTU, latent MoE, attention residuals and modified MLA.
Its KDA recurrence is an explicit boundary shape contract. [Source revision and hashes](results/kimi-k3-source.json) document the adaptation.

This is a representative catalog of public general-purpose model architecture families, not a benchmark ranking.
A failed entry has no substitute or fabricated execution graph.

| Model | Architecture | Prefill | Decode |
| --- | --- | --- | --- |
| [zai-org/GLM-4.5-Base](https://huggingface.co/zai-org/GLM-4.5-Base) | `glm4_moe` | [graph](results/prefill/zai-org--GLM-4.5-Base.txt) (2566 steps) | [graph](results/decode/zai-org--GLM-4.5-Base.txt) (2566 steps) |
| [zai-org/GLM-4.5-Air-Base](https://huggingface.co/zai-org/GLM-4.5-Air-Base) | `glm4_moe` | [graph](results/prefill/zai-org--GLM-4.5-Air-Base.txt) (1152 steps) | [graph](results/decode/zai-org--GLM-4.5-Air-Base.txt) (1152 steps) |
| [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) | `glm_moe_dsa` | [graph](results/prefill/zai-org--GLM-5.txt) (2954 steps) | [graph](results/decode/zai-org--GLM-5.txt) (2954 steps) |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | `glm_moe_dsa` | [graph](results/prefill/zai-org--GLM-5.3.txt) (2441 steps) | [graph](results/decode/zai-org--GLM-5.3.txt) (2441 steps) |
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `glm5_next_text` | [graph](results/prefill/zai-org--GLM-5.3-Flash.txt) (1925 steps) | [graph](results/decode/zai-org--GLM-5.3-Flash.txt) (1925 steps) |
| [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | `kimi_linear` | [graph](results/prefill/moonshotai--Kimi-K3.txt) (4063 steps) | [graph](results/decode/moonshotai--Kimi-K3.txt) (4063 steps) |
| [moonshotai/Kimi-K2-Base](https://huggingface.co/moonshotai/Kimi-K2-Base) | `kimi_k2` | [graph](results/prefill/moonshotai--Kimi-K2-Base.txt) (1771 steps) | [graph](results/decode/moonshotai--Kimi-K2-Base.txt) (1771 steps) |
| [moonshotai/Kimi-Linear-48B-A3B-Base](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) | `kimi_linear` | [graph](results/prefill/moonshotai--Kimi-Linear-48B-A3B-Base.txt) (870 steps) | [graph](results/decode/moonshotai--Kimi-Linear-48B-A3B-Base.txt) (870 steps) |
| [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | `qwen3_5_moe_text` | [graph](results/prefill/Qwen--Qwen3.8-2.4T-A95B.txt) (2607 steps) | [graph](results/decode/Qwen--Qwen3.8-2.4T-A95B.txt) (2607 steps) |
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `qwen4_exp_text` | [graph](results/prefill/Qwen--Qwen3.8-Flash-Next.txt) (2082 steps) | [graph](results/decode/Qwen--Qwen3.8-Flash-Next.txt) (2082 steps) |
| [deepseek-ai/DeepSeek-V3-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) | `deepseek_v3` | [graph](results/prefill/deepseek-ai--DeepSeek-V3-Base.txt) (1759 steps) | [graph](results/decode/deepseek-ai--DeepSeek-V3-Base.txt) (1759 steps) |
| [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | `deepseek_v32` | [graph](results/prefill/deepseek-ai--DeepSeek-V3.2.txt) (2308 steps) | [graph](results/decode/deepseek-ai--DeepSeek-V3.2.txt) (2308 steps) |
| [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) | `deepseek_v4` | [graph](results/prefill/deepseek-ai--DeepSeek-V4-Pro-Base.txt) (3695 steps) | [graph](results/decode/deepseek-ai--DeepSeek-V4-Pro-Base.txt) (3331 steps) |
| [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) | `deepseek_v4` | [graph](results/prefill/deepseek-ai--DeepSeek-V4-Flash-Base.txt) (2588 steps) | [graph](results/decode/deepseek-ai--DeepSeek-V4-Flash-Base.txt) (2340 steps) |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `deepseek_v41_text` | unsupported ([log](results/logs/deepseek-ai--DeepSeek-V4.1-Flash.prefill.log)) | unsupported ([log](results/logs/deepseek-ai--DeepSeek-V4.1-Flash.decode.log)) |
| [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | `minimax_m3_vl` | [graph](results/prefill/MiniMaxAI--MiniMax-M3.txt) (2123 steps) | [graph](results/decode/MiniMaxAI--MiniMax-M3.txt) (2123 steps) |
| [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) | `minimax_m2` | [graph](results/prefill/MiniMaxAI--MiniMax-M2.7.txt) (1124 steps) | [graph](results/decode/MiniMaxAI--MiniMax-M2.7.txt) (1124 steps) |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | `gpt_oss` | [graph](results/prefill/openai--gpt-oss-120b.txt) (656 steps) | [graph](results/decode/openai--gpt-oss-120b.txt) (656 steps) |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `qwen3_5_text` | [graph](results/prefill/Qwen--Qwen3.8-27B.txt) (1432 steps) | [graph](results/decode/Qwen--Qwen3.8-27B.txt) (1432 steps) |
| [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `qwen3_5_text` | [graph](results/prefill/Qwen--Qwen3.8-27B-FP8.txt) (1432 steps) | [graph](results/decode/Qwen--Qwen3.8-27B-FP8.txt) (1432 steps) |
| [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | `qwen3_5_moe_text` | [graph](results/prefill/Qwen--Qwen3.5-397B-A17B.txt) (1703 steps) | [graph](results/decode/Qwen--Qwen3.5-397B-A17B.txt) (1703 steps) |
| [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | `qwen3_next` | [graph](results/prefill/Qwen--Qwen3-Next-80B-A3B-Instruct.txt) (1220 steps) | [graph](results/decode/Qwen--Qwen3-Next-80B-A3B-Instruct.txt) (1220 steps) |
| [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) | `qwen3_moe` | [graph](results/prefill/Qwen--Qwen3-235B-A22B-Instruct-2507.txt) (2076 steps) | [graph](results/decode/Qwen--Qwen3-235B-A22B-Instruct-2507.txt) (2076 steps) |

### Unsupported or incomplete entries

- **deepseek-ai/DeepSeek-V4.1-Flash**: Installed Transformers lacks deepseek_v41 / deepseek_v41_text support. Official inference uses custom CUDA kernels and an additional Engram/vision architecture; no faithful shape adapter is provided.

Regenerate with `python generate_results.py`; use `--model OWNER/NAME` to select an entry.
Successful outputs are retained unless `--force` is supplied. Failed entries are retried.
<!-- model-results:end -->

<!-- serving-results:start -->
## CUDA serving decode estimates

Batch **32**, cached context **8192**, query **1 token**.

These graphs estimate serving operation boundaries, not captured CUDA launches. Each step carries model execution stages.
The gpt-oss graph uses the existing config-only packed serving adapter and torch.compile FX capture.
Other graphs are transformed from successful recorded decode traces: independent Q/K/V linears sharing an input are packed;
adjacent residual add and RMSNorm are grouped; attention and recurrence are marked as native boundary estimates.
Source IDs, tensor dimensions, dependencies and model-specific MLA, sparse, hybrid and expert structures are preserved.
Logical intermediate tensors and checkpoint weight slices may remain visible inside estimated boundaries.
Unmatched operations retain source-level grouping. Stage attribution follows module paths and operation metadata.
No particular vLLM/SGLang implementation, exact fusion, quantization layout, physical cache allocation or launch count is inferred.

Regenerate with `python generate_serving_results.py`. [Manifest](results/compiled/cuda-serving/manifest.json) records input/output hashes and inference rules.

| Model | Decode | Method |
| --- | --- | --- |
| zai-org/GLM-4.5-Base | [graph](results/compiled/cuda-serving/decode/zai-org--GLM-4.5-Base.txt) | rule-based source trace estimate |
| zai-org/GLM-4.5-Air-Base | [graph](results/compiled/cuda-serving/decode/zai-org--GLM-4.5-Air-Base.txt) | rule-based source trace estimate |
| zai-org/GLM-5 | [graph](results/compiled/cuda-serving/decode/zai-org--GLM-5.txt) | rule-based source trace estimate |
| zai-org/GLM-5.3 | [graph](results/compiled/cuda-serving/decode/zai-org--GLM-5.3.txt) | rule-based source trace estimate |
| zai-org/GLM-5.3-Flash | [graph](results/compiled/cuda-serving/decode/zai-org--GLM-5.3-Flash.txt) | rule-based source trace estimate |
| moonshotai/Kimi-K3 | [graph](results/compiled/cuda-serving/decode/moonshotai--Kimi-K3.txt) | rule-based source trace estimate |
| moonshotai/Kimi-K2-Base | [graph](results/compiled/cuda-serving/decode/moonshotai--Kimi-K2-Base.txt) | rule-based source trace estimate |
| moonshotai/Kimi-Linear-48B-A3B-Base | [graph](results/compiled/cuda-serving/decode/moonshotai--Kimi-Linear-48B-A3B-Base.txt) | rule-based source trace estimate |
| Qwen/Qwen3.8-2.4T-A95B | [graph](results/compiled/cuda-serving/decode/Qwen--Qwen3.8-2.4T-A95B.txt) | rule-based source trace estimate |
| Qwen/Qwen3.8-Flash-Next | [graph](results/compiled/cuda-serving/decode/Qwen--Qwen3.8-Flash-Next.txt) | rule-based source trace estimate |
| deepseek-ai/DeepSeek-V3-Base | [graph](results/compiled/cuda-serving/decode/deepseek-ai--DeepSeek-V3-Base.txt) | rule-based source trace estimate |
| deepseek-ai/DeepSeek-V3.2 | [graph](results/compiled/cuda-serving/decode/deepseek-ai--DeepSeek-V3.2.txt) | rule-based source trace estimate |
| deepseek-ai/DeepSeek-V4-Pro-Base | [graph](results/compiled/cuda-serving/decode/deepseek-ai--DeepSeek-V4-Pro-Base.txt) | rule-based source trace estimate |
| deepseek-ai/DeepSeek-V4-Flash-Base | [graph](results/compiled/cuda-serving/decode/deepseek-ai--DeepSeek-V4-Flash-Base.txt) | rule-based source trace estimate |
| deepseek-ai/DeepSeek-V4.1-Flash | unsupported | model-graph: ValueError: The checkpoint you are trying to load has model type `deepseek_v41` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date. |
| MiniMaxAI/MiniMax-M3 | [graph](results/compiled/cuda-serving/decode/MiniMaxAI--MiniMax-M3.txt) | rule-based source trace estimate |
| MiniMaxAI/MiniMax-M2.7 | [graph](results/compiled/cuda-serving/decode/MiniMaxAI--MiniMax-M2.7.txt) | rule-based source trace estimate |
| openai/gpt-oss-120b | [graph](results/compiled/cuda-serving/decode/openai--gpt-oss-120b.txt) | config-only serving adapter / torch.compile FX |
| Qwen/Qwen3.8-27B | [graph](results/compiled/cuda-serving/decode/Qwen--Qwen3.8-27B.txt) | rule-based source trace estimate |
| Qwen/Qwen3.8-27B-FP8 | [graph](results/compiled/cuda-serving/decode/Qwen--Qwen3.8-27B-FP8.txt) | rule-based source trace estimate |
| Qwen/Qwen3.5-397B-A17B | [graph](results/compiled/cuda-serving/decode/Qwen--Qwen3.5-397B-A17B.txt) | rule-based source trace estimate |
| Qwen/Qwen3-Next-80B-A3B-Instruct | [graph](results/compiled/cuda-serving/decode/Qwen--Qwen3-Next-80B-A3B-Instruct.txt) | rule-based source trace estimate |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | [graph](results/compiled/cuda-serving/decode/Qwen--Qwen3-235B-A22B-Instruct-2507.txt) | rule-based source trace estimate |
<!-- serving-results:end -->

<!-- earlier-examples:start -->
## Earlier Qwen examples

These prefill-only graphs were generated before phase selection was added.
Their headers now identify prefill explicitly; their original operation records
are preserved. Current, separately generated prefill and decode graphs are in
[the collection index](#model-execution-graphs).

- [Qwen 3.8 27B](results/examples/prefill/Qwen--Qwen3.8-27B.txt)
- [Qwen 3.8 27B FP8](results/examples/prefill/Qwen--Qwen3.8-27B-fp8.txt)
<!-- earlier-examples:end -->
