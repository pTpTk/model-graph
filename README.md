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

Both modes use eager attention, assume all tokens are valid, and disable caching.
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
