# Model execution graphs

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
Its KDA recurrence is an explicit boundary shape contract. [Source revision and hashes](kimi-k3-source.json) document the adaptation.

This is a representative catalog of public general-purpose model architecture families, not a benchmark ranking.
A failed entry has no substitute or fabricated execution graph.

| Model | Architecture | Prefill | Decode |
| --- | --- | --- | --- |
| [zai-org/GLM-4.5-Base](https://huggingface.co/zai-org/GLM-4.5-Base) | `glm4_moe` | [graph](prefill/zai-org--GLM-4.5-Base.txt) (2566 steps) | [graph](decode/zai-org--GLM-4.5-Base.txt) (2566 steps) |
| [zai-org/GLM-4.5-Air-Base](https://huggingface.co/zai-org/GLM-4.5-Air-Base) | `glm4_moe` | [graph](prefill/zai-org--GLM-4.5-Air-Base.txt) (1152 steps) | [graph](decode/zai-org--GLM-4.5-Air-Base.txt) (1152 steps) |
| [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) | `glm_moe_dsa` | [graph](prefill/zai-org--GLM-5.txt) (2954 steps) | [graph](decode/zai-org--GLM-5.txt) (2954 steps) |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | `glm_moe_dsa` | [graph](prefill/zai-org--GLM-5.3.txt) (2441 steps) | [graph](decode/zai-org--GLM-5.3.txt) (2441 steps) |
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `glm5_next_text` | [graph](prefill/zai-org--GLM-5.3-Flash.txt) (1925 steps) | [graph](decode/zai-org--GLM-5.3-Flash.txt) (1925 steps) |
| [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | `kimi_linear` | [graph](prefill/moonshotai--Kimi-K3.txt) (4063 steps) | [graph](decode/moonshotai--Kimi-K3.txt) (4063 steps) |
| [moonshotai/Kimi-K2-Base](https://huggingface.co/moonshotai/Kimi-K2-Base) | `kimi_k2` | [graph](prefill/moonshotai--Kimi-K2-Base.txt) (1771 steps) | [graph](decode/moonshotai--Kimi-K2-Base.txt) (1771 steps) |
| [moonshotai/Kimi-Linear-48B-A3B-Base](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) | `kimi_linear` | [graph](prefill/moonshotai--Kimi-Linear-48B-A3B-Base.txt) (870 steps) | [graph](decode/moonshotai--Kimi-Linear-48B-A3B-Base.txt) (870 steps) |
| [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | `qwen3_5_moe_text` | [graph](prefill/Qwen--Qwen3.8-2.4T-A95B.txt) (2607 steps) | [graph](decode/Qwen--Qwen3.8-2.4T-A95B.txt) (2607 steps) |
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `qwen4_exp_text` | [graph](prefill/Qwen--Qwen3.8-Flash-Next.txt) (2082 steps) | [graph](decode/Qwen--Qwen3.8-Flash-Next.txt) (2082 steps) |
| [deepseek-ai/DeepSeek-V3-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) | `deepseek_v3` | [graph](prefill/deepseek-ai--DeepSeek-V3-Base.txt) (1759 steps) | [graph](decode/deepseek-ai--DeepSeek-V3-Base.txt) (1759 steps) |
| [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | `deepseek_v32` | [graph](prefill/deepseek-ai--DeepSeek-V3.2.txt) (2308 steps) | [graph](decode/deepseek-ai--DeepSeek-V3.2.txt) (2308 steps) |
| [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) | `deepseek_v4` | [graph](prefill/deepseek-ai--DeepSeek-V4-Pro-Base.txt) (3695 steps) | [graph](decode/deepseek-ai--DeepSeek-V4-Pro-Base.txt) (3331 steps) |
| [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) | `deepseek_v4` | [graph](prefill/deepseek-ai--DeepSeek-V4-Flash-Base.txt) (2588 steps) | [graph](decode/deepseek-ai--DeepSeek-V4-Flash-Base.txt) (2340 steps) |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `deepseek_v41_text` | unsupported ([log](logs/deepseek-ai--DeepSeek-V4.1-Flash.prefill.log)) | unsupported ([log](logs/deepseek-ai--DeepSeek-V4.1-Flash.decode.log)) |
| [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | `minimax_m3_vl` | [graph](prefill/MiniMaxAI--MiniMax-M3.txt) (2123 steps) | [graph](decode/MiniMaxAI--MiniMax-M3.txt) (2123 steps) |
| [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) | `minimax_m2` | [graph](prefill/MiniMaxAI--MiniMax-M2.7.txt) (1124 steps) | [graph](decode/MiniMaxAI--MiniMax-M2.7.txt) (1124 steps) |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | `gpt_oss` | [graph](prefill/openai--gpt-oss-120b.txt) (656 steps) | [graph](decode/openai--gpt-oss-120b.txt) (656 steps) |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `qwen3_5_text` | [graph](prefill/Qwen--Qwen3.8-27B.txt) (1432 steps) | [graph](decode/Qwen--Qwen3.8-27B.txt) (1432 steps) |
| [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `qwen3_5_text` | [graph](prefill/Qwen--Qwen3.8-27B-FP8.txt) (1432 steps) | [graph](decode/Qwen--Qwen3.8-27B-FP8.txt) (1432 steps) |
| [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | `qwen3_5_moe_text` | [graph](prefill/Qwen--Qwen3.5-397B-A17B.txt) (1703 steps) | [graph](decode/Qwen--Qwen3.5-397B-A17B.txt) (1703 steps) |
| [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | `qwen3_next` | [graph](prefill/Qwen--Qwen3-Next-80B-A3B-Instruct.txt) (1220 steps) | [graph](decode/Qwen--Qwen3-Next-80B-A3B-Instruct.txt) (1220 steps) |
| [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) | `qwen3_moe` | [graph](prefill/Qwen--Qwen3-235B-A22B-Instruct-2507.txt) (2076 steps) | [graph](decode/Qwen--Qwen3-235B-A22B-Instruct-2507.txt) (2076 steps) |

## Unsupported or incomplete entries

- **deepseek-ai/DeepSeek-V4.1-Flash**: Installed Transformers lacks deepseek_v41 / deepseek_v41_text support. Official inference uses custom CUDA kernels and an additional Engram/vision architecture; no faithful shape adapter is provided.

Regenerate with `python generate_results.py`; use `--model OWNER/NAME` to select an entry.
Successful outputs are retained unless `--force` is supplied. Failed entries are retried.
