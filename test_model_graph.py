"""Exercise real exports using local configs, without downloading model weights."""
import io
import os
import threading
import re

import torch
import transformers
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from transformers import BertConfig, GPT2Config, LlamaConfig, T5Config

from model_graph import progress, resolve_dtype, has_quantization_metadata, describe
from model_ops import ModuleGraph


class ModelGraphTests(unittest.TestCase):
    def run_graph(self, config, task="causal-lm", extra_args=()):
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            result = subprocess.run(
                [str(Path(sys.executable).parent / "model-graph"), directory,
                 "--batch", "32", "--seq-len", "8192", "--task", task,
                 "--local-files-only", *extra_args],
                text=True, capture_output=True, timeout=120,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("batch=32, seq_len=8192", result.stdout)
        self.assertRegex(result.stdout, r"user_input -> Tensor\(shape=\(32, 8192\), dtype=torch.int64\)")
        self.assertRegex(result.stdout, r"parameter .* -> Tensor\(shape=\(")
        self.assertIn("[step ", result.stdout)
        self.assertIn("Embedding", result.stdout)
        self.assertNotIn("aten.", result.stdout)
        self.assertIn("return ", result.stdout)
        self.assertIn("Recording model operations with shape-only tensors: done", result.stderr)
        self.assertNotIn("still running", result.stdout)
        return result.stdout

    def test_llama(self):
        output = self.run_graph(LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=8192,
        ))
        self.assertIn("shape=(64, 16)", output)
        self.assertIn("Attention [model.layers.0.self_attn]", output)
        self.assertIn("MLP [model.layers.0.mlp]", output)
        self.assertIn("ResidualAdd [model.layers.0]", output)
        self.assertIn("OutputProjection [lm_head]", output)
        self.assertIn("shape=(32, 8192, 64)", output)

    def test_config_dtype_and_explicit_override(self):
        config = LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=8192, torch_dtype="bfloat16",
        )
        for flags, expected, source in [
            ((), "bfloat16", "config"),
            (("--dtype", "float16"), "float16", "override"),
        ]:
            with self.subTest(flags=flags):
                output = self.run_graph(config, extra_args=flags)
                self.assertIn(f"dtype={expected} (source={source})", output)
                self.assertIn(
                    "parameter model.embed_tokens.weight -> "
                    f"Tensor(shape=(64, 16), dtype=torch.{expected})", output,
                )
                self.assertIn(f"Tensor(shape=(32, 8192, 64), dtype=torch.{expected})", output)

    def test_unspecified_floating_dtype_is_omitted(self):
        output = self.run_graph(LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=8192,
        ))
        self.assertIn("floating dtype annotations omitted", output)
        self.assertIn("parameter model.embed_tokens.weight -> Tensor(shape=(64, 16))", output)
        self.assertNotIn("dtype=torch.float32", output)
        self.assertIn("dtype=torch.int64", output)

    def test_quantized_graph_omits_floating_dtypes(self):
        config = LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=8192, torch_dtype="bfloat16",
            quantization_config={"quant_method": "fp8", "fmt": "e4m3",
                                 "activation_scheme": "dynamic", "weight_block_size": [128, 128],
                                 "modules_to_not_convert": ["lm_head"]},
        )
        for flags in ((), ("--dtype", "float32")):
            with self.subTest(flags=flags):
                output = self.run_graph(config, extra_args=flags)
                self.assertIn("Quantization metadata present", output)
                self.assertIn("floating dtype annotations omitted", output)
                self.assertIn("parameter model.embed_tokens.weight -> Tensor(shape=(64, 16))", output)
                self.assertNotRegex(output, r"dtype=(?:torch\.)?(?:bfloat|float)\w*")
                self.assertIn("dtype=torch.int64", output)

    def test_quantized_aten_graph_omits_floating_dtypes(self):
        with tempfile.TemporaryDirectory() as directory:
            LlamaConfig(
                vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                torch_dtype="bfloat16", quantization_config={"quant_method": "fp8"},
            ).save_pretrained(directory)
            result = subprocess.run(
                [str(Path(sys.executable).parent / "model-graph"), directory,
                 "--seq-len", "16", "--local-files-only", "--format", "aten"],
                text=True, capture_output=True, timeout=120,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("aten.", result.stdout)
        self.assertIn("Quantization metadata present", result.stdout)
        self.assertNotRegex(result.stdout, r"dtype=(?:torch\.)?(?:bfloat|float)\w*")

    def test_gpt2(self):
        self.run_graph(GPT2Config(vocab_size=64, n_embd=16, n_layer=1,
                                 n_head=2, n_positions=8192,
                                 bos_token_id=0, eos_token_id=0))

    def test_encoder_and_masked_lm(self):
        config = BertConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2,
                            max_position_embeddings=8192)
        for task in ("encoder", "masked-lm"):
            with self.subTest(task=task):
                self.run_graph(config, task)

    def test_seq2seq(self):
        self.run_graph(T5Config(vocab_size=64, d_model=16, d_ff=32, d_kv=8,
                                num_layers=1, num_decoder_layers=1, num_heads=2,
                                decoder_start_token_id=0), "seq2seq-lm")

    @unittest.skipUnless(hasattr(transformers, "Qwen3_5TextConfig"), "Requires Qwen3.5 support")
    def test_hybrid_attention_labels(self):
        config = transformers.Qwen3_5TextConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000,
                             "partial_rotary_factor": 1.0,
                             "mrope_section": [1, 1, 2], "mrope_interleaved": True},
        )
        output = self.run_graph(config)
        self.assertIn("LinearAttention [model.layers.0.linear_attn]", output)
        self.assertIn("LinearAttentionCore [model.layers.0.linear_attn]", output)
        self.assertIn("Attention [model.layers.1.self_attn]", output)
        self.assertIn("AttentionCore [model.layers.1.self_attn]", output)

    def test_optional_aten_format(self):
        with tempfile.TemporaryDirectory() as directory:
            LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2).save_pretrained(directory)
            result = subprocess.run(
                [str(Path(sys.executable).parent / "model-graph"), directory,
                 "--seq-len", "16", "--local-files-only", "--format", "aten"],
                text=True, capture_output=True, timeout=120,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("aten.", result.stdout)
        self.assertIn("Exporting execution graph: done", result.stderr)

    def test_invalid_dimensions(self):
        for flag in ("--batch", "--seq-len"):
            result = subprocess.run(
                [str(Path(sys.executable).parent / "model-graph"),
                 "username/modelname", flag, "0"],
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be greater than zero", result.stderr)


class PhaseGraphTests(unittest.TestCase):
    def phase_graph(self, config, phase):
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            result = subprocess.run(
                [str(Path(sys.executable).parent / "model-graph"), directory,
                 "--batch", "32", "--seq-len", "8192", "--phase", phase,
                 "--local-files-only"], text=True, capture_output=True, timeout=120,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"phase={phase}", result.stdout)
        return result.stdout

    def test_standard_prefill_and_decode_are_distinct(self):
        config=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
                           num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                           max_position_embeddings=8193)
        prefill=self.phase_graph(config,"prefill")
        decode=self.phase_graph(config,"decode")
        self.assertIn("user_input -> Tensor(shape=(32, 8192), dtype=torch.int64)",prefill)
        self.assertNotIn("cache_input",prefill)
        self.assertIn("user_input -> Tensor(shape=(32, 1), dtype=torch.int64)",decode)
        self.assertIn("cache_input past_key_values.layers.0.keys -> Tensor(shape=(32, 1, 8192, 8))",decode)
        self.assertIn("Tensor(shape=(32, 1, 8193, 8))",decode)
        self.assertIn("Tensor(shape=(32, 1, 64))",decode.splitlines()[-1])
        self.assertEqual(decode.count("Embedding [model.embed_tokens]"),1)

    @unittest.skipUnless(hasattr(transformers,"Qwen3_5TextConfig"),"Requires Qwen3.5")
    def test_hybrid_decode_exposes_fixed_recurrent_state_and_growing_kv(self):
        config=transformers.Qwen3_5TextConfig(
            vocab_size=64,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
            layer_types=["linear_attention","full_attention"],
            rope_parameters={"rope_type":"default","rope_theta":10000,"partial_rotary_factor":1.0,
                             "mrope_section":[1,1,2],"mrope_interleaved":True},
        )
        output=self.phase_graph(config,"decode")
        self.assertIn("cache_input past_key_values.layers.0.conv_states.0 -> Tensor(shape=(32, 48, 4))",output)
        self.assertIn("cache_input past_key_values.layers.0.recurrent_states.0 -> Tensor(shape=(32, 2, 8, 8))",output)
        self.assertIn("Tensor(shape=(32, 1, 8193, 8))",output)
        updated=next(line for line in output.splitlines() if line.startswith("%updated_past_key_values ="))
        self.assertIn("Tensor(shape=(32, 2, 8, 8))",updated)
        self.assertNotIn("%past_key_values_layers_0_recurrent_states_0",updated)

    @unittest.skipUnless(hasattr(transformers,"DeepseekV3Config"),"Requires DeepSeek V3")
    def test_expert_dispatch_contract_and_forward_restoration(self):
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts
        config=transformers.DeepseekV3Config(hidden_size=16,moe_intermediate_size=8,n_routed_experts=4)
        with torch.device("meta"):
            model=DeepseekV3Experts(config)
            original=model.forward.__func__
            graph=ModuleGraph(model).trace({
                "hidden_states":torch.empty(32,16),
                "top_k_index":torch.empty(32,2,dtype=torch.long),
                "top_k_weights":torch.empty(32,2),
            })
        self.assertIs(model.forward.__func__,original)
        stream=io.StringIO()
        graph.print(stream)
        output=stream.getvalue()
        self.assertIn("ExpertDispatch [<root>]",output)
        self.assertIn("tokens_per_expert=data-dependent",output)
        self.assertIn("Tensor(shape=(32, 16)",output.splitlines()[-1])


class CompiledGraphTests(unittest.TestCase):
    def run_compiled(self, config, phase="prefill", extra_args=()):
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            result = subprocess.run(
                [sys.executable, "-m", "model_graph", directory,
                 "--batch", "2", "--seq-len", "8", "--phase", phase,
                 "--compile", "--local-files-only", *extra_args],
                text=True, capture_output=True, timeout=120,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("torch.compile FX graph; fullgraph=True, dynamic=False", result.stdout)
        self.assertIn("Capturing torch.compile FX graph with shape-only tensors: done", result.stderr)
        self.assertIn("[step ", result.stdout)
        for line in result.stdout.splitlines():
            if "[step " in line:
                self.assertIn("stages=[", line)
            if " op=" in line:
                self.assertRegex(line, r"\[step \d+\] \w+ \[.*\] .* -> ")
        self.assertTrue(result.stdout.splitlines()[-1].startswith("return "))
        return result.stdout

    def test_compiled_llama_prefill_has_parameters_and_shapes(self):
        output = self.run_compiled(LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            dtype="bfloat16",
        ))
        self.assertIn("user_input input_ids -> Tensor(shape=(2, 8), dtype=torch.int64)", output)
        self.assertIn("parameter model.embed_tokens.weight -> Tensor(shape=(64, 16), dtype=torch.bfloat16)", output)
        self.assertIn("Tensor(shape=(2, 8, 64), dtype=torch.bfloat16)", output)
        self.assertIn("torch.nn.functional.embedding", output)
        self.assertNotIn("cache_input", output)

    @unittest.skipUnless(hasattr(transformers, "GptOssConfig"), "Requires gpt-oss support")
    def test_gpt_oss_compiled_decode_cache_and_expert_dependencies(self):
        config = transformers.GptOssConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, num_local_experts=4, num_experts_per_tok=2,
            sliding_window=4, layer_types=["sliding_attention", "full_attention"],
            quantization_config={"quant_method": "mxfp4"},
        )
        output = self.run_compiled(config, "decode")
        self.assertIn("phase=decode", output)
        self.assertIn("user_input input_ids -> Tensor(shape=(2, 1), dtype=torch.int64)", output)
        self.assertIn("cache_input past_key_values.layers.0.keys -> Tensor(shape=(2, 1, 3, 8))", output)
        self.assertIn("cache_input past_key_values.layers.1.keys -> Tensor(shape=(2, 1, 8, 8))", output)
        updated = next(line for line in output.splitlines() if line.startswith("%updated_past_key_values ="))
        self.assertRegex(updated, r"'past_key_values.layers.1.keys': %s\d+_\d+ Tensor\(shape=\(2, 1, 9, 8\)\)")
        self.assertIn("Tensor(shape=(2, 1, 64))", output)
        self.assertIn("tokens_per_expert=data-dependent", output)
        experts = [line for line in output.splitlines() if " op=model_graph.expert_dispatch.default(" in line]
        self.assertEqual(len(experts), 2)
        self.assertIn("stages=[Inference > Decoder > Layer0 > MoE > ExpertDispatch]", experts[0])
        self.assertIn("%p_model_layers_0_mlp_experts_gate_up_proj Tensor", experts[0])
        self.assertIn("%p_model_layers_0_mlp_experts_down_proj Tensor", experts[0])
        routing = [line for line in output.splitlines() if " op=" in line and "Routing]" in line]
        indices = next(line for line in routing if "op=_operator.getitem(" in line and ", 1) -> " in line).split(" -> ")[1].split()[0]
        scores = next(line for line in routing if "op=torch.nn.functional.softmax" in line).split(" -> ")[1].split()[0]
        self.assertIn(indices, experts[0])
        self.assertIn(scores, experts[0])
        self.assertEqual(output.count(" op=torch.nn.functional.embedding("), 1)
        for stage in ("AttentionMask", "PositionalEncoding", "AttentionNormalization", "QueryProjection",
                      "KeyProjection", "ValueProjection", "RotaryPositionEncoding", "KVCacheUpdate",
                      "AttentionCore", "AttentionOutputPreparation", "AttentionOutputProjection", "AttentionResidual", "MLPNormalization",
                      "Routing", "ExpertDispatch", "MLPResidual", "FinalNormalization", "OutputProjection"):
            self.assertIn(stage + "]", output, stage)
        self.assertIn("TransformerBlock [model.layers.0] type=GptOssDecoderLayer", output)
        shaping = [line for line in output.splitlines() if " op=" in line and "Reshape [model.layers.0.self_attn]" in line]
        self.assertTrue(any("Attention > QueryProjection]" in line for line in shaping))
        self.assertRegex(experts[0], r"^\s{10}\[step \d+\]")
        self.assertNotRegex(output, r"dtype=(?:torch\.)?(?:bfloat|float)\w*")

    def test_compile_and_aten_flags_are_rejected_before_loading(self):
        result = subprocess.run(
            [sys.executable, "-m", "model_graph", "unused/model", "--compile", "--format", "aten"],
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)
        self.assertNotIn("Loading configuration", result.stderr)

    @unittest.skipUnless(hasattr(transformers, "GptOssConfig"), "Requires gpt-oss support")
    def test_cuda_serving_decode_has_packed_qkv_and_native_stage_contracts(self):
        config = transformers.GptOssConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, num_local_experts=4, num_experts_per_tok=2,
            sliding_window=4, layer_types=["sliding_attention", "full_attention"],
            quantization_config={"quant_method": "mxfp4"},
        )
        output = self.run_compiled(config, "decode", ("--compile-backend", "cuda-serving"))
        self.assertIn("independent of host device", output)
        self.assertIn("not a measured kernel graph", output)
        self.assertIn("logical active tokens", output)
        projections = [line for line in output.splitlines() if "qkv_proj]" in line and " op=" in line]
        self.assertEqual(len(projections), 2)
        self.assertIn("covers=[QueryProjection, KeyProjection, ValueProjection]", projections[0])
        self.assertIn("Tensor(shape=(32, 16))", projections[0])
        self.assertIn("Tensor(shape=(2, 1, 32))", projections[0])
        self.assertNotIn("self_attn.q_proj]", output)
        self.assertNotIn("self_attn.k_proj]", output)
        self.assertNotIn("self_attn.v_proj]", output)
        self.assertEqual(output.count(" op=model_graph.native_attention.default("), 2)
        self.assertEqual(output.count(" op=model_graph.expert_dispatch.default("), 2)
        self.assertIn("covers=[RotaryPositionEncoding, KVCacheUpdate]", output)
        self.assertIn("covers=[ResidualAdd, Normalization]", output)
        self.assertIn("Tensor(shape=(2, 1, 9, 8))", output)
        self.assertIn("cache_input past_key_values.layers.0.keys -> Tensor(shape=(2, 1, 8, 8))", output)
        self.assertIn("Tensor(shape=(2, 1, 64))", output.splitlines()[-1])
        self.assertNotRegex(output, r"dtype=(?:torch\.)?(?:bfloat|float)\w*")

    @unittest.skipUnless(hasattr(transformers, "GptOssConfig"), "Requires gpt-oss support")
    def test_cuda_serving_prefill_does_not_expose_cache_state(self):
        config = transformers.GptOssConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            num_local_experts=4, num_experts_per_tok=2, layer_types=["full_attention"])
        output = self.run_compiled(config, extra_args=("--compile-backend", "cuda-serving"))
        self.assertNotIn("cache_input", output)
        self.assertNotIn("%updated_past_key_values", output)
        self.assertIn("Tensor(shape=(2, 8, 64))", output.splitlines()[-1])

    def test_cuda_serving_requires_compile_and_rejects_unsupported_architecture(self):
        result = subprocess.run([sys.executable, "-m", "model_graph", "unused/model",
            "--compile-backend", "cuda-serving"], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --compile", result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2).save_pretrained(directory)
            result = subprocess.run([sys.executable, "-m", "model_graph", directory,
                "--compile", "--compile-backend", "cuda-serving", "--local-files-only"],
                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertIn("currently supports gpt-oss", result.stderr)

    @unittest.skipUnless(hasattr(transformers, "GptOssConfig"), "Requires gpt-oss support")
    def test_cuda_serving_adapter_rejects_numeric_execution(self):
        from cuda_serving_graph import CudaServingForCausalLM
        config = transformers.GptOssConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            num_local_experts=4, num_experts_per_tok=2, layer_types=["full_attention"])
        with torch.device("meta"):
            model = CudaServingForCausalLM(config)
        with self.assertRaisesRegex(ValueError, "shape tensors only"):
            model(input_ids=torch.ones(2, 1, dtype=torch.long), position_ids=torch.zeros(2, 1, dtype=torch.long))

    def test_compiled_format_preserves_reused_modules_and_dependency_edges(self):
        from compiled_graph import CompiledGraph

        class Reused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                return self.projection(self.norm(x + self.projection(x)))

        with torch.device("meta"):
            graph = CompiledGraph(Reused().eval()).trace({"x": torch.empty(2, 3, 4)})
        stream = io.StringIO()
        graph.print(stream)
        output = stream.getvalue()
        self.assertEqual(output.count("Linear [projection] type=Linear"), 2)
        operations = [line for line in output.splitlines() if " op=" in line]
        self.assertEqual(len(operations), sum(node.op not in ("placeholder", "output")
                                             for node in graph.graph_module.graph.nodes))
        declared = set()
        for line in output.splitlines():
            if line.startswith("%") and " = " in line:
                declared.add(line.split()[0])
            if " op=" in line:
                inputs, outputs = line.split(" -> ", 1)
                self.assertTrue(set(re.findall(r"%\w+", inputs)) <= declared, line)
                declared.update(re.findall(r"%\w+", outputs))
            if line.startswith("return "):
                self.assertTrue(set(re.findall(r"%\w+", line)) <= declared, line)
        additions = [line for line in operations if " op=add(" in line or " op=_operator.add(" in line]
        self.assertEqual(len(additions), 1)
        self.assertIn("%x Tensor(shape=(2, 3, 4)", additions[0])
        self.assertIn("stages=[Inference > Functional]", additions[0])

    def test_graph_break_fails_and_restores_expert_forward(self):
        from compiled_graph import CompiledGraph

        class GptOssExperts(torch.nn.Module):
            def forward(self, hidden_states, router_indices=None, routing_weights=None):
                raise AssertionError("The shape contract should replace this forward")

        class DataDependent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = GptOssExperts()

            def forward(self, x):
                return self.experts(x)[:int(x.sum().item())]

        with torch.device("meta"):
            model = DataDependent()
            original = model.experts.forward.__func__
            with self.assertRaises(Exception):
                CompiledGraph(model).trace({"x": torch.empty(4)})
        self.assertIs(model.experts.forward.__func__, original)
        self.assertNotIn("forward", vars(model.experts))


class DtypeResolutionTests(unittest.TestCase):
    def test_nested_and_legacy_config(self):
        class Config:
            def __init__(self, metadata, text=None):
                self.metadata, self.text = metadata, text

            def to_dict(self):
                return self.metadata

            def get_text_config(self):
                return self.text or self

        text = Config({"torch_dtype": "bfloat16"})
        self.assertEqual(resolve_dtype(Config({}, text)), (torch.bfloat16, "text_config"))
        self.assertEqual(resolve_dtype(Config({"dtype": "float16"})), (torch.float16, "config"))
        self.assertEqual(resolve_dtype(Config({})), (None, None))
        self.assertEqual(resolve_dtype(Config({}, text), "float32"), (torch.float32, "override"))

    def test_quantization_metadata_on_parent_and_nested_configs(self):
        class Config:
            def __init__(self, metadata):
                self.metadata = metadata

            def to_dict(self):
                return self.metadata

        for metadata in (
            {"dtype": "bfloat16", "quantization_config": {"quant_method": "fp8"}},
            {"text_config": {"quantization_config": {"quant_method": "fp8"}}},
            {"decoder": {"compression_config": {"format": "compressed-tensors"}}},
            {"load_in_8bit": True},
        ):
            with self.subTest(metadata=metadata):
                self.assertTrue(has_quantization_metadata(Config(metadata)))
        for metadata in ({}, {"dtype": "bfloat16"}, {"quantization_config": None},
                         {"quantization_config": {}}, {"load_in_4bit": False}):
            with self.subTest(metadata=metadata):
                self.assertFalse(has_quantization_metadata(Config(metadata)))

    def test_floating_dtype_operands_are_not_exposed_when_omitted(self):
        self.assertEqual(describe(torch.bfloat16, False), "unspecified_floating_dtype")
        self.assertEqual(describe(torch.int64, False), "torch.int64")

    def test_explicit_float32_cast_is_retained(self):
        class Upcast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 4, dtype=torch.bfloat16)

            def forward(self, x):
                return self.projection(x).float()

        with torch.device("meta"):
            graph = ModuleGraph(Upcast()).trace({"x": torch.empty(2, 4, dtype=torch.bfloat16)})
        stream = io.StringIO()
        graph.print(stream)
        output = stream.getvalue()
        self.assertIn("dtype=torch.bfloat16", output)
        self.assertIn("ops=[Cast]", output)
        self.assertIn("dtype=torch.float32", output.splitlines()[-1])


class ModuleDependencyTests(unittest.TestCase):
    def test_residual_edges_and_reused_module_calls(self):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                projected = self.projection(x)
                residual = x + projected
                normalized = self.norm(residual)
                return self.projection(normalized)

        with torch.device("meta"):
            graph = ModuleGraph(Residual().eval()).trace({"x": torch.empty(2, 3, 4)})
        stream = io.StringIO()
        graph.print(stream)
        output = stream.getvalue()
        lines = output.splitlines()
        projections = [line for line in lines if "Linear [projection]" in line]
        self.assertEqual(len(projections), 2)
        self.assertIn("shape=(4, 4)", projections[0])
        first_result = re.search(r" -> (%s\d+_\d+)", projections[0]).group(1)
        add = next(line for line in lines if "ops=[Add]" in line)
        self.assertIn("%x Tensor", add)
        self.assertIn(first_result, add)
        add_result = re.search(r" -> \((%s\d+_\d+)", add).group(1)
        norm = next(line for line in lines if "Normalization [norm]" in line)
        self.assertIn(add_result, norm)
        norm_result = re.search(r" -> (%s\d+_\d+)", norm).group(1)
        self.assertIn(norm_result, projections[1])
        self.assertEqual(len(graph.model.projection._forward_hooks), 0)
        self.assertEqual(len(graph.model.projection._forward_pre_hooks), 0)

    def test_data_dependent_operation_fails_without_cpu_fallback(self):
        class DataDependent(torch.nn.Module):
            def forward(self, x):
                return x[:int(x.sum().item())]
        with torch.device("meta"):
            model = DataDependent()
            with self.assertRaises(Exception):
                ModuleGraph(model).trace({"x": torch.empty(4)})
        self.assertEqual(len(model._forward_hooks), 0)
        self.assertEqual(len(model._forward_pre_hooks), 0)


class ProgressTests(unittest.TestCase):
    def test_heartbeat_on_stderr_and_stops_on_completion(self):
        reported = threading.Event()

        class Capture(io.StringIO):
            def write(self, value):
                result = super().write(value)
                if "still running" in value:
                    reported.set()
                return result

        stderr, stdout = Capture(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            with progress("Exporting execution graph", interval=0.01):
                self.assertTrue(reported.wait(timeout=2), "Missing export heartbeat")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("still running", stderr.getvalue())
        self.assertTrue(stderr.getvalue().splitlines()[-1].startswith(
            "model-graph: Exporting execution graph: done"))

    def test_failed_stage_does_not_report_success(self):
        stderr = io.StringIO()
        with self.assertRaisesRegex(RuntimeError, "export failed"):
            with progress("Exporting execution graph", stream=stderr):
                raise RuntimeError("export failed")
        self.assertIn("Exporting execution graph", stderr.getvalue())
        self.assertNotIn("done", stderr.getvalue())


class KimiK3GraphTests(unittest.TestCase):
    phase_graph = PhaseGraphTests.phase_graph

    def test_k3_architecture_and_decode_cache_boundaries(self):
        from kimi_k3_graph import KimiK3TextConfig
        config = KimiK3TextConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
            hidden_act="situ", activation_situ_beta=4., activation_situ_linear_beta=25.,
            q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,
            v_head_dim=4, mla_use_nope=True, mla_use_output_gate=True,
            num_experts=4, num_experts_per_token=2, moe_intermediate_size=16,
            num_shared_experts=1, first_k_dense_replace=1,
            routed_expert_hidden_size=8, latent_moe_use_norm=True, attn_res_block_size=2,
            linear_attn_config={"kda_layers":[1,2], "full_attn_layers":[3],
                                "num_heads":2, "head_dim":4, "short_conv_kernel_size":4,
                                "use_full_rank_gate":True, "gate_lower_bound":-5.},
            quantization_config={"quant_method":"compressed-tensors"},
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                graph = self.phase_graph(config, phase)
                for label in ("SiTUAndMultiply", "AttentionResidual", "LinearAttentionCore",
                              "CausalConvolution", "routed_expert_down_proj", "routed_expert_norm"):
                    self.assertIn(label, graph)
                self.assertIn("Tensor(shape=(4, 16, 8))", graph)
                self.assertNotRegex(graph, r"dtype=(?:torch\.)?(?:bfloat|float)\w*")
                if phase == "decode":
                    self.assertIn("cache_input past_key_values.conv_states.0.0 -> Tensor(shape=(32, 8, 4))", graph)
                    self.assertIn("cache_input past_key_values.recurrent_states.0 -> Tensor(shape=(32, 2, 4, 4))", graph)
                    self.assertIn("cache_input past_key_values.key_cache.2 -> Tensor(shape=(32, 2, 8192, 8))", graph)
                    self.assertIn("Tensor(shape=(32, 2, 8193, 8))", graph)
                    self.assertIn("Tensor(shape=(32, 1, 64))", graph.splitlines()[-1])

    def test_situ_gate_and_linear_branch_match_official_equation(self):
        from kimi_k3_graph import SituAndMul
        gate = torch.tensor([[-4., 0., 4.]])
        up = torch.tensor([[25., 0., -25.]])
        result = SituAndMul(beta=4., linear_beta=25.)(torch.cat((gate, up), dim=-1))
        expected = (4. * torch.tanh(gate / 4.) * gate.sigmoid()) * (25. * torch.tanh(up / 25.))
        torch.testing.assert_close(result, expected)
        self.assertEqual(tuple(result.shape), (1, 3))

    def test_shape_adapter_rejects_numeric_model_execution(self):
        from kimi_k3_graph import KimiK3DeltaRule
        with self.assertRaisesRegex(ValueError, "shape tensors only"):
            KimiK3DeltaRule(-5.)(*[torch.ones(1, 1, 1, 4)] * 7)


class SparseIndexerGraphTests(unittest.TestCase):
    def test_boundary_contracts_match_native_output_dimensions(self):
        from types import SimpleNamespace
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer
        cases = [
            (Glm5NextTextIndexer, SimpleNamespace(hidden_size=16, index_n_heads=2, index_head_dim=4,
                qk_rope_head_dim=2, index_topk=4, q_lora_rank=8, index_kpool=2,
                index_kpool_always_select_tail=True)),
            (Qwen4ExpTextQSAIndexer, SimpleNamespace(hidden_size=16, indexer_n_heads=2,
                indexer_kv_heads=1, indexer_head_dim=4, indexer_budget=4,
                indexer_compress_ratio=2, rms_norm_eps=1e-6)),
        ]
        for cls, config in cases:
            for query in (1, 8):
                with self.subTest(cls=cls.__name__, query=query):
                    native = cls(config, 0).eval()
                    inputs = {"hidden_states":torch.ones(2, query, 16), "past_key_values":None}
                    if cls is Glm5NextTextIndexer:
                        inputs.update(q_resid=torch.ones(2, query, 8), attention_mask=torch.ones(2, query, dtype=torch.bool))
                    else:
                        inputs.update(position_embeddings=(torch.ones(2, query, 4), torch.zeros(2, query, 4)),
                                      attention_mask=torch.zeros(2, 1, query, query))
                    with torch.no_grad():
                        expected = native(**inputs)
                    with torch.device("meta"):
                        model = cls(config, 0).eval()
                        arguments = {key: value.to('meta') if isinstance(value, torch.Tensor) else
                                     tuple(item.to('meta') for item in value) if isinstance(value, tuple) else value
                                     for key, value in inputs.items()}
                        original = model.forward.__func__
                        graph = ModuleGraph(model).trace(arguments)
                    self.assertIs(model.forward.__func__, original)
                    self.assertEqual(graph.result.shape, tuple(expected.shape))
                    self.assertEqual(graph.result.dtype, expected.dtype)
                    output = io.StringIO(); graph.print(output)
                    self.assertIn("selected_positions=data-dependent", output.getvalue())


if __name__ == "__main__":
    unittest.main()
