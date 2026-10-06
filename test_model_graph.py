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

from model_graph import progress
from model_ops import ModuleGraph


class ModelGraphTests(unittest.TestCase):
    def run_graph(self, config, task="causal-lm"):
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            result = subprocess.run(
                [str(Path(sys.executable).parent / "model-graph"), directory,
                 "--batch", "32", "--seq-len", "8192", "--task", task,
                 "--local-files-only"],
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


if __name__ == "__main__":
    unittest.main()
