import json
from pathlib import Path
import re
import unittest

from generate_serving_results import ROOT, DEST, estimate


class ServingEstimateTests(unittest.TestCase):
    def test_qkv_packing_and_dependencies(self):
        source = (ROOT/'results/decode/Qwen--Qwen3-235B-A22B-Instruct-2507.txt').read_text()
        graph, counts = estimate(source)
        self.assertEqual(counts['PackedQKVProjection'], 94)
        self.assertIn('packed_output_shape=(32, 1, 9216)', graph)
        self.assertIn('covers=[QueryProjection, KeyProjection, ValueProjection]', graph)
        self.assertNotRegex(graph, r'\[step \d+\] Linear \[[^]]+\.[qkv]_proj\]')
        # Every source output remains defined, including logical views of grouped operations.
        def outputs(text):
            return {ref for line in text.splitlines() if '[step ' in line
                    for ref in re.findall(r'%\w+', line.split(' -> ', 1)[1])}
        self.assertEqual(outputs(source), outputs(graph))

    def test_mla_is_not_replaced_with_standard_qkv(self):
        graph, counts = estimate((ROOT/'results/decode/deepseek-ai--DeepSeek-V3-Base.txt').read_text())
        self.assertNotIn('PackedQKVProjection', counts)
        self.assertIn('.q_a_proj]', graph)
        self.assertIn('.kv_b_proj]', graph)
        self.assertIn('Tensor(shape=(32, 1, 8192, 512))', graph)
        self.assertIn('NativeAttentionBoundary', graph)

    def test_recurrent_state_and_fused_norm(self):
        graph, counts = estimate((ROOT/'results/decode/Qwen--Qwen3-Next-80B-A3B-Instruct.txt').read_text())
        self.assertGreater(counts['NativeRecurrentBoundary'], 0)
        self.assertGreater(counts['FusedAddRMSNorm'], 0)
        self.assertIn('covers=[Recurrence, StateUpdate, ConvolutionStateUpdate]', graph)
        self.assertIn('Tensor(shape=(32, 32, 128, 128), dtype=torch.float32)', graph)

    def test_generated_catalog_and_hashes(self):
        import hashlib
        manifest = json.loads((DEST/'manifest.json').read_text())
        self.assertEqual(sum(r['status'] == 'success' for r in manifest['runs']), 22)
        for record in manifest['runs']:
            if record['status'] != 'success':
                self.assertEqual(record['model_id'], 'deepseek-ai/DeepSeek-V4.1-Flash')
                continue
            graph = (ROOT/record['graph']).read_text()
            self.assertEqual(hashlib.sha256(graph.encode()).hexdigest(), record['graph_sha256'])
            self.assertTrue(graph.splitlines()[-1].startswith('return '))
            self.assertIn('cache_input', graph)
            if record['method'] == 'rule-based source trace estimate':
                source = (ROOT/record['source_graph']).read_text()
                def outputs(text):
                    return {ref for line in text.splitlines() if '[step ' in line
                            for ref in re.findall(r'%\w+', line.split(' -> ', 1)[1])}
                self.assertEqual(outputs(source), outputs(graph), record['model_id'])
            for line in graph.splitlines():
                if '[step ' in line:
                    self.assertIn('stages=[', line)


if __name__ == '__main__':
    unittest.main()
