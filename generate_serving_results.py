#!/usr/bin/env python3
"""Estimate CUDA serving boundaries from the catalog's recorded decode traces."""
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

from results_documentation import update_index

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'results/compiled/cuda-serving'
STEP = re.compile(r'^( *)(\[step (\d+)\]) ([^ ]+) \[([^]]+)\] (.*)$')
REF = re.compile(r'%\w+(?: Tensor\(shape=\([^)]*\)(?:, dtype=[^)]*)?\))?')


def stages(path, label, body):
    result = ['Inference', 'Decoder']
    layer = re.search(r'layers\.(\d+)', path)
    if layer:
        result.append('Layer' + layer[1])
    if 'embed' in path:
        result.append('Embedding')
    elif path == 'lm_head':
        result.append('OutputProjection')
    elif 'attn' in path or 'Attention' in label:
        result.append('Attention')
        if 'index' in path:
            result.append('SparseIndexer')
        if 'recurrent_states' in body:
            result.extend(['Recurrence', 'StateUpdate'])
        if 'conv_states' in body:
            result.append('ConvolutionStateUpdate')
        if 'past_key_values_layers' in body and 'recurrent_states' not in body:
            result.append('KVCache')
        if 'MatMul' in body or 'AttentionCore' in label:
            result.append('AttentionCore')
        if 'Cat' in body and ('Multiply' in body or 'Neg' in body):
            result.append('RotaryEncoding')
    elif 'mlp' in path or 'expert' in path:
        result.append('FeedForward')
        if 'shared' in path:
            result.append('SharedExperts')
        elif 'expert' in path:
            result.append('Experts')
        if label == 'Router' or path.endswith('.gate'):
            result.append('Routing')
    if 'norm' in path or label == 'Normalization':
        result.append('Normalization')
    if label == 'ResidualAdd':
        result.append('ResidualAdd')
    if 'rotary' in path:
        result.append('RotaryEncoding')
    if label == 'Linear':
        leaf = path.rsplit('.', 1)[-1]
        result.append({'q_proj': 'QueryProjection', 'k_proj': 'KeyProjection',
                       'v_proj': 'ValueProjection', 'o_proj': 'AttentionOutputProjection',
                       'out_proj': 'AttentionOutputProjection'}.get(leaf, leaf))
    result.append(label)
    return list(dict.fromkeys(result))


def estimate(text):
    lines = text.splitlines()
    parsed = {}
    for i, line in enumerate(lines):
        match = STEP.match(line)
        if match:
            parsed[i] = match
    groups = {}
    removed = set()
    # Pack only independent Q/K/V linears with identical input and parent scope.
    scopes = {}
    for i, match in parsed.items():
        if match[4] == 'Linear' and match[5].endswith(('.q_proj', '.k_proj', '.v_proj')):
            scopes.setdefault(match[5].rsplit('.', 1)[0], {})[match[5].rsplit('.', 1)[1]] = i
    for scope, projections in scopes.items():
        if set(projections) != {'q_proj', 'k_proj', 'v_proj'}:
            continue
        indices = sorted(projections.values())
        inputs = [re.search(r'input=(%\w+)', parsed[i][6]) for i in indices]
        if not all(inputs) or len({m[1] for m in inputs}) != 1:
            continue
        groups[indices[0]] = ('PackedQKVProjection', scope + '.qkv_proj', indices,
                             ['QueryProjection', 'KeyProjection', 'ValueProjection'])
        removed.update(indices[1:])
    # Only adjacent residual + RMSNorm with a direct dependency are fused.
    for i, match in parsed.items():
        following = parsed.get(i + 1)
        if match[4] != 'ResidualAdd' or not following or following[4] != 'Normalization' or 'RMSNorm' not in following[6]:
            continue
        outputs = REF.findall(match[6].split(' -> ', 1)[1])
        if not outputs or outputs[0].split()[0] not in following[6].split(' -> ', 1)[0]:
            continue
        groups[i] = ('FusedAddRMSNorm', following[5], [i, i+1], ['ResidualAdd', 'Normalization'])
        removed.add(i+1)
    counts = Counter()
    output = ['# CUDA serving estimate: inferred from recorded model execution; NOT compiler output or CUDA capture.',
              '# No vLLM/SGLang version, kernel selection, quantization kernels or physical paged-cache layout is asserted.',
              '# Numbered steps retain source IDs (gaps denote grouped steps); hierarchy summaries are not launches.',
              '# Packed QKV exposes logical split views; original parameter references describe checkpoint slices.',
              '# Group outputs include logical intermediates to retain dependencies; they need not be materialized.',
              '# Native attention/recurrent/expert boundaries can require multiple launches. Ungrouped operations remain source-level.',
              '# MLA projections/cache layouts are retained from the source trace, not rewritten to a particular serving implementation.']
    for i, line in enumerate(lines):
        if i in removed:
            continue
        match = parsed.get(i)
        if not match:
            output.append(line)
            continue
        label, path, body = match[4], match[5], match[6]
        covers = []
        if i in groups:
            label, path, indices, covers = groups[i]
            produced = {ref.split()[0] for j in indices for ref in REF.findall(parsed[j][6].split(' -> ', 1)[1])}
            inputs = {}
            outputs = []
            for j in indices:
                left, right = parsed[j][6].split(' -> ', 1)
                for ref in REF.findall(left):
                    if ref.split()[0] not in produced:
                        inputs.setdefault(ref.split()[0], ref)
                outputs.extend(REF.findall(right))
            body = f"source_steps=[{', '.join(parsed[j][3] for j in indices)}]({', '.join(inputs.values())}) -> ({', '.join(outputs)})"
            if label == 'PackedQKVProjection':
                shapes = [re.search(r'-> %\w+ Tensor\(shape=\((\d+), (\d+), (\d+)\)', parsed[j][6]) for j in indices]
                if all(shapes):
                    body = f"packed_output_shape=({shapes[0][1]}, {shapes[0][2]}, {sum(int(m[3]) for m in shapes)}) logical_split_views=True " + body
            counts[label] += 1
        elif label == 'AttentionCore':
            label = 'NativeAttentionBoundary'
            covers = ['AttentionCore']
            if 'Cat' in body:
                covers.append('KVCacheUpdate')
            if 'Neg' in body or 'Subtract' in body:
                covers.append('RotaryEncoding')
            counts[label] += 1
        elif label == 'Functional' and 'recurrent_states' in body:
            label = 'NativeRecurrentBoundary'
            covers = ['Recurrence', 'StateUpdate']
            if 'conv_states' in body:
                covers.append('ConvolutionStateUpdate')
            counts[label] += 1
        stage = stages(path, label, body)
        annotation = f" stages=[{' > '.join(stage)}]"
        if covers:
            annotation += f" covers=[{', '.join(covers)}] boundary=estimated"
        output.append(f'{match[1]}{match[2]} {label} [{path}]{annotation} {body}')
    return '\n'.join(output) + '\n', dict(counts)


def main():
    catalog = json.loads((ROOT/'results/catalog.json').read_text())
    source_manifest = json.loads((ROOT/'results/manifest.json').read_text())
    source_runs = {r['model_id']: r for r in source_manifest['runs'] if r['phase'] == 'decode'}
    (DEST/'decode').mkdir(parents=True, exist_ok=True)
    records = []
    index = ['# CUDA serving decode estimates', '',
             f"Batch **{catalog['batch']}**, cached context **{catalog['seq_len']}**, query **1 token**.", '',
             'These graphs estimate serving operation boundaries, not captured CUDA launches. Each step carries model execution stages.',
             'The gpt-oss graph uses the existing config-only packed serving adapter and torch.compile FX capture.',
             'Other graphs are transformed from successful recorded decode traces: independent Q/K/V linears sharing an input are packed;',
             'adjacent residual add and RMSNorm are grouped; attention and recurrence are marked as native boundary estimates.',
             'Source IDs, tensor dimensions, dependencies and model-specific MLA, sparse, hybrid and expert structures are preserved.',
             'Logical intermediate tensors and checkpoint weight slices may remain visible inside estimated boundaries.',
             'Unmatched operations retain source-level grouping. Stage attribution follows module paths and operation metadata.',
             'No particular vLLM/SGLang implementation, exact fusion, quantization layout, physical cache allocation or launch count is inferred.', '',
             'Regenerate with `python generate_serving_results.py`. [Manifest](manifest.json) records input/output hashes and inference rules.', '',
             '| Model | Decode | Method |', '| --- | --- | --- |']
    for entry in catalog['models']:
        source = ROOT/'results/decode'/f"{entry['slug']}.txt"
        target = DEST/'decode'/source.name
        record = {'model_id': entry['model_id'], 'revision': entry['revision'], 'phase': 'decode',
                  'cuda_kernel_capture': False, 'cuda_fusion_validated': False,
                  'config_sha256': hashlib.sha256((ROOT/entry['config_path']).read_bytes()).hexdigest()}
        run = source_runs.get(entry['model_id'], {})
        if run.get('status') != 'success' or not source.exists():
            record.update(status='unsupported', error=run.get('error', 'No successful source decode trace'))
            cell, method = 'unsupported', record['error'].replace('|', '/') .replace('\n', ' ')
        else:
            record['source_graph'] = str(source.relative_to(ROOT))
            record['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            if entry['model_type'] == 'gpt_oss' and target.exists():
                record['method'] = 'config-only serving adapter / torch.compile FX'
                record['adapter_metadata'] = 'results/compiled/cuda-serving/gpt-oss-decode.json'
            else:
                graph, counts = estimate(source.read_text())
                target.write_text(graph)
                record.update(method='rule-based source trace estimate', boundary_counts=counts)
            graph = target.read_text()
            record.update(status='success', graph=str(target.relative_to(ROOT)),
                          graph_sha256=hashlib.sha256(graph.encode()).hexdigest(), steps=graph.count('[step '))
            cell, method = f'[graph](decode/{source.name})', record['method']
        records.append(record)
        index.append(f"| {entry['model_id']} | {cell} | {method} |")
    (DEST/'manifest.json').write_text(json.dumps({'batch': catalog['batch'], 'cached_context': catalog['seq_len'],
        'query_tokens': 1, 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': records}, indent=2)+'\n')
    update_index('serving-results', '\n'.join(index)+'\n', DEST)
    print(f"Generated/indexed {sum(r['status']=='success' for r in records)} decode graphs; {sum(r['status']!='success' for r in records)} unsupported.")


if __name__ == '__main__':
    main()
