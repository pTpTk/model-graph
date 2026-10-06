#!/usr/bin/env python3
"""Generate pinned, separately indexed prefill and decode graphs from a catalog."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def generate(entry, phase, batch, seq_len, timeout):
    slug = entry["slug"]
    target = RESULTS / phase / f"{slug}.txt"
    temporary = target.with_suffix(".txt.partial")
    log_path = RESULTS / "logs" / f"{slug}.{phase}.log"
    command = [str(Path(sys.executable).parent / "model-graph"), entry["model_id"],
               "--batch", str(batch), "--seq-len", str(seq_len), "--phase", phase,
               "--revision", entry["revision"], "--local-files-only"]
    record = {"model_id": entry["model_id"], "phase": phase,
              "revision": entry["revision"], "command": command,
              "log": str(log_path.relative_to(ROOT)),
              "tool_sha256": hashlib.sha256(b"".join((ROOT / name).read_bytes()
                                        for name in ("model_graph.py", "model_ops.py", "kimi_k3_graph.py"))).hexdigest()}
    started = time.monotonic()
    try:
        with temporary.open("w") as stdout, log_path.open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout,
                                    stdin=subprocess.DEVNULL, cwd=ROOT)
        if result.returncode != 0:
            error_lines = log_path.read_text().splitlines()
            record.update(status="failed", exit_code=result.returncode,
                          error=next((line for line in error_lines if line.startswith("model-graph:")
                                      and ("Error:" in line or "Exception:" in line)),
                                     "\n".join(error_lines[-4:])))
        else:
            output = temporary.read_text()
            expected = f"Tensor(shape=({batch}, {seq_len if phase == 'prefill' else 1})"
            if expected not in output or "[step " not in output or not output.splitlines()[-1].startswith("return "):
                raise ValueError("Graph output failed phase/dimension validation")
            if phase == "decode" and "cache_input" not in output:
                raise ValueError("Decode output has no initialized cache input tensors")
            temporary.replace(target)
            record.update(status="success", graph=str(target.relative_to(ROOT)),
                          graph_sha256=hashlib.sha256(output.encode()).hexdigest(),
                          lines=len(output.splitlines()), steps=output.count("[step "))
    except subprocess.TimeoutExpired:
        record.update(status="timeout", error=f"Exceeded {timeout}s timeout")
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        temporary.unlink(missing_ok=True)
    record["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return record


def render_index(catalog, records):
    rows = {(record["model_id"], record["phase"]): record for record in records}
    lines = ["# Model execution graphs", "",
             "Text-only inference graphs at batch **32** and prompt/cached-context length **8192**.", "",
             "- `prefill/`: all 8192 prompt tokens, no preceding cache; cache output disabled.",
             "- `decode/`: one new token after a shape-only 8192-token prefill; caching enabled.",
             "  Standard full-attention KV grows to 8193 positions. Sliding/compressed and recurrent caches follow their native layouts.",
             "- `configs/`: original configuration JSON, pinned to the revision in `catalog.json`.",
             "- `manifest.json`: per-phase commands, source revision, tool/output hashes, timing and status.",
             "- `logs/`: stderr diagnostics, including unsupported models and tracing failures.", "",
             "MoE expert dispatch and GLM5 Next / Qwen4 Exp sparse indexers use reviewed native boundary shape contracts.",
             "Routing, shared experts, projections and other fixed-shape operations remain visible.",
             "Per-expert token counts and sparse-indexer selected positions require tensor values and are explicitly labeled data-dependent.",
             "Floating dtype annotations are omitted for quantized configurations.", "",
             "Kimi K3 uses the reviewed local text adapter in `kimi_k3_graph.py`, preserving SiTU, latent MoE, attention residuals and modified MLA.",
             "Its KDA recurrence is an explicit boundary shape contract. [Source revision and hashes](kimi-k3-source.json) document the adaptation.", "",
             "This is a representative catalog of public general-purpose model architecture families, not a benchmark ranking.",
             "A failed entry has no substitute or fabricated execution graph.", "",
             "| Model | Architecture | Prefill | Decode |", "| --- | --- | --- | --- |"]
    for entry in catalog["models"]:
        cells=[]
        for phase in ("prefill", "decode"):
            record=rows.get((entry["model_id"],phase))
            if record and record["status"]=="success":
                graph=Path(record["graph"]).relative_to("results")
                cells.append(f"[graph]({graph.as_posix()}) ({record.get('steps', '?')} steps)")
            elif record:
                cells.append(f"{record['status']} ([log]({Path(record['log']).relative_to('results').as_posix()}))" if record.get("log") else record["status"])
            elif entry["status"]!="config_ready":
                cells.append("configuration unavailable")
            else:
                cells.append("pending")
        model=f"[{entry['model_id']}](https://huggingface.co/{entry['model_id']})"
        architecture=entry.get("text_model_type") or entry.get("model_type") or "unavailable"
        lines.append(f"| {model} | `{architecture}` | {' | '.join(cells)} |")
    unsuccessful = [entry for entry in catalog["models"]
                    if any(rows.get((entry["model_id"], phase), {}).get("status") != "success"
                           for phase in ("prefill", "decode"))]
    if unsuccessful:
        lines += ["", "## Unsupported or incomplete entries", ""]
        for entry in unsuccessful:
            reasons = list(dict.fromkeys(rows.get((entry["model_id"], phase), {}).get("error", "")
                                         for phase in ("prefill", "decode")))
            reason = entry.get("unsupported_reason") or "; ".join(reason for reason in reasons if reason)
            if reason:
                lines.append(f"- **{entry['model_id']}**: {reason}")
    lines += ["", "Regenerate with `python generate_results.py`; use `--model OWNER/NAME` to select an entry.",
              "Successful outputs are retained unless `--force` is supplied. Failed entries are retried."]
    (RESULTS/"README.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",action="append")
    parser.add_argument("--phase",choices=("both","prefill","decode"),default="both")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--timeout",type=int,default=600)
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    catalog=json.loads((RESULTS/"catalog.json").read_text())
    path=RESULTS/"manifest.json"
    records=json.loads(path.read_text()).get("runs",[]) if path.exists() else []
    indexed={(item["model_id"],item["phase"]):item for item in records}
    selected=[entry for entry in catalog["models"] if not args.model or entry["model_id"] in args.model]
    jobs=[]
    phases=("prefill","decode") if args.phase=="both" else (args.phase,)
    for entry in selected:
        if entry["status"]!="config_ready":
            continue
        for phase in phases:
            previous=indexed.get((entry["model_id"],phase))
            if previous and previous["status"]=="success" and not args.force:
                continue
            jobs.append((entry,phase))
    def save():
        current=list(indexed.values())
        write_json(path,{"updated_utc":datetime.now(timezone.utc).isoformat(),
                         "batch":catalog["batch"],"seq_len":catalog["seq_len"],"runs":current})
        render_index(catalog,current)
    save()
    print(f"Generating {len(jobs)} graphs with {args.workers} workers",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(generate,entry,phase,catalog["batch"],catalog["seq_len"],args.timeout):(entry,phase)
                 for entry,phase in jobs}
        for future in as_completed(futures):
            result=future.result()
            indexed[(result["model_id"],result["phase"])]=result
            save()
            print(f"{result['status']}: {result['model_id']} {result['phase']} ({result['elapsed_seconds']}s)"
                  + (f": {result['error'][:220]}" if result.get('error') else ""),flush=True)
    save()


if __name__=="__main__":
    main()
