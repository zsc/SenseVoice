#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import html
import base64
import json
import math
import os
import random
import re
import shutil
import unicodedata
from pathlib import Path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


_RICH_TAG_RE = re.compile(r"<\|.*?\|>")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _RICH_TAG_RE.sub("", s)
    s = re.sub(r"\s+", "", s)
    return "".join(ch for ch in s if unicodedata.category(ch)[:1] != "P").lower()


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def load_rows(jsonl_path: Path):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if not key:
                continue
            rows.append(obj)
    return rows


def run_asr(model_dir: Path, remote_code: Path, init_param: str | None, jsonl: Path, language: str, use_itn: bool, batch_size: int, device: str):
    kwargs = dict(
        model=str(model_dir),
        trust_remote_code=True,
        remote_code=str(remote_code),
        device=device,
        disable_update=True,
        disable_pbar=True,
        batch_size=batch_size,
    )
    if init_param:
        kwargs["init_param"] = str(init_param)

    am = AutoModel(**kwargs)
    results = am.generate(input=str(jsonl), language=language, use_itn=use_itn, cache={})
    hyps = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not key:
            continue
        hyps[key] = rich_transcription_postprocess(item.get("text", ""))
    return hyps


def score_rows(rows, refs, base_hyps, ft_hyps):
    scored = []
    for r in rows:
        key = r["key"]
        ref = r.get("target", "")
        b = base_hyps.get(key)
        f = ft_hyps.get(key)
        if b is None or f is None:
            continue
        ref_n = normalize_text(ref)
        b_n = normalize_text(b)
        f_n = normalize_text(f)
        ref_len = len(ref_n)
        if ref_len == 0:
            continue
        b_edit = edit_distance(ref_n, b_n)
        f_edit = edit_distance(ref_n, f_n)
        scored.append(
            {
                "key": key,
                "source": r.get("source", ""),
                "ref": ref,
                "ref_n": ref_n,
                "base": b,
                "ft": f,
                "base_n": b_n,
                "ft_n": f_n,
                "ref_len": ref_len,
                "base_edit": b_edit,
                "ft_edit": f_edit,
                "base_cer": b_edit / ref_len,
                "ft_cer": f_edit / ref_len,
            }
        )
    for row in scored:
        row["delta"] = row["base_cer"] - row["ft_cer"]

    total_ref = sum(r["ref_len"] for r in scored)
    total_base = sum(r["base_edit"] for r in scored)
    total_ft = sum(r["ft_edit"] for r in scored)
    return scored, total_ref, total_base, total_ft


def select_examples(scored, k, seed):
    random.Random(seed).shuffle(scored)
    k = max(0, min(k, len(scored)))

    improved = [x for x in scored if x["delta"] > 0]
    regressed = [x for x in scored if x["delta"] < 0]
    other = [x for x in scored if x["delta"] == 0]

    n_impr = min(max(1, k // 3), len(improved)) if k else 0
    n_reg = min(max(1, k // 3), len(regressed)) if k else 0
    n_rand = k - n_impr - n_reg

    improved = sorted(improved, key=lambda x: x["delta"], reverse=True)
    regressed = sorted(regressed, key=lambda x: x["delta"])

    picked = []
    picked_idx = set()

    def add_many(items, n):
        for x in items[:n]:
            if x["key"] not in picked_idx:
                picked_idx.add(x["key"])
                picked.append(x)

    add_many(improved, n_impr)
    add_many(regressed, n_reg)

    if n_rand <= 0:
        return picked[:k]

    random.Random(seed).shuffle(other)
    random.shuffle(other)

    i = 0
    while len(picked) < k and i < len(other):
        x = other[i]
        i += 1
        if x["key"] not in picked_idx:
            picked_idx.add(x["key"])
            picked.append(x)

    if len(picked) < k:
        for x in improved + regressed:
            if x["key"] not in picked_idx:
                picked_idx.add(x["key"])
                picked.append(x)
            if len(picked) >= k:
                break

    return picked[:k]


def safe_file_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = s.strip("._")
    if not s:
        s = "sample"
    return s


def _wav_to_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    return "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")


def render_html(out_path: Path, selected, summary, inline_audio: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not inline_audio:
        audio_dir = out_path.parent / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
    else:
        audio_dir = None

    def esc(x):
        return html.escape(str(x if x is not None else ""))

    rows_html = []
    copied = {}
    for idx, r in enumerate(selected, start=1):
        src = r["source"]
        audio_tag = ""
        if src and os.path.isfile(src):
            dst_name = f"{idx:03d}_{safe_file_name(r['key'])}.wav"
            if inline_audio:
                wav_uri = _wav_to_data_uri(Path(src))
                audio_tag = (
                    f'<audio controls preload="none" src="{esc(wav_uri)}"></audio>'
                )
            else:
                dst = audio_dir / dst_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                audio_tag = (
                    f'<audio controls preload="none" src="audio/{esc(dst_name)}"></audio>'
                )
                copied[r["key"]] = str(dst)
        else:
            audio_tag = "<span class=\"missing\">audio unavailable</span>"

        row_class = "better"
        if r["delta"] < 0:
            row_class = "worse"
        elif r["delta"] == 0:
            row_class = "same"

        rows_html.append(
            f'<tr class="{row_class}">'
            f"<td>{idx}</td>"
            f"<td><code>{esc(r['key'])}</code></td>"
            f"<td><code>{r['ref_len']}</code></td>"
            f"<td>{esc(r['ref'])}</td>"
            f"<td>{esc(r['base'])}</td>"
            f"<td>{esc(r['ft'])}</td>"
            f"<td class=\"num\">{r['base_cer']:.4f}</td>"
            f"<td class=\"num\">{r['ft_cer']:.4f}</td>"
            f"<td class=\"num\">{r['delta']:.4f}</td>"
            f"<td>{audio_tag}</td>"
            f"</tr>"
        )

    style = r'''
        <style>
            body { font-family: Arial, 'Helvetica Neue', Helvetica, sans-serif; margin: 24px; }
            h1 { margin-bottom: 10px; }
            .meta { margin: 6px 0 16px 0; color: #333; }
            table { border-collapse: collapse; width: 100%; table-layout: fixed; }
            th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
            th { background: #f5f5f5; position: sticky; top: 0; }
            tr.better { background: #e8f7ea; }
            tr.worse { background: #fdf0f0; }
            tr.same { background: #f5f5f5; }
            pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
            .num { text-align: right; font-family: ui-monospace, Menlo, Consolas, monospace; }
            audio { width: 220px; }
            .missing { color: #999; font-size: 12px; }
        </style>
    '''

    header = f"""
        <h1>SenseVoice fine-tune vs baseline sample compare</h1>
        <div class=\"meta\">
            Baseline model: {esc(summary['model_dir'])}; ft checkpoint: {esc(summary['ft_init'])}<br>
            Split: {esc(summary['split'])}; device={esc(summary['device'])}; language={esc(summary['language'])}; use_itn={summary['use_itn']}<br>
            Samples scored: {summary['scored_samples']}; baseline CER={summary['base_cer']:.6f}; ft CER={summary['ft_cer']:.6f};
            abs improv={summary['abs_impr']:.6f}; relative={summary['rel_impr']:.2f}%
        </div>
    """

    table = f"""
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>key</th>
              <th>len</th>
              <th>ref</th>
              <th>baseline</th>
              <th>finetuned</th>
              <th>base CER</th>
              <th>ft CER</th>
              <th>delta(base-ft)</th>
              <th>audio</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows_html)}
          </tbody>
        </table>
    """

    with out_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html>\n")
        f.write('<html><head><meta charset="utf-8">')
        f.write(style)
        f.write(f"<title>SenseVoice CER compare</title></head><body>{header}{table}</body></html>")

    if not inline_audio:
        with (out_path.parent / "copy_map.json").open("w", encoding="utf-8") as m:
            json.dump(copied, m, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--remote_code", default=str(Path(__file__).parent / "model.py"))
    ap.add_argument("--base_init", default="")
    ap.add_argument("--ft_init", required=True)
    ap.add_argument("--out_html", default="outputs/cer_compare_run3/index.html")
    ap.add_argument("--num_examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--language", default="zh")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_itn", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--inline_audio", action="store_true", help="Embed audio bytes as base64 data URIs")
    args = ap.parse_args()

    val_jsonl = Path(args.val_jsonl)
    model_dir = Path(args.model_dir)
    remote_code = Path(args.remote_code)

    rows = load_rows(val_jsonl)
    if not rows:
        raise RuntimeError("No data in val_jsonl")

    print(f"Loaded {len(rows)} val rows")

    base_hyps = run_asr(
        model_dir=model_dir,
        remote_code=remote_code,
        init_param=args.base_init.strip() or None,
        jsonl=val_jsonl,
        language=args.language,
        use_itn=args.use_itn,
        batch_size=args.batch_size,
        device=args.device,
    )

    ft_hyps = run_asr(
        model_dir=model_dir,
        remote_code=remote_code,
        init_param=args.ft_init,
        jsonl=val_jsonl,
        language=args.language,
        use_itn=args.use_itn,
        batch_size=args.batch_size,
        device=args.device,
    )

    scored, total_ref, total_base, total_ft = score_rows(rows, rows, base_hyps, ft_hyps)
    if not scored:
        raise RuntimeError("No scored samples")

    base_cer = total_base / total_ref if total_ref else math.inf
    ft_cer = total_ft / total_ref if total_ref else math.inf
    abs_impr = base_cer - ft_cer
    rel_impr = 0.0 if base_cer == 0 else 100.0 * abs_impr / base_cer

    print(f"Scored samples: {len(scored)}")
    print(f"baseline CER: {base_cer:.8f}")
    print(f"finetuned CER: {ft_cer:.8f}")
    print(f"abs improvement: {abs_impr:.8f}")
    print(f"rel improvement: {rel_impr:.2f}%")

    selected = select_examples(scored, args.num_examples, args.seed)
    summary = {
        "model_dir": str(model_dir),
        "ft_init": args.ft_init,
        "split": str(val_jsonl),
        "device": args.device,
        "language": args.language,
        "use_itn": bool(args.use_itn),
        "scored_samples": len(scored),
        "base_cer": base_cer,
        "ft_cer": ft_cer,
        "abs_impr": abs_impr,
        "rel_impr": rel_impr,
    }
    out_html = Path(args.out_html)
    render_html(out_html, selected, summary, inline_audio=args.inline_audio)
    print(f"Report generated: {out_html}")
    if args.inline_audio:
        print("Audio embedded as base64 data URI (single-file HTML)")
    else:
        print(f"Audio+meta folder: {out_html.parent}")


if __name__ == "__main__":
    main()
