#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import unicodedata

import torch

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


_RICH_TAG_RE = re.compile(r"<\\|.*?\\|>")


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _RICH_TAG_RE.sub("", s)
    s = re.sub(r"\\s+", "", s)
    # remove punctuation (Unicode category starts with "P")
    s = "".join(ch for ch in s if unicodedata.category(ch)[:1] != "P")
    return s.lower()


def _edit_distance(a: str, b: str) -> int:
    # Levenshtein distance, O(len(a)*len(b)) with 2-row DP
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


def _load_refs(jsonl_path: str) -> dict:
    refs = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if not key:
                raise ValueError(f"Missing key at line {ln} in {jsonl_path}")
            refs[key] = obj.get("target", "")
    return refs


@torch.inference_mode()
def _run_asr_and_cer(
    *,
    model_dir: str,
    init_param: str | None,
    remote_code: str,
    val_jsonl: str,
    language: str,
    use_itn: bool,
    device: str,
    batch_size: int,
) -> dict:
    am_kwargs = dict(
        model=model_dir,
        trust_remote_code=True,
        remote_code=remote_code,
        device=device,
        disable_update=True,
        disable_pbar=True,
        batch_size=batch_size,
    )
    if init_param:
        am_kwargs["init_param"] = init_param

    am = AutoModel(**am_kwargs)
    refs = _load_refs(val_jsonl)
    # AutoModel can take jsonl directly (reads "source"/"key")
    results = am.generate(input=val_jsonl, language=language, use_itn=use_itn, cache={})

    total_edits = 0
    total_ref = 0
    n = 0
    missing_ref = 0
    missing_key = 0

    for it in results:
        if not isinstance(it, dict):
            continue
        key = it.get("key")
        if not key:
            missing_key += 1
            continue
        ref = refs.get(key)
        if ref is None:
            missing_ref += 1
            continue
        hyp = it.get("text", "")
        hyp = rich_transcription_postprocess(hyp)
        ref_n = _normalize_text(ref)
        hyp_n = _normalize_text(hyp)
        total_edits += _edit_distance(ref_n, hyp_n)
        total_ref += len(ref_n)
        n += 1

    cer = float("inf") if total_ref == 0 else (total_edits / total_ref)
    return {
        "samples_scored": n,
        "missing_key": missing_key,
        "missing_ref": missing_ref,
        "total_ref_chars": total_ref,
        "total_edits": total_edits,
        "cer": cer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--remote_code", default=os.path.join(os.path.dirname(__file__), "model.py"))
    ap.add_argument("--base_init", default="")
    ap.add_argument("--ft_init", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--use_itn", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    base_init = args.base_init.strip() or None
    ft_init = args.ft_init.strip() or None

    base = _run_asr_and_cer(
        model_dir=args.model_dir,
        init_param=base_init,
        remote_code=args.remote_code,
        val_jsonl=args.val_jsonl,
        language=args.language,
        use_itn=bool(args.use_itn),
        device=args.device,
        batch_size=args.batch_size,
    )
    ft = _run_asr_and_cer(
        model_dir=args.model_dir,
        init_param=ft_init,
        remote_code=args.remote_code,
        val_jsonl=args.val_jsonl,
        language=args.language,
        use_itn=bool(args.use_itn),
        device=args.device,
        batch_size=args.batch_size,
    )

    print("baseline:", base)
    print("finetuned:", ft)
    if base["cer"] != float("inf") and ft["cer"] != float("inf"):
        print("abs_impr_cer:", base["cer"] - ft["cer"])
        if base["cer"] > 0:
            print("rel_impr_pct:", 100.0 * (base["cer"] - ft["cer"]) / base["cer"])


if __name__ == "__main__":
    main()

