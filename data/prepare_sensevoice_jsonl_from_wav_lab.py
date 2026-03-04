#!/usr/bin/env python3
"""
Prepare SenseVoice jsonl (train/val) from a directory containing paired *.wav and *.lab files.

Input layout example:
  /mnt/sda2/中文 - Chinese/<speaker_or_group>/xxx.wav
  /mnt/sda2/中文 - Chinese/<speaker_or_group>/xxx.lab

Output jsonl format follows data/train_example.jsonl in this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


PUNC_CHARS = set(
    "，。！？；：、"
    "“”‘’（）《》【】"
    "…—·"
    ",.!?;:\"'()[]{}<>-"
)


def _has_punc(text: str) -> bool:
    return any(ch in PUNC_CHARS for ch in text)


def _read_lab_text(lab_path: Path) -> str:
    # Most .lab files in this dataset are single-line UTF-8 text.
    # Keep it robust to occasional encoding oddities.
    raw = lab_path.read_text(encoding="utf-8", errors="replace")
    # Collapse whitespace/newlines
    text = " ".join(raw.strip().split())
    return text


def _target_len(text: str) -> int:
    # For zh data, SenseVoice examples use character count (punct counts, spaces do not).
    return sum(1 for ch in text if not ch.isspace())


def _source_len_10ms(wav_path: Path) -> int:
    info = sf.info(str(wav_path))
    if info.samplerate <= 0:
        return 0
    dur_s = float(info.frames) / float(info.samplerate)
    # 10ms hop => ~100 frames/sec. Use rounding to be stable.
    return max(1, int(dur_s * 100.0 + 0.5))


@dataclass
class Row:
    key: str
    text_language: str
    emo_target: str
    event_target: str
    with_or_wo_itn: str
    target: str
    source: str
    target_len: int
    source_len: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        required=True,
        help="Root dir containing wav/lab pairs (e.g. /mnt/sda2/中文 - Chinese)",
    )
    ap.add_argument("--out_train", default="data/train_chinese.jsonl")
    ap.add_argument("--out_val", default="data/val_chinese.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_dur_s", type=float, default=0.1)
    ap.add_argument("--max_dur_s", type=float, default=30.0)
    ap.add_argument("--language", default="<|zh|>")
    ap.add_argument("--emo_target", default="<|EMO_UNKNOWN|>")
    ap.add_argument("--event_target", default="<|Speech|>")
    ap.add_argument("--limit", type=int, default=0, help="For debug: limit rows (0 = no limit)")
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"ERROR: in_dir not found: {in_dir}", file=sys.stderr)
        return 2

    rows: list[Row] = []
    seen_keys: set[str] = set()

    skipped_no_lab = 0
    skipped_empty = 0
    skipped_dur = 0
    skipped_err = 0

    for root, _dirs, files in os.walk(in_dir):
        for fn in files:
            if not fn.lower().endswith(".wav"):
                continue

            wav_path = Path(root) / fn
            lab_path = wav_path.with_suffix(".lab")
            if not lab_path.exists():
                skipped_no_lab += 1
                continue

            try:
                text = _read_lab_text(lab_path)
                if not text:
                    skipped_empty += 1
                    continue

                # Duration filter (fast header read)
                info = sf.info(str(wav_path))
                if info.samplerate <= 0:
                    skipped_err += 1
                    continue
                dur_s = float(info.frames) / float(info.samplerate)
                if dur_s < args.min_dur_s or dur_s > args.max_dur_s:
                    skipped_dur += 1
                    continue

                source_len = max(1, int(dur_s * 100.0 + 0.5))
                tgt_len = _target_len(text)

                key = wav_path.stem
                if key in seen_keys:
                    # Avoid collisions while keeping key stable-ish.
                    # Use relative path to generate a deterministic suffix.
                    rel = str(wav_path.relative_to(in_dir))
                    suf = hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
                    key = f"{key}__{suf}"
                seen_keys.add(key)

                with_or_wo_itn = "<|withitn|>" if _has_punc(text) else "<|woitn|>"

                rows.append(
                    Row(
                        key=key,
                        text_language=args.language,
                        emo_target=args.emo_target,
                        event_target=args.event_target,
                        with_or_wo_itn=with_or_wo_itn,
                        target=text,
                        source=str(wav_path),
                        target_len=tgt_len,
                        source_len=source_len,
                    )
                )

                if args.limit and len(rows) >= args.limit:
                    break
            except Exception:
                skipped_err += 1
                continue

        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print(
            "ERROR: no usable wav/lab pairs found. "
            "Check directory layout and file extensions.",
            file=sys.stderr,
        )
        return 3

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)

    val_n = int(len(rows) * args.val_ratio)
    val_n = max(1, val_n) if len(rows) >= 2 else 0

    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    out_train.write_text("\n".join(r.to_json() for r in train_rows) + "\n", encoding="utf-8")
    out_val.write_text("\n".join(r.to_json() for r in val_rows) + "\n", encoding="utf-8")

    print(f"Prepared jsonl from: {in_dir}")
    print(f"Train: {out_train} ({len(train_rows)} samples)")
    print(f"Val:   {out_val} ({len(val_rows)} samples)")
    print(
        "Skipped:"
        f" no_lab={skipped_no_lab}, empty_text={skipped_empty},"
        f" dur_out_of_range={skipped_dur}, errors={skipped_err}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
