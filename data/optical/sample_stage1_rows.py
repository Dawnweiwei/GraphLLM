#!/usr/bin/env python3
"""Create a balanced subset of optical Stage-1 alignment rows."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_QUOTAS = {
    "oms_summary": 5200,
    "device_summary": 5000,
    "service_summary": 6600,
    "network_adjacency_summary": 1600,
    "network_common_oms_summary": 1600,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quota", action="append", default=[])
    args = parser.parse_args()

    quotas = dict(DEFAULT_QUOTAS)
    for item in args.quota:
        key, value = item.split("=", 1)
        quotas[key] = int(value)

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Input TSV is missing a header")
        for row in reader:
            buckets[row["subtask"]].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for subtask, quota in quotas.items():
        rows = buckets.get(subtask, [])
        rng.shuffle(rows)
        selected.extend(rows[: min(quota, len(rows))])
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)

    counts = {subtask: len([row for row in selected if row["subtask"] == subtask]) for subtask in sorted(quotas)}
    print({"rows": len(selected), "counts": counts})


if __name__ == "__main__":
    main()
