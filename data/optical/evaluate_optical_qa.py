#!/usr/bin/env python3
"""Task-aware evaluation for optical-network QA generations.

The original exact-match metric is useful for regression tests, but it is too
strict for user-facing QA. This evaluator keeps exact match as a reference and
adds slot/entity metrics for each optical-network task.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SITE_RE = re.compile(r"(?<![A-Z0-9])([A-F])(?![A-Z0-9])", re.IGNORECASE)
OMS_RE = re.compile(r"(?:OMS|O)\s*([0-9]+)", re.IGNORECASE)
LAMBDA_RE = re.compile(r"(?:lambda|λ|位)\s*([0-9]+)", re.IGNORECASE)
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
EDFA_RE = re.compile(r"(?<![A-Z0-9])([0-9]{2}S)(?![A-Z0-9])", re.IGNORECASE)
EDGE_TO_RE = re.compile(r"([A-F])\s*(?:到|->|-->|至|to)\s*([A-F])", re.IGNORECASE)
EDGE_WITH_OMS_RE = re.compile(
    r"([A-F])\s*(?:到|->|-->|至|to)\s*([A-F]).{0,12}?(?:OMS|O)\s*([0-9]+)",
    re.IGNORECASE,
)


@dataclass
class Bucket:
    count: int = 0
    sums: Counter[str] = field(default_factory=Counter)
    metric_counts: Counter[str] = field(default_factory=Counter)

    def add(self, metrics: dict[str, float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            self.sums[key] += float(value)
            self.metric_counts[key] += 1

    def averages(self) -> dict[str, float]:
        if not self.count:
            return {}
        return {
            key: value / self.metric_counts[key]
            for key, value in sorted(self.sums.items())
            if self.metric_counts[key]
        }


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("\\n", "\n")
    text = re.sub(r"\s+", "", text)
    return text.upper()


def contains_norm(pred: str, gold: str) -> float:
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    if not pred_n or not gold_n:
        return 0.0
    return float(gold_n in pred_n or pred_n in gold_n)


def extract_sites(text: str) -> list[str]:
    return [m.group(1).upper() for m in SITE_RE.finditer(text or "")]


def extract_oms_nums(text: str) -> list[str]:
    return [str(int(m.group(1))) for m in OMS_RE.finditer(text or "")]


def extract_lambda_nums(text: str) -> list[str]:
    return [str(int(m.group(1))) for m in LAMBDA_RE.finditer(text or "")]


def extract_floats(text: str) -> list[float]:
    return [float(m.group(0)) for m in FLOAT_RE.finditer(text or "")]


def extract_edfa_types(text: str) -> list[str]:
    return [m.group(1).upper() for m in EDFA_RE.finditer(text or "")]


def float_present(pred: str, target: float, tol: float = 0.05) -> bool:
    return any(abs(value - target) <= tol for value in extract_floats(pred))


def first_pair_sites(text: str) -> tuple[str | None, str | None]:
    sites = extract_sites(text)
    if len(sites) < 2:
        return None, None
    return sites[0], sites[1]


def direction_metrics(gold: str, pred: str) -> dict[str, float]:
    src, dst = first_pair_sites(gold)
    if src is None or dst is None:
        return {"slot_accuracy": 0.0, "direction_accuracy": 0.0}
    pred_sites = extract_sites(pred)
    src_ok = src in pred_sites
    dst_ok = dst in pred_sites
    order_ok = False
    if src_ok and dst_ok:
        src_pos = normalize_text(pred).find(src)
        dst_pos = normalize_text(pred).find(dst)
        order_ok = 0 <= src_pos < dst_pos
    return {
        "src_accuracy": float(src_ok),
        "dst_accuracy": float(dst_ok),
        "slot_accuracy": (float(src_ok) + float(dst_ok)) / 2.0,
        "direction_accuracy": float(order_ok),
    }


def device_metrics(gold: str, pred: str) -> dict[str, float]:
    gold_types = extract_edfa_types(gold)
    pred_types = set(extract_edfa_types(pred))
    type_ok = not gold_types or gold_types[0] in pred_types

    # In these QA rows, the EDFA gain is the last number before "dB" in the
    # gold answer after the OMS id. Comparing against any generated numeric
    # mention makes the metric robust to phrasing such as "19" vs "19.0 dB".
    gold_numbers = extract_floats(gold)
    gain_target = gold_numbers[-1] if gold_numbers else None
    gain_ok = gain_target is not None and float_present(pred, gain_target)

    slots = [float(type_ok)]
    if gain_target is not None:
        slots.append(float(gain_ok))
    return {
        "edfa_type_accuracy": float(type_ok),
        "gain_accuracy": float(gain_ok),
        "slot_accuracy": sum(slots) / len(slots),
    }


def service_metrics(gold: str, pred: str) -> dict[str, float]:
    gold_lambdas = extract_lambda_nums(gold)
    pred_lambdas = set(extract_lambda_nums(pred))
    lambda_ok = bool(gold_lambdas) and gold_lambdas[0] in pred_lambdas

    gold_oms = set(extract_oms_nums(gold))
    pred_oms = set(extract_oms_nums(pred))
    oms_intersection = gold_oms & pred_oms
    path_precision = len(oms_intersection) / len(pred_oms) if pred_oms else 0.0
    path_recall = len(oms_intersection) / len(gold_oms) if gold_oms else 0.0
    path_f1 = 2 * path_precision * path_recall / (path_precision + path_recall) if path_precision + path_recall else 0.0

    gold_numbers = extract_floats(gold)
    q_margin_target = gold_numbers[-1] if gold_numbers else None
    q_margin_ok = q_margin_target is not None and float_present(pred, q_margin_target)
    slots = [float(lambda_ok), path_f1]
    if q_margin_target is not None:
        slots.append(float(q_margin_ok))
    return {
        "lambda_accuracy": float(lambda_ok),
        "path_oms_f1": path_f1,
        "q_margin_accuracy": float(q_margin_ok),
        "slot_accuracy": sum(slots) / len(slots),
    }


def extract_edges(text: str) -> set[tuple[str, str, str | None]]:
    edges: set[tuple[str, str, str | None]] = set()
    for src, dst, oms in EDGE_WITH_OMS_RE.findall(text or ""):
        edges.add((src.upper(), dst.upper(), str(int(oms))))
    if edges:
        return edges
    for src, dst in EDGE_TO_RE.findall(text or ""):
        edges.add((src.upper(), dst.upper(), None))
    return edges


def edge_set_metrics(gold: str, pred: str) -> dict[str, float]:
    gold_edges = extract_edges(gold)
    pred_edges = extract_edges(pred)
    if not gold_edges:
        return {"edge_set_f1": 0.0}

    def edge_key(edge: tuple[str, str, str | None]) -> tuple[str, str]:
        return edge[0], edge[1]

    gold_pairs = {edge_key(edge) for edge in gold_edges}
    pred_pairs = {edge_key(edge) for edge in pred_edges}
    intersection = gold_pairs & pred_pairs
    precision = len(intersection) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(intersection) / len(gold_pairs) if gold_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "edge_set_precision": precision,
        "edge_set_recall": recall,
        "edge_set_f1": f1,
        "slot_accuracy": f1,
    }


def path_membership_metrics(gold: str, pred: str) -> dict[str, float]:
    gold_positive = bool(extract_oms_nums(gold))
    pred_positive = bool(extract_oms_nums(pred)) or any(token in pred for token in ["是", "YES", "共同", "都经过"])
    yes_no_ok = gold_positive == pred_positive

    gold_oms = set(extract_oms_nums(gold))
    pred_oms = set(extract_oms_nums(pred))
    common_ok = bool(gold_oms) and bool(gold_oms & pred_oms)
    if not gold_positive:
        common_ok = not pred_oms
    return {
        "yes_no_accuracy": float(yes_no_ok),
        "common_oms_accuracy": float(common_ok),
        "slot_accuracy": (float(yes_no_ok) + float(common_ok)) / 2.0,
    }


def generic_entity_metrics(gold: str, pred: str) -> dict[str, float]:
    gold_entities = set(extract_sites(gold)) | {f"OMS{n}" for n in extract_oms_nums(gold)}
    pred_entities = set(extract_sites(pred)) | {f"OMS{n}" for n in extract_oms_nums(pred)}
    if not gold_entities:
        return {"entity_recall": 0.0}
    return {"entity_recall": len(gold_entities & pred_entities) / len(gold_entities)}


def task_metrics(row: dict[str, str]) -> dict[str, float]:
    gold = row["answer"]
    pred = row["prediction"]
    subtask = row.get("subtask", "")
    metrics = {
        "exact_match": float(normalize_text(pred) == normalize_text(gold)),
        "contains": contains_norm(pred, gold),
        **generic_entity_metrics(gold, pred),
    }
    if subtask in {"fact_extraction", "reverse_relation"}:
        metrics.update(direction_metrics(gold, pred))
    elif subtask == "parameter_lookup":
        metrics.update(device_metrics(gold, pred))
    elif subtask == "service_lookup":
        metrics.update(service_metrics(gold, pred))
    elif subtask == "adjacency_understanding":
        metrics.update(edge_set_metrics(gold, pred))
    elif subtask == "path_membership":
        metrics.update(path_membership_metrics(gold, pred))
    return metrics


def load_gold_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["id"]: row for row in reader}


def load_prediction_rows(pred_path: Path, gold_meta: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with pred_path.open("r", encoding="utf-8", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                row_id, question, answer, prediction = parts[0], parts[1], parts[2], "\t".join(parts[3:])
            elif len(parts) == 2:
                row_id, prediction = parts
                meta = gold_meta.get(row_id, {})
                question = meta.get("question", "")
                answer = meta.get("answer", "")
            else:
                row_id = str(line_no)
                prediction = parts[0]
                meta = gold_meta.get(row_id, {})
                question = meta.get("question", "")
                answer = meta.get("answer", "")
            meta = gold_meta.get(row_id, {})
            rows.append(
                {
                    "id": row_id,
                    "question": question or meta.get("question", ""),
                    "answer": answer or meta.get("answer", ""),
                    "prediction": prediction,
                    "task_type": meta.get("task_type", ""),
                    "subtask": meta.get("subtask", ""),
                    "sample_id": meta.get("sample_id", ""),
                }
            )
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    overall = Bucket()
    by_subtask: dict[str, Bucket] = defaultdict(Bucket)
    by_task: dict[str, Bucket] = defaultdict(Bucket)
    examples: list[dict[str, Any]] = []

    for row in rows:
        metrics = task_metrics(row)
        overall.add(metrics)
        by_subtask[row.get("subtask", "")].add(metrics)
        by_task[f"{row.get('task_type', '')}/{row.get('subtask', '')}"].add(metrics)
        if len(examples) < 20 and metrics.get("slot_accuracy", metrics.get("exact_match", 0.0)) < 1.0:
            examples.append(
                {
                    "id": row["id"],
                    "subtask": row.get("subtask", ""),
                    "question": row.get("question", ""),
                    "gold": row["answer"],
                    "prediction": row["prediction"],
                    "metrics": metrics,
                }
            )

    return {
        "count": overall.count,
        "overall": overall.averages(),
        "by_task": {key: bucket.averages() | {"count": bucket.count} for key, bucket in sorted(by_task.items())},
        "by_subtask": {key: bucket.averages() | {"count": bucket.count} for key, bucket in sorted(by_subtask.items())},
        "examples": examples,
    }


def print_report(summary: dict[str, Any]) -> None:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    print(f"Rows: {summary['count']}")
    print("Overall:")
    for key, value in summary["overall"].items():
        print(f"  {key}: {pct(value)}")
    print("\nBy task:")
    for key, values in summary["by_task"].items():
        count = values.pop("count")
        metric_text = ", ".join(f"{name}={pct(value)}" for name, value in values.items())
        print(f"  {key} (n={count}): {metric_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--gold-tsv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = load_prediction_rows(args.pred, load_gold_metadata(args.gold_tsv))
    summary = summarize(rows)
    print_report(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
