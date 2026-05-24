#!/usr/bin/env python3
"""Build edge-as-node GNN embeddings for optical-network QA rows.

This is the first graph-side baseline for adapting GraphTranslator. The encoder
uses a heterogeneous edge-as-node graph:

site -> oms -> site
oms <-> device
service <-> oms

For each QA row, the exported embedding is the representation of its focus node
mixed with the graph-level pooled representation. The GNN weights are
deterministically initialized; this gives us a reproducible graph-structured
input format before we start training/fine-tuning a graph encoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_optical_qa import load_source_samples, parse_sample  # noqa: E402


SITE_ORDER = {site: idx + 1 for idx, site in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
NODE_TYPES = {"site": 0, "oms": 1, "device": 2, "service": 3}
FEATURE_DIM = 64


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).replace("λ", "").strip())
    except ValueError:
        return default


def one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def pad_feature(values: list[float]) -> list[float]:
    if len(values) > FEATURE_DIM:
        return values[:FEATURE_DIM]
    return values + [0.0] * (FEATURE_DIM - len(values))


def oms_number(oms_id: str) -> int:
    digits = "".join(ch for ch in oms_id if ch.isdigit())
    return int(digits) if digits else 0


def node_feature(node_type: str, attrs: dict[str, Any]) -> list[float]:
    values: list[float] = []
    values.extend(one_hot(NODE_TYPES[node_type], len(NODE_TYPES)))

    if node_type == "site":
        site_idx = SITE_ORDER.get(str(attrs.get("site", "")), 0)
        values.extend([site_idx / 26.0, 0.0, 0.0, 0.0, 0.0])
    elif node_type == "oms":
        values.extend(
            [
                oms_number(attrs.get("oms_id", "")) / 16.0,
                SITE_ORDER.get(str(attrs.get("src", "")), 0) / 26.0,
                SITE_ORDER.get(str(attrs.get("dst", "")), 0) / 26.0,
                safe_float(attrs.get("path_count", 0)) / 128.0,
                0.0,
            ]
        )
    elif node_type == "device":
        location = str(attrs.get("location", ""))
        site_num = safe_float(location.replace("Site", "").replace("Span", ""))
        edfa_type = str(attrs.get("edfa_type", ""))
        values.extend(
            [
                oms_number(attrs.get("oms_id", "")) / 16.0,
                site_num / 8.0,
                1.0 if "21" in edfa_type else 0.0,
                1.0 if "25" in edfa_type else 0.0,
                safe_float(attrs.get("gain", 0)) / 30.0,
                safe_float(attrs.get("tilt", 0)) / 10.0,
            ]
        )
    elif node_type == "service":
        path_len = len(attrs.get("path_oms_ids", []))
        values.extend(
            [
                safe_float(attrs.get("service_id", 0)) / 128.0,
                safe_float(attrs.get("lambda_id", 0)) / 64.0,
                path_len / 8.0,
                safe_float(attrs.get("q_margin", 0)) / 20.0,
                0.0,
            ]
        )

    return pad_feature(values)


def add_node(
    node_ids: list[str],
    node_features: list[list[float]],
    node_index: dict[str, int],
    node_id: str,
    node_type: str,
    attrs: dict[str, Any],
) -> None:
    if node_id in node_index:
        return
    node_index[node_id] = len(node_ids)
    node_ids.append(node_id)
    node_features.append(node_feature(node_type, attrs))


def add_edge(edges: list[tuple[int, int]], node_index: dict[str, int], src: str, dst: str) -> None:
    if src not in node_index or dst not in node_index:
        return
    edges.append((node_index[src], node_index[dst]))


def build_graph(parsed: dict[str, Any]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    node_ids: list[str] = []
    node_features: list[list[float]] = []
    node_index: dict[str, int] = {}
    edges: list[tuple[int, int]] = []

    topology = parsed["topology"]
    device = parsed["device"]
    services = parsed["services"]
    path_counts: dict[str, int] = {}
    for service in services:
        for oms_id in service["path_oms_ids"]:
            path_counts[oms_id] = path_counts.get(oms_id, 0) + 1

    for link in topology:
        add_node(node_ids, node_features, node_index, f"site:{link['src']}", "site", {"site": link["src"]})
        add_node(node_ids, node_features, node_index, f"site:{link['dst']}", "site", {"site": link["dst"]})
        add_node(
            node_ids,
            node_features,
            node_index,
            f"oms:{link['oms_id']}",
            "oms",
            {**link, "path_count": path_counts.get(link["oms_id"], 0)},
        )
        add_edge(edges, node_index, f"site:{link['src']}", f"oms:{link['oms_id']}")
        add_edge(edges, node_index, f"oms:{link['oms_id']}", f"site:{link['dst']}")
        add_edge(edges, node_index, f"oms:{link['oms_id']}", f"site:{link['src']}")
        add_edge(edges, node_index, f"site:{link['dst']}", f"oms:{link['oms_id']}")

    for (oms_id, location), params in sorted(device.items()):
        edfa_type = params.get("E.type")
        gain = params.get("E.gain_dB")
        if not edfa_type and not gain:
            continue
        node_id = f"device:{oms_id}:{location}:E"
        add_node(
            node_ids,
            node_features,
            node_index,
            node_id,
            "device",
            {
                "oms_id": oms_id,
                "location": location,
                "edfa_type": edfa_type or "",
                "gain": gain or 0,
                "tilt": params.get("E.tilt", 0),
            },
        )
        add_edge(edges, node_index, f"oms:{oms_id}", node_id)
        add_edge(edges, node_index, node_id, f"oms:{oms_id}")

    for service in services:
        service_id = service["service_id"]
        node_id = f"service:{service_id}"
        add_node(node_ids, node_features, node_index, node_id, "service", service)
        for oms_id in service["path_oms_ids"]:
            add_edge(edges, node_index, node_id, f"oms:{oms_id}")
            add_edge(edges, node_index, f"oms:{oms_id}", node_id)

    if not node_ids:
        add_node(node_ids, node_features, node_index, "site:EMPTY", "site", {"site": ""})

    x = torch.tensor(node_features, dtype=torch.float32)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return node_ids, x, edge_index


class SimpleGraphEncoder(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 256, out_dim: int = 768, layers: int = 3):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.neigh_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input(x))
        for self_layer, neigh_layer in zip(self.self_layers, self.neigh_layers):
            neigh = torch.zeros_like(h)
            if edge_index.numel() > 0:
                src, dst = edge_index
                neigh.index_add_(0, dst, h[src])
                degree = torch.zeros(h.size(0), device=h.device)
                degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
                neigh = neigh / degree.clamp_min(1.0).unsqueeze(-1)
            h = F.relu(self_layer(h) + neigh_layer(neigh))
        return F.normalize(self.output(h), dim=-1)


def focus_node_id(row: dict[str, Any]) -> str | None:
    focus_type = row.get("focus_type")
    focus_id = row.get("focus_id", "")
    if focus_type == "oms":
        return f"oms:{focus_id}"
    if focus_type == "device":
        return f"device:{focus_id}"
    if focus_type == "service":
        return f"service:{focus_id}"
    return None


def format_embedding(tensor: torch.Tensor) -> str:
    return ",".join(f"{value:.6f}" for value in tensor.tolist())


def load_qa_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--qa-jsonl", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder = SimpleGraphEncoder()
    encoder.eval()

    source_samples = load_source_samples(args.source_jsonl, args.max_samples)
    parsed_by_sample = {
        f"sample_{idx:04d}": parse_sample(text)
        for idx, text in enumerate(source_samples, start=1)
    }
    qa_rows = load_qa_rows(args.qa_jsonl)
    if args.max_samples is not None:
        allowed = set(parsed_by_sample)
        qa_rows = [row for row in qa_rows if row["sample_id"] in allowed]

    node_repr_by_sample: dict[str, tuple[list[str], torch.Tensor]] = {}
    graph_repr_by_sample: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for sample_id, parsed in parsed_by_sample.items():
            node_ids, x, edge_index = build_graph(parsed)
            node_repr = encoder(x, edge_index)
            graph_repr = F.normalize(node_repr.mean(dim=0), dim=0)
            node_repr_by_sample[sample_id] = (node_ids, node_repr)
            graph_repr_by_sample[sample_id] = graph_repr

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "optical_translator_rows.tsv"
    pt_path = args.output_dir / "optical_gnn_embeddings.pt"
    meta_path = args.output_dir / "embedding_stats.json"

    embeddings: dict[str, torch.Tensor] = {}
    missing_focus = 0
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "embedding", "producer_text", "question", "answer", "sample_id", "task_type", "subtask"])
        for row in qa_rows:
            sample_id = row["sample_id"]
            node_ids, node_repr = node_repr_by_sample[sample_id]
            graph_repr = graph_repr_by_sample[sample_id]
            node_id = focus_node_id(row)
            if node_id and node_id in node_ids:
                focus_repr = node_repr[node_ids.index(node_id)]
                embedding = F.normalize(0.7 * focus_repr + 0.3 * graph_repr, dim=0)
            else:
                missing_focus += int(node_id is not None)
                embedding = graph_repr
            embeddings[row["id"]] = embedding.cpu()
            writer.writerow(
                [
                    row["id"],
                    format_embedding(embedding.cpu()),
                    row["producer_text"],
                    row["input"],
                    row["output"],
                    sample_id,
                    row["task_type"],
                    row["subtask"],
                ]
            )

    torch.save(embeddings, pt_path)
    stats = {
        "samples": len(parsed_by_sample),
        "qa_rows": len(qa_rows),
        "embedding_dim": 768,
        "missing_focus": missing_focus,
        "tsv": str(tsv_path),
        "pt": str(pt_path),
    }
    meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
