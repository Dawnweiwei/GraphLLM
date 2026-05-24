#!/usr/bin/env python3
"""Build edge-as-node GNN embeddings for optical-network QA rows.

This is the first graph-side baseline for adapting GraphTranslator. The encoder
uses a heterogeneous edge-as-node graph:

site -> oms -> site
oms <-> device
service <-> oms
service <-> lambda

For each QA row, the exported embedding is the representation of its focus node
mixed with the graph-level pooled representation. To stay close to the original
GraphTranslator pipeline, GraphSAGE is trained offline with link prediction and
then frozen when exporting embeddings for the translator.
"""

from __future__ import annotations

import argparse
import csv
import json
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
NODE_TYPES = {"site": 0, "oms": 1, "device": 2, "service": 3, "lambda": 4}
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
    elif node_type == "lambda":
        values.extend([safe_float(attrs.get("lambda_id", 0)) / 64.0, 0.0, 0.0, 0.0, 0.0])

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
        lambda_id = service.get("lambda_id", "")
        if lambda_id:
            lambda_node_id = f"lambda:{lambda_id}"
            add_node(node_ids, node_features, node_index, lambda_node_id, "lambda", {"lambda_id": lambda_id})
            add_edge(edges, node_index, node_id, lambda_node_id)
            add_edge(edges, node_index, lambda_node_id, node_id)
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


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 256, out_dim: int = 768, layers: int = 3):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.neigh_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, out_dim)
        self.type_head = nn.Linear(hidden_dim, len(NODE_TYPES))

    def encode_hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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
        return h

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.output(self.encode_hidden(x, edge_index)), dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(x, edge_index)
        return F.normalize(self.output(hidden), dim=-1), self.type_head(hidden)


def node_type_labels(node_ids: list[str], device: torch.device) -> torch.Tensor:
    labels = [NODE_TYPES[node_id.split(":", 1)[0]] for node_id in node_ids]
    return torch.tensor(labels, dtype=torch.long, device=device)


def sample_negative_edges(
    num_nodes: int,
    positive_edges: torch.Tensor,
    count: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    if num_nodes <= 1 or count <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    positive = set(zip(positive_edges[0].tolist(), positive_edges[1].tolist()))
    negatives: set[tuple[int, int]] = set()
    max_attempts = max(count * 20, 100)
    attempts = 0
    while len(negatives) < count and attempts < max_attempts:
        src = rng.randrange(num_nodes)
        dst = rng.randrange(num_nodes)
        attempts += 1
        if src == dst or (src, dst) in positive or (src, dst) in negatives:
            continue
        negatives.add((src, dst))

    if not negatives:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(sorted(negatives), dtype=torch.long, device=device).t().contiguous()


def train_graphsage(
    encoder: GraphSAGEEncoder,
    graphs: dict[str, tuple[list[str], torch.Tensor, torch.Tensor]],
    train_sample_ids: set[str],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    train_ids = sorted(sample_id for sample_id in train_sample_ids if sample_id in graphs)
    if not train_ids:
        raise ValueError("No train graphs were found for GraphSAGE training.")

    encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    final_stats: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        encoder.train()
        rng.shuffle(train_ids)
        total_loss = 0.0
        total_link_loss = 0.0
        total_type_loss = 0.0
        total_link_correct = 0
        total_link_count = 0
        total_type_correct = 0
        total_type_count = 0

        for sample_id in train_ids:
            node_ids, x_cpu, edge_cpu = graphs[sample_id]
            x = x_cpu.to(device)
            edge_index = edge_cpu.to(device)
            labels = node_type_labels(node_ids, device)

            optimizer.zero_grad(set_to_none=True)
            embeddings, type_logits = encoder(x, edge_index)

            if edge_index.numel() > 0:
                pos_edges = edge_index
                neg_edges = sample_negative_edges(x.size(0), pos_edges, pos_edges.size(1), rng, device)
                pos_scores = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=-1)
                if neg_edges.numel() > 0:
                    neg_scores = (embeddings[neg_edges[0]] * embeddings[neg_edges[1]]).sum(dim=-1)
                    scores = torch.cat([pos_scores, neg_scores])
                    targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                else:
                    scores = pos_scores
                    targets = torch.ones_like(pos_scores)
                link_loss = F.binary_cross_entropy_with_logits(scores, targets)
                predictions = (torch.sigmoid(scores) >= 0.5).long()
                total_link_correct += int((predictions == targets.long()).sum().item())
                total_link_count += int(targets.numel())
            else:
                link_loss = embeddings.sum() * 0.0

            type_loss = F.cross_entropy(type_logits, labels)
            loss = link_loss + 0.2 * type_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_link_loss += float(link_loss.detach().cpu())
            total_type_loss += float(type_loss.detach().cpu())
            total_type_correct += int((type_logits.argmax(dim=-1) == labels).sum().item())
            total_type_count += int(labels.numel())

        denom = max(len(train_ids), 1)
        final_stats = {
            "epoch": float(epoch),
            "loss": total_loss / denom,
            "link_loss": total_link_loss / denom,
            "type_loss": total_type_loss / denom,
            "link_accuracy": total_link_correct / max(total_link_count, 1),
            "type_accuracy": total_type_correct / max(total_type_count, 1),
        }
        print(json.dumps({"graphsage_train": final_stats}, ensure_ascii=False))

    encoder.eval()
    return final_stats


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
        split = "train" if "train" in path.stem else "test" if "test" in path.stem else "all"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_split"] = split
                rows.append(row)
    return rows


def sample_ids_for_split(rows: list[dict[str, Any]], split: str) -> set[str]:
    return {row["sample_id"] for row in rows if row.get("_split") == split}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--qa-jsonl", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sage", action="store_true")
    parser.add_argument("--sage-epochs", type=int, default=5)
    parser.add_argument("--sage-lr", type=float, default=1e-3)
    parser.add_argument("--sage-weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    encoder = GraphSAGEEncoder()

    source_samples = load_source_samples(args.source_jsonl, args.max_samples)
    parsed_by_sample = {
        f"sample_{idx:04d}": parse_sample(text)
        for idx, text in enumerate(source_samples, start=1)
    }
    qa_rows = load_qa_rows(args.qa_jsonl)
    if args.max_samples is not None:
        allowed = set(parsed_by_sample)
        qa_rows = [row for row in qa_rows if row["sample_id"] in allowed]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "optical_translator_rows.tsv"
    split_tsv_paths = {
        "train": args.output_dir / "optical_train_translator_rows.tsv",
        "test": args.output_dir / "optical_test_translator_rows.tsv",
    }
    pt_path = args.output_dir / "optical_gnn_embeddings.pt"
    checkpoint_path = args.checkpoint or (args.output_dir / "graphsage_encoder.pt")
    meta_path = args.output_dir / "embedding_stats.json"

    graphs = {
        sample_id: build_graph(parsed)
        for sample_id, parsed in parsed_by_sample.items()
    }

    train_stats = None
    if args.train_sage:
        train_stats = train_graphsage(
            encoder,
            graphs,
            sample_ids_for_split(qa_rows, "train"),
            epochs=args.sage_epochs,
            lr=args.sage_lr,
            weight_decay=args.sage_weight_decay,
            seed=args.seed,
            device=device,
        )
        torch.save(
            {
                "model_state_dict": encoder.state_dict(),
                "node_types": NODE_TYPES,
                "feature_dim": FEATURE_DIM,
                "train_stats": train_stats,
            },
            checkpoint_path,
        )
    elif checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(checkpoint["model_state_dict"])
        train_stats = checkpoint.get("train_stats")

    encoder.to(device)
    encoder.eval()

    node_repr_by_sample: dict[str, tuple[list[str], torch.Tensor]] = {}
    graph_repr_by_sample: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for sample_id, (node_ids, x_cpu, edge_cpu) in graphs.items():
            node_repr = encoder.encode(x_cpu.to(device), edge_cpu.to(device)).cpu()
            graph_repr = F.normalize(node_repr.mean(dim=0), dim=0)
            node_repr_by_sample[sample_id] = (node_ids, node_repr)
            graph_repr_by_sample[sample_id] = graph_repr

    embeddings: dict[str, torch.Tensor] = {}
    missing_focus = 0
    header = ["id", "embedding", "producer_text", "question", "answer", "sample_id", "task_type", "subtask", "split"]
    split_files = {
        split: path.open("w", encoding="utf-8", newline="")
        for split, path in split_tsv_paths.items()
    }
    try:
        split_writers = {split: csv.writer(file, delimiter="\t") for split, file in split_files.items()}
        for writer in split_writers.values():
            writer.writerow(header)
        with tsv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
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
                output_row = [
                    row["id"],
                    format_embedding(embedding.cpu()),
                    row["producer_text"],
                    row["input"],
                    row["output"],
                    sample_id,
                    row["task_type"],
                    row["subtask"],
                    row["_split"],
                ]
                writer.writerow(output_row)
                split_writer = split_writers.get(row["_split"])
                if split_writer is not None:
                    split_writer.writerow(output_row)
    finally:
        for file in split_files.values():
            file.close()

    torch.save(embeddings, pt_path)
    stats = {
        "samples": len(parsed_by_sample),
        "qa_rows": len(qa_rows),
        "embedding_dim": 768,
        "missing_focus": missing_focus,
        "tsv": str(tsv_path),
        "train_tsv": str(split_tsv_paths["train"]),
        "test_tsv": str(split_tsv_paths["test"]),
        "pt": str(pt_path),
        "checkpoint": str(checkpoint_path),
        "train_sage": args.train_sage,
        "sage_train_stats": train_stats,
    }
    meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
