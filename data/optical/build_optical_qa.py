#!/usr/bin/env python3
"""Build simple optical-network QA data from the markdown JSONL dataset.

The source JSONL stores each sample as a one-element JSON list whose item is a
markdown document. This script extracts the topology, device and service tables
and generates deterministic QA pairs for the first two evaluation categories:

1. Basic fact understanding.
2. Topology semantic understanding.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DASHES = {"", "-", "—", "–", "nan", "None"}


def normalize_oms_id(value: str) -> str:
    value = str(value).strip()
    match = re.search(r"(?:OMS|O)?\s*(\d+)", value, flags=re.IGNORECASE)
    if not match:
        return value
    return f"O{int(match.group(1))}"


def display_oms_id(value: str) -> str:
    oms_id = normalize_oms_id(value)
    match = re.search(r"(\d+)", oms_id)
    return f"OMS{match.group(1)}" if match else oms_id


def oms_num(value: str) -> str:
    match = re.search(r"(\d+)", normalize_oms_id(value))
    return match.group(1) if match else str(value)


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-+:?", cell.strip()) for cell in cells)


def extract_markdown_tables(text: str) -> list[dict[str, Any]]:
    lines = text.replace("\\n", "\n").splitlines()
    tables: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue

        block: list[str] = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1

        if len(block) < 2:
            continue
        header = split_markdown_row(block[0])
        rows = [split_markdown_row(row) for row in block[1:]]
        rows = [row for row in rows if not is_separator_row(row)]
        tables.append({"header": header, "rows": rows})
    return tables


def load_source_samples(path: Path, max_samples: int | None = None) -> list[str]:
    samples: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and obj and isinstance(obj[0], str):
                samples.append(obj[0])
            elif isinstance(obj, str):
                samples.append(obj)
            else:
                raise ValueError(f"Unsupported JSONL row type: {type(obj)!r}")
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def parse_topology(table: dict[str, Any]) -> list[dict[str, str]]:
    header = table["header"]
    dest_sites = header[1:]
    links: list[dict[str, str]] = []
    for row in table["rows"]:
        if len(row) < len(header):
            continue
        src = row[0]
        for dst, cell in zip(dest_sites, row[1:]):
            if cell.strip() in DASHES:
                continue
            oms_id = normalize_oms_id(cell)
            links.append(
                {
                    "oms_id": oms_id,
                    "oms_display": display_oms_id(oms_id),
                    "oms_num": oms_num(oms_id),
                    "src": src,
                    "dst": dst,
                }
            )
    return links


def parse_device(table: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    header = table["header"]
    rows = table["rows"]
    device: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if len(row) < len(header):
            continue
        record = dict(zip(header, row))
        oms_id = normalize_oms_id(record.get("OMS id", ""))
        location = record.get("Location", "").strip()
        component = record.get("Component", "").strip()
        parameter = record.get("Parameters", "").strip()
        value = record.get("Values", "").strip()
        if not oms_id or not location or value in DASHES:
            continue

        key = (oms_id, location)
        device.setdefault(key, {})
        compact_key = f"{component}.{parameter}"
        device[key][compact_key] = value
    return device


def parse_services(table: dict[str, Any]) -> list[dict[str, str]]:
    header = table["header"]
    services: list[dict[str, str]] = []
    for row in table["rows"]:
        if len(row) < len(header):
            continue
        record = dict(zip(header, row))
        service_id = record.get("Service_ID", "").strip()
        if not service_id or service_id in DASHES:
            continue
        path_raw = record.get("Optical_path(OMS)", "").strip()
        path_oms_ids = [normalize_oms_id(part) for part in re.split(r"[-,，\s]+", path_raw) if part]
        services.append(
            {
                "service_id": service_id,
                "lambda_id": record.get("λ ID", "").strip(),
                "path_raw": path_raw,
                "path_oms_ids": path_oms_ids,
                "transponder": record.get("Transponder", "").strip(),
                "q_margin": record.get("Q_margin(dB)", "").strip(),
            }
        )
    return services


def parse_sample(text: str) -> dict[str, Any]:
    tables = extract_markdown_tables(text)
    topology: list[dict[str, str]] = []
    device: dict[tuple[str, str], dict[str, str]] = {}
    services: list[dict[str, str]] = []

    for table in tables:
        header = table["header"]
        if header and header[0] == "Source/Destination":
            topology = parse_topology(table)
        elif header == ["Band", "OMS id", "Location", "Component", "Parameters", "Attribute", "Values"]:
            device = parse_device(table)
        elif header == ["Service_ID", "λ ID", "Optical_path(OMS)", "Transponder", "Q_margin(dB)"]:
            services = parse_services(table)

    return {"topology": topology, "device": device, "services": services}


def make_producer_text(sample_id: str, parsed: dict[str, Any]) -> str:
    links = parsed["topology"]
    services = parsed["services"]
    sites = sorted({link["src"] for link in links} | {link["dst"] for link in links})
    link_text = "，".join(f"{link['src']}到{link['dst']}（{link['oms_display']}）" for link in links)
    service_text = ""
    if services:
        first = services[0]
        service_text = (
            f" 示例业务{first['service_id']}使用波长{first['lambda_id']}，"
            f"路径OMS为{first['path_raw']}，Q_margin为{first['q_margin']} dB。"
        )
    return (
        f"样本{sample_id}包含{len(sites)}个站点、{len(links)}条有向OMS链路和"
        f"{len(services)}条业务。已存在的有向链路包括：{link_text}。{service_text}"
    )


def add_common_fields(
    qa: dict[str, Any],
    sample_id: str,
    focus_type: str,
    focus_id: str,
    producer_text: str,
) -> dict[str, Any]:
    qa["sample_id"] = sample_id
    qa["focus_type"] = focus_type
    qa["focus_id"] = focus_id
    qa["producer_text"] = producer_text
    return qa


def generate_qas_for_sample(sample_index: int, parsed: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = f"sample_{sample_index:04d}"
    producer_text = make_producer_text(sample_id, parsed)
    links = parsed["topology"]
    device = parsed["device"]
    services = parsed["services"]
    qas: list[dict[str, Any]] = []
    serial = 1

    def next_id(focus: str = "NET") -> str:
        nonlocal serial
        qa_id = f"HW_{sample_id}_{focus}_{serial:03d}"
        serial += 1
        return qa_id

    for link in links:
        oms_display = link["oms_display"]
        qa = {
            "id": next_id(oms_display),
            "task_type": "topology_qa",
            "subtask": "fact_extraction",
            "input": f"{oms_display}对应的链路是哪个源节点到哪个目的节点？",
            "output": f"{oms_display}对应链路为{link['src']}到{link['dst']}。",
            "answer_type": "entity",
            "evidence_span": ["topology_table"],
            "difficulty": "easy",
        }
        qas.append(add_common_fields(qa, sample_id, "oms", link["oms_id"], producer_text))

        qa = {
            "id": next_id(oms_display),
            "task_type": "topology_reasoning",
            "subtask": "reverse_relation",
            "input": f"从拓扑角度看，{oms_display}的上游节点和下游节点分别是什么？",
            "output": f"{oms_display}的上游节点是{link['src']}，下游节点是{link['dst']}。",
            "answer_type": "entity",
            "evidence_span": ["topology_table"],
            "difficulty": "easy",
        }
        qas.append(add_common_fields(qa, sample_id, "oms", link["oms_id"], producer_text))

    for (oms_id, location), params in sorted(device.items()):
        edfa_type = params.get("E.type")
        edfa_gain = params.get("E.gain_dB")
        if not edfa_type or not edfa_gain:
            continue
        oms_display = display_oms_id(oms_id)
        qa = {
            "id": next_id(oms_display),
            "task_type": "device_qa",
            "subtask": "parameter_lookup",
            "input": f"{oms_display}在{location}的EDFA类型和增益是多少？",
            "output": f"{oms_display}在{location}的EDFA类型为{edfa_type}，增益为{edfa_gain} dB。",
            "answer_type": "entity",
            "evidence_span": ["device_parameter_table"],
            "difficulty": "easy",
        }
        qas.append(add_common_fields(qa, sample_id, "device", f"{oms_id}:{location}:E", producer_text))

    for service in services:
        qa = {
            "id": next_id(f"S{service['service_id']}"),
            "task_type": "service_qa",
            "subtask": "service_lookup",
            "input": f"业务{service['service_id']}对应的波长、路径OMS和Q_margin分别是什么？",
            "output": (
                f"业务{service['service_id']}对应波长为{service['lambda_id']}，"
                f"路径OMS为{service['path_raw']}，Q_margin为{service['q_margin']} dB。"
            ),
            "answer_type": "entity",
            "evidence_span": ["service_table"],
            "difficulty": "easy",
        }
        qas.append(add_common_fields(qa, sample_id, "service", service["service_id"], producer_text))

    if links:
        link_text = "，".join(f"{link['src']}到{link['dst']}（{link['oms_display']}）" for link in links)
        qa = {
            "id": next_id("NET"),
            "task_type": "topology_reasoning",
            "subtask": "adjacency_understanding",
            "input": "请给出该网络中所有已存在的有向链路。",
            "output": f"已存在的有向链路包括：{link_text}。",
            "answer_type": "list",
            "evidence_span": ["topology_table"],
            "difficulty": "medium",
        }
        qas.append(add_common_fields(qa, sample_id, "network", sample_id, producer_text))

    if services:
        first_path = services[0]["path_oms_ids"]
        common = set(first_path)
        for service in services[1:]:
            common &= set(service["path_oms_ids"])
        first_common = next((oms_id for oms_id in first_path if oms_id in common), None)
        if first_common:
            output = f"是，所有给出的{len(services)}条业务都经过{display_oms_id(first_common)}。"
        else:
            output = f"否，所有给出的{len(services)}条业务不存在共同经过的OMS链路。"
        qa = {
            "id": next_id("NET"),
            "task_type": "topology_reasoning",
            "subtask": "path_membership",
            "input": "所有业务是否都经过同一条OMS链路？如果是，请指出是哪一条。",
            "output": output,
            "answer_type": "entity",
            "evidence_span": ["service_table"],
            "difficulty": "easy",
        }
        qas.append(add_common_fields(qa, sample_id, "network", sample_id, producer_text))

    return qas


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    source_samples = load_source_samples(args.input, args.max_samples)
    all_qas: list[dict[str, Any]] = []
    stats = {"samples": len(source_samples), "links": 0, "services": 0, "qas": 0}
    for idx, text in enumerate(source_samples, start=1):
        parsed = parse_sample(text)
        stats["links"] += len(parsed["topology"])
        stats["services"] += len(parsed["services"])
        all_qas.extend(generate_qas_for_sample(idx, parsed))

    rng = random.Random(args.seed)
    shuffled = list(all_qas)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * args.train_ratio)
    train_rows = shuffled[:split]
    test_rows = shuffled[split:]

    write_jsonl(args.output_dir / "optical_qa.jsonl", all_qas)
    write_jsonl(args.output_dir / "optical_train.jsonl", train_rows)
    write_jsonl(args.output_dir / "optical_test.jsonl", test_rows)
    stats["qas"] = len(all_qas)
    stats["train_qas"] = len(train_rows)
    stats["test_qas"] = len(test_rows)
    (args.output_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
