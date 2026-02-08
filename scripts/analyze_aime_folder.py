#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def iter_json_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix == ".json" and p.is_file())


def load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_len(value) -> int:
    return len(value) if isinstance(value, list) else 0


def analyze_directory(directory: Path) -> Dict[str, float]:
    json_files = iter_json_files(directory)
    items: List[Dict] = []
    for path in json_files:
        obj = load_json(path)
        if obj is not None:
            items.append(obj)

    total_time = 0.0
    speed_sum = 0.0
    speed_count = 0
    total_gen_tokens = 0
    total_full_tokens = 0

    for item in items:
        time_taken = item.get("time_taken")
        if isinstance(time_taken, (int, float)):
            total_time += float(time_taken)

        speed = item.get("speed")
        if isinstance(speed, (int, float)):
            speed_sum += float(speed)
            speed_count += 1

        total_gen_tokens += safe_len(item.get("generation_tokens"))
        total_full_tokens += safe_len(item.get("full_tokens"))

    avg_speed = (speed_sum / speed_count) if speed_count else 0.0
    weighted_speed_gen = (total_gen_tokens / total_time) if total_time > 0 else 0.0
    weighted_speed_full = (total_full_tokens / total_time) if total_time > 0 else 0.0

    return {
        "json_files": len(json_files),
        "records": len(items),
        "total_time": total_time,
        "avg_speed": avg_speed,
        "speed_sum": speed_sum,
        "speed_count": speed_count,
        "weighted_speed_gen": weighted_speed_gen,
        "weighted_speed_full": weighted_speed_full,
        "total_gen_tokens": total_gen_tokens,
        "total_full_tokens": total_full_tokens,
    }


def find_directories_with_prefix(prefix: str, base_dir: Path) -> List[Path]:
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix))


def format_time(seconds: float) -> str:
    minutes = seconds / 60.0
    return f"{seconds:.2f} s ({minutes:.2f} min)"


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(f"{label}")
    print(f"  JSON files: {metrics['json_files']}")
    print(f"  Records: {metrics['records']}")
    print(f"  Total time: {format_time(metrics['total_time'])}")
    print(f"  Avg speed (mean of per-file speed): {metrics['avg_speed']:.2f} tokens/s")
    print(f"  Overall speed (total gen tokens / total time): {metrics['weighted_speed_gen']:.2f} tokens/s")
    print(f"  Overall speed (total full tokens / total time): {metrics['weighted_speed_full']:.2f} tokens/s")
    print(f"  Total generation tokens: {metrics['total_gen_tokens']}")
    print(f"  Total full tokens: {metrics['total_full_tokens']}")


def merge_metrics(metrics_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    total = {
        "json_files": 0,
        "records": 0,
        "total_time": 0.0,
        "avg_speed": 0.0,  # recompute from speeds; keep placeholder
        "speed_sum": 0.0,
        "speed_count": 0,
        "weighted_speed_gen": 0.0,
        "weighted_speed_full": 0.0,
        "total_gen_tokens": 0,
        "total_full_tokens": 0,
    }

    for m in metrics_list:
        total["json_files"] += m["json_files"]
        total["records"] += m["records"]
        total["total_time"] += m["total_time"]
        total["total_gen_tokens"] += m["total_gen_tokens"]
        total["total_full_tokens"] += m["total_full_tokens"]
        total["speed_sum"] += m["speed_sum"]
        total["speed_count"] += m["speed_count"]

    total["avg_speed"] = (
        total["speed_sum"] / total["speed_count"] if total["speed_count"] > 0 else 0.0
    )
    total["weighted_speed_gen"] = (
        total["total_gen_tokens"] / total["total_time"] if total["total_time"] > 0 else 0.0
    )
    total["weighted_speed_full"] = (
        total["total_full_tokens"] / total["total_time"] if total["total_time"] > 0 else 0.0
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze AIME JSON results: total time and token speed."
    )
    parser.add_argument(
        "path_or_prefix",
        help="Directory path or prefix (e.g. AIME_20260206_171838_).",
    )
    args = parser.parse_args()

    base_dir = Path.cwd()
    path_or_prefix = args.path_or_prefix
    target_dir = Path(path_or_prefix)

    if target_dir.exists() and target_dir.is_dir():
        metrics = analyze_directory(target_dir)
        print_metrics(f"Directory: {target_dir}", metrics)
        return

    directories = find_directories_with_prefix(path_or_prefix, base_dir)
    if not directories:
        print(f"No directories found with prefix '{path_or_prefix}' in {base_dir}")
        return

    per_dir_metrics: List[Dict[str, float]] = []
    for directory in directories:
        metrics = analyze_directory(directory)
        per_dir_metrics.append(metrics)
        print_metrics(f"Directory: {directory}", metrics)
        print()

    if len(per_dir_metrics) > 1:
        overall = merge_metrics(per_dir_metrics)
        print_metrics("Overall:", overall)


if __name__ == "__main__":
    main()
