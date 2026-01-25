#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def http_json(url, method="GET", payload=None, timeout=10):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, raw
    except urllib.error.URLError as e:
        return None, str(e)


def normalize_base_url(base_url):
    return base_url.rstrip("/")


def fetch_models(base_url, timeout):
    status, raw = http_json(f"{base_url}/models", timeout=timeout)
    if status != 200:
        return None, status, raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, status, raw
    data = payload.get("data", [])
    models = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            models.append(item["id"])
    return models, status, raw


def percentile(values, p):
    if not values:
        return None
    values_sorted = sorted(values)
    idx = int((p / 100.0) * (len(values_sorted) - 1))
    return values_sorted[idx]


def run_one(base_url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    status, raw = http_json(f"{base_url}/completions", method="POST", payload=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0
    if status != 200:
        return False, elapsed, status, raw
    try:
        payload = json.loads(raw)
        choices = payload.get("choices", [])
        ok = bool(choices)
    except json.JSONDecodeError:
        ok = False
    return ok, elapsed, status, None if ok else raw


def parse_args():
    parser = argparse.ArgumentParser(description="Load test vLLM server on port 8000.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--base-url", default=None, help="Override base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None, help="Model id to use for completions")
    parser.add_argument("--prompt", default="Say OK.", help="Prompt for test completion")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max tokens for each completion")
    parser.add_argument("--total", type=int, default=100, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.base_url:
        base_url = normalize_base_url(args.base_url)
    else:
        base_url = normalize_base_url(f"http://{args.host}:{args.port}/v1")

    model = args.model
    if not model:
        models, status, raw = fetch_models(base_url, args.timeout)
        if not models:
            print("Model list request failed.")
            print(f"Base URL: {base_url}")
            print(f"HTTP status: {status}")
            print(f"Response: {raw}")
            return 1
        model = models[0]

    total = max(1, args.total)
    concurrency = max(1, min(args.concurrency, total))

    latencies = []
    failures = 0
    error_counts = {}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_one,
                base_url,
                model,
                args.prompt,
                args.max_tokens,
                args.timeout,
            )
            for _ in range(total)
        ]
        for future in as_completed(futures):
            ok, latency, status, error = future.result()
            if ok:
                latencies.append(latency)
            else:
                failures += 1
                key = str(status)
                error_counts[key] = error_counts.get(key, 0) + 1

    wall = time.perf_counter() - t_start
    success = total - failures

    print("Load test complete.")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Total requests: {total}")
    print(f"Concurrency: {concurrency}")
    print(f"Success: {success}")
    print(f"Failures: {failures}")
    print(f"Wall time: {wall:.3f}s")
    if wall > 0:
        print(f"Throughput: {total / wall:.2f} req/s")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Latency avg: {avg_latency:.3f}s")
        print(f"Latency p50: {percentile(latencies, 50):.3f}s")
        print(f"Latency p90: {percentile(latencies, 90):.3f}s")
        print(f"Latency p99: {percentile(latencies, 99):.3f}s")
        print(f"Latency max: {max(latencies):.3f}s")

    if error_counts:
        print("Error counts by status:")
        for key in sorted(error_counts.keys()):
            print(f"  {key}: {error_counts[key]}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
