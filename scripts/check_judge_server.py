#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


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


def run_completion(base_url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    return http_json(f"{base_url}/completions", method="POST", payload=payload, timeout=timeout)


def parse_args():
    parser = argparse.ArgumentParser(description="Check vLLM server on port 8000.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--base-url", default=None, help="Override base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None, help="Model id to use for test completion")
    parser.add_argument("--prompt", default="Say OK.", help="Prompt for test completion")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max tokens for test completion")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.base_url:
        base_url = normalize_base_url(args.base_url)
    else:
        base_url = normalize_base_url(f"http://{args.host}:{args.port}/v1")

    t0 = time.perf_counter()
    models, status, raw = fetch_models(base_url, args.timeout)
    if not models:
        print("Model list request failed.")
        print(f"Base URL: {base_url}")
        print(f"HTTP status: {status}")
        print(f"Response: {raw}")
        return 1

    model = args.model or models[0]
    status, raw = run_completion(base_url, model, args.prompt, args.max_tokens, args.timeout)
    elapsed = time.perf_counter() - t0
    if status != 200:
        print("Completion request failed.")
        print(f"Base URL: {base_url}")
        print(f"Model: {model}")
        print(f"HTTP status: {status}")
        print(f"Response: {raw}")
        return 1

    try:
        payload = json.loads(raw)
        text = payload.get("choices", [{}])[0].get("text", "").strip()
        finish_reason = payload.get("choices", [{}])[0].get("finish_reason")
    except json.JSONDecodeError:
        text = ""
        finish_reason = None

    print("Server check OK.")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Finish reason: {finish_reason}")
    print(f"Elapsed: {elapsed:.3f}s")
    if text:
        print(f"Sample output: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
