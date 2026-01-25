import argparse
import sys
from openai import OpenAI


def check_completion(base_url, model, label, prompt, max_tokens=32):
    client = OpenAI(base_url=base_url, api_key="None", timeout=60)
    try:
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.95,
        )
        text = resp.choices[0].text
        print(f"[OK] {label}: {model} -> {text.strip()[:200]}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {model} -> {exc}")
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_url", default="http://127.0.0.1:12347/v1")
    parser.add_argument("--draft_url", default="http://127.0.0.1:12345/v1")
    parser.add_argument("--judge_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--target_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--draft_model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    return parser.parse_args()


def main():
    args = parse_args()
    prompt = "Say hello in one short sentence."
    ok = True
    ok &= check_completion(args.target_url, args.target_model, "target", prompt)
    ok &= check_completion(args.draft_url, args.draft_model, "draft", prompt)
    ok &= check_completion(args.judge_url, args.judge_model, "judge", prompt)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
