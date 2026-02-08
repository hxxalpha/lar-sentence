import openai
import asyncio
from transformers import AutoTokenizer
import os
import datetime
import argparse
import random
import json

from vllm_model import Targeter, Drafter, RemoteTargeter, RemoteDrafter
from lr import run_problem

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='aime-2024.jsonl')
parser.add_argument('--start_qid', type=int, default=None)
parser.add_argument('--end_qid', type=int, default=None)
parser.add_argument('--prefix', type=str, default='AIME')

parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--width', type=int, default=1)

parser.add_argument('--model', type=str, default='Qwen/Qwen3-14B')
parser.add_argument('--draft_model', type=str, default='Qwen/Qwen3-1.7B')
parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--judge_port', type=int, default=8000)
parser.add_argument('--target_port', type=int, default=12347)
parser.add_argument('--draft_port', type=int, default=12345)
parser.add_argument('--target_host', type=str, default='127.0.0.1')
parser.add_argument('--draft_host', type=str, default='127.0.0.1')
parser.add_argument('--use_local_vllm', action='store_true')

parser.add_argument('--target_gpu_id', type=str, default='0,1')
parser.add_argument('--draft_gpu_id', type=str, default='2')

parser.add_argument('--enable_n_gram', action='store_true')
parser.add_argument('--num_speculative_tokens', type=int, default=6)
parser.add_argument('--prompt_lookup_max', type=int, default=2)

parser.add_argument('--max_tokens_len', type=int, default=37000)
parser.add_argument('--use_spec', action='store_true')
parser.add_argument('--target_dtype', type=str, default=None)
parser.add_argument('--draft_dtype', type=str, default=None)
parser.add_argument('--target_quantization', type=str, default=None)
parser.add_argument('--draft_quantization', type=str, default=None)

args = parser.parse_args()

judge_client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{args.judge_port}/v1", api_key="None", timeout=None)

DEFAULT_STOP_SEQUENCES = [". ", "? ", "...", "\n"]

MODEL_CONFIGS = {
    'deepseek': {
        'name': 'deepseek',
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 0,
        'max_tokens': 32768,
        'prompt_template': 'deepseek',
        'eos_id': [151643, 151645],
        'stop': list(DEFAULT_STOP_SEQUENCES),
        'step_tokens': 100,
    },
    'qwen3': {
        'name': 'qwen3',
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 20,
        'max_tokens': 38912,
        'prompt_template': 'qwen3',
        'eos_id': [151643, 151645],
        'stop': list(DEFAULT_STOP_SEQUENCES),
        'step_tokens': 100,
    }
}


def get_model_config(model_name):
    """Get configuration for a specific model."""
    model_name_lower = model_name.lower()
    
    # Match model configurations
    if 'deepseek-r1' in model_name_lower:
        return MODEL_CONFIGS['deepseek']
    elif 'qwen3' in model_name_lower:
        return MODEL_CONFIGS['qwen3']
    else:
        assert False, f"Unknown model: {model_name}"

def load_questions(file_path):
    """Load questions from jsonl file"""
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)
    return questions


def resolve_quantization(model_name, override):
    if override:
        return override
    model_name_lower = model_name.lower()
    if "awq" in model_name_lower:
        return "awq"
    if "gptq" in model_name_lower:
        return "gptq"
    if "int8" in model_name.lower():
        return "bitsandbytes"
    return None


def resolve_dtype(model_name, override, default_dtype):
    if override:
        return override
    if "fp8" in model_name.lower():
        return "auto"
    return default_dtype


async def main():

    output_dir = args.prefix + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + str(random.randint(100000, 999999))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    questions = load_questions(args.dataset)[args.start_qid:args.end_qid]

    target_tokenizer = AutoTokenizer.from_pretrained(args.model)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)

    target_config = get_model_config(args.model)
    draft_config = get_model_config(args.draft_model)

    if args.use_local_vllm:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.target_gpu_id
        # Set environment variables for GPU usage
        target_quantization = resolve_quantization(args.model, args.target_quantization)
        draft_quantization = resolve_quantization(args.draft_model, args.draft_quantization)
        target_dtype = resolve_dtype(args.model, args.target_dtype, "float16")
        draft_dtype = resolve_dtype(args.draft_model, args.draft_dtype, "float16")

        target_model = Targeter(args.model, eos_id=target_config['eos_id'], target_gpu_id=args.target_gpu_id,
                        enable_n_gram=args.enable_n_gram, vllm_config={'force_eager': False, 'num_speculative_tokens': args.num_speculative_tokens, 'prompt_lookup_max': args.prompt_lookup_max, 'dtype': target_dtype, 'quantization': target_quantization}) 

        os.environ['CUDA_VISIBLE_DEVICES'] = args.draft_gpu_id
        draft_model = Drafter(args.draft_model, eos_id=draft_config['eos_id'], draft_gpu_id=args.draft_gpu_id,
                        enable_n_gram=args.enable_n_gram, vllm_config={'force_eager': False, 'num_speculative_tokens': args.num_speculative_tokens, 'prompt_lookup_max': args.prompt_lookup_max, 'dtype': draft_dtype, 'quantization': draft_quantization})
    else:
        target_base_url = f"http://{args.target_host}:{args.target_port}/v1"
        draft_base_url = f"http://{args.draft_host}:{args.draft_port}/v1"
        target_model = RemoteTargeter(args.model, tokenizer=target_tokenizer, base_url=target_base_url,
                                      eos_id=target_config['eos_id'], enable_n_gram=args.enable_n_gram)
        draft_model = RemoteDrafter(args.draft_model, tokenizer=draft_tokenizer, base_url=draft_base_url,
                                    eos_id=draft_config['eos_id'], enable_n_gram=args.enable_n_gram)


    assert target_config['name'] == draft_config['name'], \
        "Target and draft models must be of the same type (e.g., both Qwen3)."

    target_config['judge_model'] = args.judge_model
    print(f"Target Model Config: {target_config}")
    print(f"Draft Model Config: {draft_config}")

    for i in range(len(questions)):
        await run_problem(questions[i], i, target_model, draft_model, \
                          target_tokenizer, draft_tokenizer, judge_client, \
                          target_config, draft_config, output_dir, \
                          use_spec=args.use_spec, width=args.width, max_depth=args.max_depth, \
                          ignore_half_sentence=True)

    print(f"Results saved to {output_dir}")


if __name__ == '__main__':

    asyncio.run(main())
