import json
import asyncio
import os
import time
import traceback
from typing import List, Dict, Any
from lr_tree import TreeNode

DEFAULT_STOP_SEQUENCES = [". ", "? ", "...", "\n"]

equal_prompt = '''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nEvaluate whether the following two reasoning steps (s1 and s2) convey exactly the same meaning. Focus on semantic similarity rather than exact wording. 

Compare the main ideas, key points, overall message, logical structure, and numerical calculations/results of both reasoning steps.

If the reasoning steps convey essentially the same meaning and generate same calculation results, respond with [aligned].
If the reasoning steps express different meanings, respond with [unaligned]. If it is too hard to determine, respond with [unaligned]

Please directly provide the final result in [aligned] or [unaligned].

Reasoning step 1 (s1):
<start_s1>
{}
<end_s1>

Reasoning step 2 (s2):
<start_s2>
{}
<end_s2><|im_end|>\n<|im_start|>assistant\n['''


async def get_model_response(prompt, model="gpt-4-turbo-preview", temperature=0.0, max_tokens=1000, stop=None, client=None):
    """Get response from OpenAI API"""
    start_time = time.time()
    if client is None:
        print("Error getting response: client is None")
        return None
    try:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95
        )
        return response.choices[0].text
    except Exception as e:
        elapsed = time.time() - start_time
        base_url = getattr(client, "base_url", None)
        if base_url:
            print(f"Judge base_url: {base_url}")
        print(
            "Completion request failed:",
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
                "elapsed_s": round(elapsed, 3),
                "error_type": type(e).__name__,
            },
        )
        status_code = getattr(e, "status_code", None)
        if status_code is not None:
            print(f"HTTP status: {status_code}")
        response_obj = getattr(e, "response", None)
        if response_obj is not None:
            try:
                print(f"Response text: {response_obj.text}")
            except Exception:
                print("Response text: <unavailable>")
        print(traceback.format_exc())
        return None

async def text_accept(response1, response2, last_ending, judge_client, judge_model):
    if not last_ending:
        return {
            'accepted': False,
            'judge_response': None,
            'reason': 'last_ending_false',
        }

    # Check if the responses are exactly the same
    if response1.replace(" ", "").replace("\n", "").lower() == response2.replace(" ", "").replace("\n", "").lower(): 
        return {
            'accepted': True,
            'judge_response': '[exact_match]',
            'reason': 'exact_match',
        }

    prompt = equal_prompt.format(response1.strip(), response2.strip())

    response = await get_model_response(prompt, client=judge_client, model=judge_model, max_tokens=1)

    if response is None:
        return {
            'accepted': False,
            'judge_response': None,
            'reason': 'judge_response_none',
        }

    return {
        'accepted': ('ali' in response.lower() and 'un' not in response.lower()),
        'judge_response': response,
        'reason': 'judge_model',
    }

async def accept_func(response1, response2, last_ending, judge_client, judge_model):
    return await text_accept(response1, response2, last_ending, judge_client, judge_model)

def token_transform(token_ids, tokenizer_source, tokenizer_target):
    """Transform token IDs from source tokenizer to target tokenizer"""
    text = tokenizer_source.decode(token_ids)
    return tokenizer_target.encode(text, add_special_tokens=False)


def resolve_step_stop_sequences(config):
    stop_sequences = config.get('stop')
    if not stop_sequences or stop_sequences == ['\n\n']:
        return list(DEFAULT_STOP_SEQUENCES)
    return stop_sequences


async def run_problem(question, i, target_model, draft_model, \
                      target_tokenizer, draft_tokenizer, judge_client, \
                        target_config, draft_config, output_dir, \
                        use_spec, width, max_depth, ignore_half_sentence):
    """Run a single problem with the target and draft models"""
    # Use per-step sentence stopping by default in speculative mode.
    # Preserve non-default custom stop strings from config.
    target_config = dict(target_config)
    draft_config = dict(draft_config)
    target_config['stop'] = resolve_step_stop_sequences(target_config)
    draft_config['stop'] = resolve_step_stop_sequences(draft_config)

    question_id = i
    question_text = question['question']
    gold_answer = question['answer']

    
    # Prepare the prompt
    prompt = question_text + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"

    target_prompt = target_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        )

    draft_prompt = draft_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        )


    target_token_ids = target_tokenizer.encode(target_prompt, add_special_tokens=False)
    draft_token_ids = draft_tokenizer.encode(draft_prompt, add_special_tokens=False)
    # response = await target_model.target(prompt_token_ids)

    if target_config['name'] == draft_config['name']:
        target2draft = lambda x: x 
        draft2target = lambda x: x
    else:
        target2draft = lambda x: token_transform(x, target_tokenizer, draft_tokenizer)
        draft2target = lambda x: token_transform(x, draft_tokenizer, target_tokenizer)

    print('Running question:', question_id, 
          'Draft prompt:', [draft_prompt], 'Target prompt:', [target_prompt],
          'Tokens: ', [target_token_ids, draft_token_ids])

    t0 = time.time()
    draft_generation_logs = []
    target_generation_logs = []
    comparison_logs = []
    accepts = []
    accepted_count = 0
    judge_elapsed_total = 0.0
    termination_reason = None

    seen_target_node_ids = set()
    seen_draft_node_ids = set()
    comparison_id = 0

    def node_log_id(node):
        return f"node_{id(node)}"

    def normalize_text(text):
        return text.replace(" ", "").replace("\n", "").lower()

    if use_spec:
        root = TreeNode(prefix=target_prompt, prefix_token_ids=target_token_ids, draft_prefix_token_ids=draft_token_ids, \
                       # draft_prefix=draft_prompt, draft_prefix_token_ids=draft_token_ids, \
                        width=width, idx=0, depth=1, drafter=draft_model, \
                            targeter=target_model, empty=True, max_depth=max_depth, \
                            target_config=target_config, draft_config=draft_config,\
                                qid=question_id, ignore_half_sentence=ignore_half_sentence, \
                                    accept_func=accept_func,judge_client=judge_client,draft2target=draft2target, target2draft=target2draft, judge_model=target_config['judge_model'])

        root_node = root

        while True:
            await root.traverse(root_node)
            await root.traverse_collect_main()
            await root.travel_set_accepted()
            found_eos = False
            while root.main_data is not None and root.check_judge_children():
                print('Main data: ', root.depth, root.idx)
                root_id = node_log_id(root)
                if root_id not in seen_target_node_ids:
                    seen_target_node_ids.add(root_id)
                    target_generation_logs.append({
                        'node_id': root_id,
                        'depth': root.depth,
                        'idx': root.idx,
                        'sentence': root.main_data['t'],
                        'num_tokens': root.main_data['n'],
                        'elapsed_s': root.main_data.get('elapsed_s'),
                        'finish_reason': root.main_data['r'],
                        'stop_reason': root.main_data['s'],
                        'last_ending': root.main_data['l'],
                    })

                if any(eos_id in root.main_data['i'] for eos_id in target_config['eos_id']) or \
                   any(eos_id in root.drafter_data['i'] for eos_id in draft_config['eos_id']):
                    print('Found EOS in main data:', root.depth, root.idx, root.main_data, root.drafter_data)
                    termination_reason = 'eos_found'
                    found_eos = True
                    break

                total_tokens = root.generated_tokens + root.main_data['n']

                if total_tokens > target_config['max_tokens']:
                    termination_reason = 'max_tokens_exceeded'
                    found_eos = True
                    break
                
                if root.main_data['n'] == 0 and root.drafter_data['n'] == 0:
                    print('No progress (0 tokens from both), stopping at:', root.depth, root.idx)
                    termination_reason = 'no_progress'
                    found_eos = True
                    break

                children = root.children
                accepted_id = -1
                for cid, child in enumerate(children):
                    if child is None:
                        continue
                    if child.drafter_data is not None:
                        child_id = node_log_id(child)
                        if child_id not in seen_draft_node_ids:
                            seen_draft_node_ids.add(child_id)
                            draft_generation_logs.append({
                                'node_id': child_id,
                                'depth': child.depth,
                                'idx': child.idx,
                                'sentence': child.drafter_data['t'],
                                'num_tokens': child.drafter_data['n'],
                                'elapsed_s': child.drafter_data.get('elapsed_s'),
                                'finish_reason': child.drafter_data['r'],
                                'stop_reason': child.drafter_data['s'],
                            })

                        judge_result = child.drafter_data['j'].result()
                        if isinstance(judge_result, dict):
                            accepted = bool(judge_result.get('accepted', False))
                            judge_response = judge_result.get('judge_response')
                            judge_reason = judge_result.get('reason')
                            judge_elapsed = judge_result.get('elapsed_s')
                        else:
                            accepted = bool(judge_result)
                            judge_response = str(judge_result)
                            judge_reason = 'legacy_bool'
                            judge_elapsed = None

                        normalized_equal = normalize_text(root.main_data['t']) == normalize_text(child.drafter_data['t'])
                        comparison_logs.append({
                            'compare_id': comparison_id,
                            'root_node_id': root_id,
                            'child_node_id': child_id,
                            'depth': root.depth,
                            'target_sentence': root.main_data['t'],
                            'target_num_tokens': root.main_data['n'],
                            'target_elapsed_s': root.main_data.get('elapsed_s'),
                            'draft_sentence': child.drafter_data['t'],
                            'draft_num_tokens': child.drafter_data['n'],
                            'draft_elapsed_s': child.drafter_data.get('elapsed_s'),
                            'normalized_equal': normalized_equal,
                            'judge_response': judge_response,
                            'judge_reason': judge_reason,
                            'judge_elapsed_s': judge_elapsed,
                            'accepted': accepted,
                        })
                        comparison_id += 1
                        accepts.append(1 if accepted else 0)
                        if accepted:
                            accepted_count += 1
                        if isinstance(judge_elapsed, (int, float)):
                            judge_elapsed_total += float(judge_elapsed)

                        if accepted:
                            if accepted_id == -1:
                                print('Accepted child...', child.depth, child.idx, [root.main_data['t'], child.drafter_data['t']])
                                accepted_id = cid

                if accepted_id != -1:
                    for cid, child in enumerate(children):
                        if cid == accepted_id:
                            continue
                        if child is not None:
                            child.cancel()
                            root.canceled.append(child)
                            children[cid] = None
                            del child
                    
                    root = children[accepted_id]
                    root.accepted = True
                    root_node = root
                    print('Processing ', root.depth, root.idx, root.main_data, [root.drafter_data, root.main_data])
                    termination_reason = 'accepted_child'
                    
                else:
                    for cid, child in enumerate(children):
                        if child is not None:
                            child.cancel()
                            root.canceled.append(child)
                            children[cid] = None
                            del child

                    old_root = root
                    root = TreeNode(prefix=root.prefix + root.drafter_data['t'] + root.main_data['t'], \
                                    prefix_token_ids=root.prefix_token_ids + root.drafter_data['ti'] + root.main_data['i'], 
                                    draft_prefix_token_ids=root.draft_prefix_token_ids + root.drafter_data['i'] + root.main_data['di'],\
                                        width=root.width, idx=root.idx, depth=root.depth + 1, \
                                            drafter=root.drafter, targeter=root.targeter, empty=True, max_depth=root.max_depth, \
                                                generated_tokens=root.generated_tokens + root.drafter_data['n'] + root.main_data['n'], \
                                                target_config=target_config, draft_config=draft_config,\
                                                    qid=question_id, ignore_half_sentence=ignore_half_sentence, \
                                                        accept_func=accept_func, judge_client=judge_client, draft2target=draft2target, target2draft=target2draft, judge_model=target_config['judge_model'])
                    old_root.children.append(root)
                    root_node = root

                    print('Failed to find a child, creating a new root...', root.depth, root.idx, root.main_data)
                    termination_reason = 'fallback_new_root'
                    break
            await asyncio.sleep(0.0)
            #times_step.append((t3 - t2, t2 - t1, t1 - t0))
            if found_eos:
                break

        #target_response = root.main.prefix + root.main_data['t'] if root.main_data is not None else root.prefix

        full_tokens = root.main.prefix_token_ids + root.main_data['i'] #- len(tokenizer.encode(prompt))
        full_text = target_tokenizer.decode(full_tokens, skip_special_tokens=False)
        generation_tokens = full_tokens[len(target_token_ids):]
        
        
    else:
        # Run the target model
        target_start_time = time.time()
        response = await target_model.target(target_token_ids, max_tokens=target_config['max_tokens'],
                                              temperature=target_config['temperature'],
                                              top_p=target_config['top_p'], top_k=target_config['top_k'],
                                              stop=[])
        target_elapsed = time.time() - target_start_time
        generation_text = response[0]
        full_text = target_prompt + generation_text
        generation_tokens = response[-1]
        full_tokens = target_token_ids + generation_tokens
        target_generation_logs.append({
            'node_id': 'direct_target',
            'depth': 1,
            'idx': 0,
            'sentence': generation_text,
            'num_tokens': len(generation_tokens),
            'elapsed_s': target_elapsed,
            'finish_reason': response[1] if len(response) > 1 else None,
            'stop_reason': response[2] if len(response) > 2 else None,
            'last_ending': True,
        })
        termination_reason = 'direct_target'

    t1 = time.time()

    elapsed = t1 - t0
    speed = (len(generation_tokens) / elapsed) if elapsed > 0 else 0.0
    accept_rate = (accepted_count / len(accepts)) if len(accepts) > 0 else 0.0
    draft_elapsed_total = sum(item.get('elapsed_s', 0.0) or 0.0 for item in draft_generation_logs)
    target_elapsed_total = sum(item.get('elapsed_s', 0.0) or 0.0 for item in target_generation_logs)

    print('Finished: ', len(generation_tokens), elapsed, 'Speed: ', speed , 'tokens/s')

    # Save the results
    result = {
        'question_id': question_id,
        'question': question_text,
        'draft_prompt': draft_prompt,
        'target_prompt': target_prompt,
        'generation_tokens': generation_tokens,
        'generation_text': target_tokenizer.decode(generation_tokens, skip_special_tokens=False),
        'full_text': full_text,
        'full_tokens': full_tokens,
        'gold': gold_answer,
        'time_taken': elapsed,
        'speed': speed,
        'draft_config': draft_config,
        'target_config': target_config,
        'accepts': accepts,
        'accept_stats': {
            'accepted_count': accepted_count,
            'judge_count': len(accepts),
            'accept_rate': accept_rate,
        },
        'termination_reason': termination_reason,
        'draft_generation_logs': draft_generation_logs,
        'target_generation_logs': target_generation_logs,
        'comparison_logs': comparison_logs,
        'timing_stats': {
            'draft_total_elapsed_s': draft_elapsed_total,
            'target_total_elapsed_s': target_elapsed_total,
            'judge_total_elapsed_s': judge_elapsed_total,
        },
    }

    output_file = os.path.join(output_dir, f"{question_id}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)
