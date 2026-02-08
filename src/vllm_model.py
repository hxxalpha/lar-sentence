from vllm.inputs import TokensPrompt
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import RequestOutputKind
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio
import time
import inspect
from openai import AsyncOpenAI
from transformers import AutoTokenizer

DEFAULT_STOP_SEQUENCES = [". ", "? ", "...", "\n"]

class LLMModel:
    def __init__(self, model, eos_id=None, gpu_ids="", enable_n_gram=False,  vllm_config={}):
        self.gpu_ids = gpu_ids
        self.eos_id = eos_id
        dtype = vllm_config.get("dtype", "float16")
        quantization = vllm_config.get("quantization")
        if enable_n_gram:
            print('Ngram Setting Up Model: ', [model, gpu_ids])
            engine_args = AsyncEngineArgs(
                model=model,
                dtype=dtype,
                quantization=quantization,
                enforce_eager=vllm_config['force_eager'],
                tensor_parallel_size=len(gpu_ids.split(',')),
                data_parallel_size=1,
                gpu_memory_utilization=0.7,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
                speculative_config={
                    "method": "ngram",
                    "num_speculative_tokens": vllm_config['num_speculative_tokens'],
                    "prompt_lookup_max": vllm_config['prompt_lookup_max'],
                },
            )
        else:
            print('Setting Up Model: ', [model, gpu_ids])

            engine_args = AsyncEngineArgs(
                model=model,
                dtype=dtype,
                quantization=quantization,
                enforce_eager=vllm_config['force_eager'],
                tensor_parallel_size=len(gpu_ids.split(',')),
                data_parallel_size=1,
                gpu_memory_utilization=0.7,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
            )
    
        self.prefix = model.split('/')[-1].split('-')[0]  # Extract prefix from model name
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.request_id = 1
        self.enable_n_gram = enable_n_gram
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    async def generate(self,
                   prompt: PromptType, max_tokens: int, temperature, top_p, top_k, stop) -> tuple[int, str]:
        # Ensure generate doesn't complete too fast for cancellation test.
        #await asyncio.sleep(0.00001)
        prompt = TokensPrompt(prompt_token_ids=prompt)

        sampling_params = SamplingParams(max_tokens=max_tokens,
                        ignore_eos=False,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        top_k=top_k, seed=int(time.time()))
        self.request_id += 1
        async for out in self.engine.generate(request_id=self.prefix+f"request-{self.request_id}",
                                        prompt=prompt,
                                        sampling_params=sampling_params):


            await asyncio.sleep(0.)
        
        #print('Output: ', out.outputs[0])
        out.outputs[0].token_ids = list(out.outputs[0].token_ids)
        num_tokens = len(out.outputs[0].token_ids)

        if out.outputs[0].finish_reason == 'stop':
            if out.outputs[0].stop_reason is not None:
                if out.outputs[0].stop_reason in self.eos_id:
                    return out.outputs[0].text + '<|endoftext|>', out.outputs[0].finish_reason, out.outputs[0].stop_reason, num_tokens, out.outputs[0].token_ids
                completion_with_stop = out.outputs[0].text + out.outputs[0].stop_reason
                # Some backends strip the matched stop from token_ids; re-tokenize
                # the returned text so num_tokens/prefix growth stays consistent.
                token_ids_with_stop = self.tokenizer.encode(completion_with_stop, add_special_tokens=False)
                return completion_with_stop, out.outputs[0].finish_reason, out.outputs[0].stop_reason, len(token_ids_with_stop), token_ids_with_stop
            else:
                print('Finishing: ', out.outputs[0].finish_reason, out.outputs[0].stop_reason, self.eos_id)
                return out.outputs[0].text, out.outputs[0].finish_reason, out.outputs[0].stop_reason, num_tokens, out.outputs[0].token_ids
        else:
            return out.outputs[0].text, out.outputs[0].finish_reason, out.outputs[0].stop_reason, num_tokens, out.outputs[0].token_ids


class RemoteLLMModel:
    def __init__(self, model, tokenizer, base_url, eos_id=None, enable_n_gram=False, timeout=None, api_key="None"):
        self.model = model
        self.tokenizer = tokenizer
        self.base_url = base_url
        self.eos_id = eos_id or []
        self.enable_n_gram = enable_n_gram
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _prompt_to_text(self, prompt):
        if isinstance(prompt, list):
            return self.tokenizer.decode(prompt, skip_special_tokens=False)
        return prompt

    async def generate(self,
                   prompt: PromptType, max_tokens: int, temperature, top_p, top_k, stop) -> tuple[int, str]:
        prompt_text = self._prompt_to_text(prompt)
        stop_param = stop if stop else None
        request_params = {
            "model": self.model,
            "prompt": prompt_text,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop_param,
            "top_p": top_p,
        }
        if top_k is not None:
            create_sig = inspect.signature(self.client.completions.create)
            if "extra_body" in create_sig.parameters:
                request_params["extra_body"] = {"top_k": top_k}
            else:
                request_params["top_k"] = top_k
        response = await self.client.completions.create(**request_params)
        choice = response.choices[0]
        completion_text = choice.text or ""
        finish_reason = choice.finish_reason
        stop_reason = None
        if hasattr(choice, "stop_reason"):
            stop_reason = choice.stop_reason
        elif hasattr(choice, "matched_stop"):
            stop_reason = choice.matched_stop

        token_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
        num_tokens = len(token_ids)

        if finish_reason == 'stop' and stop_reason is not None:
            if stop_reason in self.eos_id:
                return completion_text + '<|endoftext|>', finish_reason, stop_reason, num_tokens, token_ids
            completion_with_stop = completion_text + stop_reason
            token_ids_with_stop = self.tokenizer.encode(completion_with_stop, add_special_tokens=False)
            return completion_with_stop, finish_reason, stop_reason, len(token_ids_with_stop), token_ids_with_stop

        return completion_text, finish_reason, stop_reason, num_tokens, token_ids

### Drafter model
class Drafter:
    def __init__(self, model, eos_id=None, draft_gpu_id=None, \
                 enable_n_gram=False, vllm_config={'force_eager': False, 'num_speculative_tokens': 4, 'prompt_lookup_max': 2, 'dtype': 'float16'}):
        print('Drafting')
        self.model = LLMModel(model, eos_id, draft_gpu_id, enable_n_gram, vllm_config)
        
    def draft(self, prompt, max_tokens=100, temperature=0.6, top_p=0.95, top_k=20, stop=None):
        if stop is None:
            stop = DEFAULT_STOP_SEQUENCES
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop)


class RemoteDrafter:
    def __init__(self, model, tokenizer, base_url, eos_id=None, \
                 enable_n_gram=False, timeout=None):
        print('Drafting (remote)')
        self.model = RemoteLLMModel(model, tokenizer, base_url, eos_id=eos_id, enable_n_gram=enable_n_gram, timeout=timeout)

    def draft(self, prompt, max_tokens=100, temperature=0.6, top_p=0.95, top_k=20, stop=None):
        if stop is None:
            stop = DEFAULT_STOP_SEQUENCES
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop)

### Target Model 
class Targeter:
    def __init__(self, model, eos_id=None, target_gpu_id=None, \
                 enable_n_gram=False, vllm_config={'force_eager': False, 'num_speculative_tokens': 4, 'prompt_lookup_max': 2, 'dtype': 'float16'}):
        print('Targeting')
        self.model = LLMModel(model, eos_id, target_gpu_id, enable_n_gram, vllm_config)
        
    def target(self, prompt, max_tokens=100, temperature=0.6, top_p=0.95, top_k=20, stop=None):
        if stop is None:
            stop = DEFAULT_STOP_SEQUENCES
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop)


class RemoteTargeter:
    def __init__(self, model, tokenizer, base_url, eos_id=None, \
                 enable_n_gram=False, timeout=None):
        print('Targeting (remote)')
        self.model = RemoteLLMModel(model, tokenizer, base_url, eos_id=eos_id, enable_n_gram=enable_n_gram, timeout=timeout)
        
    def target(self, prompt, max_tokens=100, temperature=0.6, top_p=0.95, top_k=20, stop=None):
        if stop is None:
            stop = DEFAULT_STOP_SEQUENCES
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop)
