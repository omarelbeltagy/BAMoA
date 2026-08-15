# Shared Mixture-of-Agents core for BA-MoA experiments.
# All dataset-specific runners (bbq_runner.py, winobias_runner.py, ...)
# import from here rather than redefining models/layers/prompt logic.
# This guarantees RQ1 cross-dataset comparability: the MoA system itself
# is held fixed while only the benchmark/prompt changes.
import asyncio
import os
from together import AsyncTogether, Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# Fixed model pool for all experiments — verified serverless-accessible and
# non-reasoning (see project docs for verification method/results).
REFERENCE_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "google/gemma-4-31B-it",
]
AGGREGATOR_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
LAYERS = 4
PROPOSER_MAX_TOKENS = 512
AGGREGATOR_MAX_TOKENS = 1024
TEMPERATURE = 0.7


def get_system_prompt_with_references(prev_responses):
    return (
        AGGREGATOR_SYSTEM_PROMPT
        + "\n"
        + "\n".join([f"{i+1}. {str(r)}" for i, r in enumerate(prev_responses)])
    )


async def run_llm(model, user_prompt, prev_response=None):
    for sleep_time in [1, 2, 4]:
        try:
            messages = (
                [
                    {"role": "system", "content": get_system_prompt_with_references(prev_response)},
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=PROPOSER_MAX_TOKENS,
            )
            content = response.choices[0].message.content
            if not content:
                await asyncio.sleep(sleep_time)
                continue
            return content
        except Exception as e:
            print(f"  Error [{model.split('/')[-1]}]: {e}")
            await asyncio.sleep(sleep_time)
    return None


async def run_moa(user_prompt):
    """
    Run the full MoA pipeline on a single prompt and return per-layer responses.
    Dataset-agnostic — the same function is used by every runner. Only the
    prompt formatting and scoring differ per dataset.
    """
    run_log = {"question": user_prompt, "layers": {}}

    results = await asyncio.gather(*[run_llm(model, user_prompt) for model in REFERENCE_MODELS])
    run_log["layers"]["layer_1"] = dict(zip(REFERENCE_MODELS, results))

    for layer_idx in range(1, LAYERS - 1):
        results = await asyncio.gather(
            *[run_llm(model, user_prompt, prev_response=results) for model in REFERENCE_MODELS]
        )
        run_log["layers"][f"layer_{layer_idx + 1}"] = dict(zip(REFERENCE_MODELS, results))

    final_response = ""
    final_stream = client.chat.completions.create(
        model=AGGREGATOR_MODEL,
        messages=[
            {"role": "system", "content": get_system_prompt_with_references(results)},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        max_tokens=AGGREGATOR_MAX_TOKENS,
    )
    for chunk in final_stream:
        if chunk.choices:
            final_response += chunk.choices[0].delta.content or ""

    run_log["final_response"] = final_response
    return run_log