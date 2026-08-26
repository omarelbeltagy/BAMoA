# Shared Mixture-of-Agents core for BA-MoA experiments.
# All dataset-specific runners (bbq_runner.py, winobias_runner.py, ...)
# import from here rather than redefining models/layers/prompt logic.
# This guarantees RQ1 cross-dataset comparability: the MoA system itself
# is held fixed while only the benchmark/prompt changes.
import asyncio
import os
import time
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

# Deterministic by default. Variance is introduced deliberately via seeds at
# the experiment level, never as an uncontrolled property of the pipeline.
TEMPERATURE = 0.0


def get_system_prompt_with_references(prev_responses):
    return (
        AGGREGATOR_SYSTEM_PROMPT
        + "\n"
        + "\n".join([f"{i+1}. {str(r)}" for i, r in enumerate(prev_responses)])
    )

def classify_null(response, content):
    """Distinguish truncation / refusal / api_error / empty_content so that
    a high null rate is diagnosable rather than just visible."""
    if response is None:
        return "api_error"
    reason = getattr(response.choices[0], "finish_reason", None)
    if reason == "length":
        return "truncated"
    if content and content.strip():
        return "format_violation"   # produced text, no parseable answer
    return "empty_content"

async def run_llm(model, user_prompt, prev_response=None, temperature=None,
                  max_tokens=None, system_prompt=None, order=None):
    """
    Returns a dict: content, finish_reason, reasoning, null_reason, attempts,
    latency_s. Never a bare string — callers need the metadata to diagnose
    nulls and to audit which model said what.
    """
    attempts = 0
    t0 = time.time()
    last = "api_error"
    for sleep_time in [1, 2, 4]:
        attempts += 1
        try:
            messages = (
                [
                    {"role": "system",
                     "content": get_system_prompt_with_references(prev_response, order)},
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else ([{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}]
                      if system_prompt else
                      [{"role": "user", "content": user_prompt}])
            )
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE if temperature is None else temperature,
                max_tokens=PROPOSER_MAX_TOKENS if max_tokens is None else max_tokens,
            )
            msg = response.choices[0].message
            content = response.choices[0].message.content
            reasoning = (getattr(msg, "reasoning", None)
                         or getattr(msg, "reasoning_content", None))
            if not content:
                last = classify_null(response, content)
                await asyncio.sleep(sleep_time)
                continue
            return {
                "content": content,
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "reasoning": reasoning,
                "null_reason": None,
                "attempts": attempts,
                "latency_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            print(f"  Error [{model.split('/')[-1]}]: {e}")
            last = "api_error"
            await asyncio.sleep(sleep_time)
    return {"content": None, "finish_reason": None, "reasoning": None,
            "null_reason": last, "attempts": attempts,
            "latency_s": round(time.time() - t0, 2)}


async def run_moa(user_prompt):
    """
    Run the full MoA pipeline on a single prompt and return per-layer responses.
    Dataset-agnostic — the same function is used by every runner. Only the
    prompt formatting and scoring differ per dataset.
    """
    run_log = {"question": user_prompt, "layers": {}}

    results = await asyncio.gather(*[run_llm(model, user_prompt) for model in REFERENCE_MODELS])
    run_log["layers"]["layer_1"] = {m: r["content"] for m, r in zip(REFERENCE_MODELS, results)}
    run_log.setdefault("layer_meta", {})["layer_1"] = {
        m: {k: v for k, v in r.items() if k != "content"}
        for m, r in zip(REFERENCE_MODELS, results)
    }
    texts = [r["content"] for r in results]

    for layer_idx in range(1, LAYERS - 1):
        results = await asyncio.gather(
            *[run_llm(model, user_prompt, prev_response=results) for model in REFERENCE_MODELS]
        )
        run_log["layers"][f"layer_{layer_idx + 1}"] = dict(zip(REFERENCE_MODELS, results))

    # Aggregator now uses the same retried, logged path as proposers.
    # Streaming removed: it blocked the event loop and discarded
    # finish_reason, hiding the cause of elevated aggregator null rates.
    final = await run_llm(AGGREGATOR_MODEL, user_prompt, prev_response=results,
                          max_tokens=AGGREGATOR_MAX_TOKENS)
    run_log["final_response"] = final["content"]
    run_log["final_meta"] = {k: v for k, v in final.items() if k != "content"}
    return run_log