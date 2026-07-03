# Advanced Mixture-of-Agents example – 3 layers
import asyncio
import json
import os
from datetime import datetime
from together import AsyncTogether, Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

user_prompt = "What are 3 fun things to do in SF?"
reference_models = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai/gpt-oss-20b",
    "google/gemma-4-31B-it",
    "deepseek-ai/DeepSeek-V4-Pro",
]
aggregator_model = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
layers = 3


def getFinalSystemPrompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


async def run_llm(model, prev_response=None):
    """Run a single LLM call with a model while accounting for previous responses + rate limits."""
    for sleep_time in [1, 2, 4]:
        try:
            messages = (
                [
                    {
                        "role": "system",
                        "content": getFinalSystemPrompt(
                            aggreagator_system_prompt, prev_response
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            if not content:
                print(f"Warning: empty response from {model}, retrying...")
                await asyncio.sleep(sleep_time)
                continue
            print("Model: ", model)
            return content
        except Exception as e:
            print(f"Error calling {model}: {e}")
            await asyncio.sleep(sleep_time)
    print(f"Failed to get response from {model} after all retries")
    return None


async def main():
    """Run the main loop of the MOA process."""
    run_log = {"question": user_prompt, "layers": {}}

    # Layer 1: proposers answer the raw question
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])
    run_log["layers"]["layer_1"] = dict(zip(reference_models, results))

    # Middle layers: each proposer also sees the previous layer's responses
    for layer_idx in range(1, layers - 1):
        results = await asyncio.gather(
            *[run_llm(model, prev_response=results) for model in reference_models]
        )
        run_log["layers"][f"layer_{layer_idx + 1}"] = dict(zip(reference_models, results))

    # Final aggregation: accumulate streamed output so we can log it
    final_response = ""
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {
                "role": "system",
                "content": getFinalSystemPrompt(aggreagator_system_prompt, results),
            },
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in finalStream:
        if chunk.choices:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            final_response += content

    run_log["final_response"] = final_response

    os.makedirs("outputs/raw", exist_ok=True)
    log_path = f"outputs/raw/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\n\nLog saved → {log_path}")


asyncio.run(main())
