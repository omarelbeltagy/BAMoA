# Mixture-of-Agents in 50 lines of code
import asyncio
import os
import together
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

async def run_llm(model):
    """Run a single LLM call with a reference model."""
    for sleep_time in [1, 2, 4]:
        try:
            response = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling {model}: {e}")
            await asyncio.sleep(sleep_time)
    return None

async def main():
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggreagator_system_prompt + "\n" + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in finalStream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
