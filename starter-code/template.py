"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = "mock-key"

    client = OpenAI(api_key=api_key)

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # Calculate latency
    latency = time.time() - start_time

    # Extract response text
    response_text = response.choices[0].message.content

    # Extract token usage
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

    return response_text, latency, usage
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API and return response text,
    latency, and token usage stats.

    Supports both new google-genai SDK and legacy
    google-generativeai SDK.
    """

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or "mock-key"
    )

    start_time = time.time()

    # Try importing the new Google GenAI SDK
    try:
        from google import genai
        from google.genai import types

        use_new_sdk = True

    except ImportError:
        use_new_sdk = False

    # Option A: New Google GenAI SDK
    if use_new_sdk:

        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        response_text = response.text or ""

        usage_metadata = response.usage_metadata

        input_tokens = (
            usage_metadata.prompt_token_count
            if usage_metadata else 0
        )

        output_tokens = (
            usage_metadata.candidates_token_count
            if usage_metadata else 0
        )

    # Option B: Legacy Google GenerativeAI SDK
    else:

        import google.generativeai as genai

        genai.configure(api_key=api_key)

        model_inst = genai.GenerativeModel(model)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )

        response = model_inst.generate_content(
            prompt,
            generation_config=generation_config
        )

        response_text = response.text or ""

        usage_metadata = response.usage_metadata

        input_tokens = (
            usage_metadata.prompt_token_count
            if usage_metadata else 0
        )

        output_tokens = (
            usage_metadata.candidates_token_count
            if usage_metadata else 0
        )

    # Calculate latency
    latency = time.time() - start_time

    # Token usage statistics
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

    return response_text, latency, usage
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    raise NotImplementedError("Implement call_gemini")


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------

def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:

    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"

    client = anthropic.Anthropic(api_key=api_key)

    start_time = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Calculate latency
    latency = time.time() - start_time

    # Extract response text
    response_text = "".join(
        block.text
        for block in response.content
        if block.type == "text"
    )

    # Extract token usage
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------

def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash
    with the same prompt and return a structured comparison dictionary.

    Calculate exact USD token cost for input + output.
    """

    # Call GPT-4o
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(
        prompt,
        model="gpt-4o"
    )

    gpt4o_cost = (
        gpt4o_usage["input_tokens"] *
        PRICING_1M_TOKENS["gpt-4o"]["input"] +
        gpt4o_usage["output_tokens"] *
        PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    # Call GPT-4o-mini
    mini_text, mini_lat, mini_usage = call_openai(
        prompt,
        model="gpt-4o-mini"
    )

    mini_cost = (
        mini_usage["input_tokens"] *
        PRICING_1M_TOKENS["gpt-4o-mini"]["input"] +
        mini_usage["output_tokens"] *
        PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    # Call Gemini 2.5 Flash
    gemini_text, gemini_lat, gemini_usage = call_gemini(
        prompt,
        model="gemini-2.5-flash"
    )

    gemini_cost = (
        gemini_usage["input_tokens"] *
        PRICING_1M_TOKENS["gemini-2.5-flash"]["input"] +
        gemini_usage["output_tokens"] *
        PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    # Assemble comparison dictionary
    comparison = {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"]
        },

        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"]
        },

        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"]
        }
    }

    return comparison



# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------

def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Maintains the last 3 turns of conversation history for context.
    """

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        print(
            "\033[93m[System Warning] GEMINI_API_KEY environment "
            "variable not set. Running in dummy mode.\033[0m"
        )

    print("\n\033[94m" + "=" * 60 + "\033[0m")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\033[0m\n")

    # Initialize conversation history
    history = []

    # Dummy mode if API key is missing
    if not api_key:
        print("[Demo mode] Gemini API key is not configured.")
        print("Please configure GEMINI_API_KEY to use real AI responses.")
        return

    # Initialize Gemini client
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    try:
        while True:

            # Get user input
            user_input = input("\n\033[92mYou: \033[0m").strip()

            # Check exit commands
            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye! Thank you for using Vin Smart Future.")
                break

            if not user_input:
                continue

            # Add current user message to conversation history
            history.append({
                "role": "user",
                "parts": [{"text": user_input}]
            })

            print("\n\033[94mGemini: \033[0m", end="", flush=True)

            try:
                # Stream response from Gemini
                response_stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=history,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.9,
                        max_output_tokens=1024
                    )
                )

                response_text = ""

                for chunk in response_stream:
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        response_text += chunk.text

                print()

                # Save Gemini response to conversation history
                history.append({
                    "role": "model",
                    "parts": [{"text": response_text}]
                })

                # Keep only the last 3 conversation turns
                history = history[-6:]

            except Exception as e:
                print(f"\n[Error] Gemini API request failed: {e}")

                # Remove the unanswered user message
                history.pop()

    except KeyboardInterrupt:
        print("\n\nChat session interrupted. Goodbye!")

    finally:
        client.close()



import time
from typing import Callable, Any


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------

def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """

    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    if base_delay < 0:
        raise ValueError("base_delay must be >= 0")

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return fn()

        except Exception as e:
            last_exception = e

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)

                print(
                    f"[Retry {attempt + 1}/{max_retries}] "
                    f"Error: {e}. Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

    # Raise the last exception after all retries
    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------

def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Returns a list of comparison dictionaries with an extra
    key 'prompt' containing the original prompt.
    """

    results = []

    for prompt in prompts:

        try:
            # Compare all models using the same prompt
            comparison = compare_models(prompt)

            # Add original prompt to comparison result
            comparison["prompt"] = prompt

            results.append(comparison)

        except Exception as e:
            print(f"[Error] Failed to compare prompt: {prompt}")
            print(f"Reason: {e}")

            continue

    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------

def format_comparison_table(results: list[dict]) -> str:
    """
    Format batch comparison results as a Markdown table.

    Response text is truncated to 50 characters.
    """

    headers = [
        "Prompt",
        "Model",
        "Response (truncated)",
        "Latency",
        "Tokens (In/Out)",
        "Cost (USD)"
    ]

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    # Helper to clean text for Markdown table
    def clean_text(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    # Process each prompt result
    for result in results:

        prompt = clean_text(result.get("prompt", ""))

        # Model information
        model_names = [
            "gpt4o",
            "gpt4o_mini",
            "gemini_flash"
        ]

        for model_name in model_names:

            if model_name not in result:
                continue

            stats = result[model_name]

            response = clean_text(stats.get("response", ""))

            # Truncate response to 50 characters
            if len(response) > 50:
                response = response[:47] + "..."

            latency = stats.get("latency", 0)
            input_tokens = stats.get("input_tokens", 0)
            output_tokens = stats.get("output_tokens", 0)
            cost = stats.get("cost", 0)

            row = [
                prompt,
                model_name,
                response,
                f"{latency:.2f}s",
                f"{input_tokens}/{output_tokens}",
                f"${cost:.8f}"
            ]

            table.append(
                "| " + " | ".join(row) + " |"
            )

    return "\n".join(table)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=== Model Comparison Test ===")

    test_prompt = (
        "Hãy giải thích sự khác biệt giữa temperature và top_p "
        "bằng tiếng Việt ngắn gọn trong 2 câu."
    )

    try:
        # Requires valid API keys
        result = compare_models(test_prompt)

        for model_name, stats in result.items():

            print(f"\n[{model_name.upper()}]")

            print(
                f"Latency: {stats['latency']:.2f}s | "
                f"Cost: ${stats['cost']:.6f}"
            )

            print(
                f"Tokens: {stats['input_tokens']} in / "
                f"{stats['output_tokens']} out"
            )

            print(f"Response: {stats['response']}")

    except Exception as e:

        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")

    try:
        streaming_chatbot()

    except Exception as e:
        print(f"Chatbot failed to start: {e}")
