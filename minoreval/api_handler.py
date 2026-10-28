import time
import random
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from typing import List, Dict, Union

from minoreval.utils import load_yaml_config

# --- Module-level configuration loading ---
CONFIG_PATH = "configs/01_models.yml"
models_config = load_yaml_config(CONFIG_PATH)

if models_config is None:
    raise FileNotFoundError(f"Core model configuration not found at {CONFIG_PATH}. Program cannot continue.")

# --- Initialize OpenAI client, pointing to OpenRouter ---
try:
    client = OpenAI(
        base_url=models_config["openrouter"]["base_url"],
        api_key=models_config["openrouter"]["api_key"],
        # Set a reasonable HTTP request timeout
        timeout=60.0,
    )
except KeyError as e:
    raise KeyError(f"Configuration file {CONFIG_PATH} is missing a required key: {e}")


def call_llm(
        model_identifier: str,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        max_retries: int = 5,
        base_delay: float = 5.0,
        **kwargs
) -> str:
    """
    Uniformly calls any LLM via OpenRouter.
    """
    final_messages: List[Dict[str, str]] = []
    if system_prompt:
        final_messages.append({"role": "system", "content": system_prompt})
    final_messages.extend(messages)

    last_exception: Union[Exception, None] = None
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                model=model_identifier,
                messages=final_messages,
                **kwargs
            )

            response_content = chat_completion.choices[0].message.content
            # Ensure the return is not None or just whitespace
            return response_content.strip() if response_content else "Error: Model returned empty content."

        except (RateLimitError, APITimeoutError, APIError) as e:
            last_exception = e
            print(f"  - API Retriable Error (Attempt {attempt + 1}/{max_retries}) on model '{model_identifier}': {e.__class__.__name__}")
            if attempt == max_retries - 1:
                break  # Last attempt failed, break the loop to return an error

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Add a small random jitter to avoid synchronized retries
            print(f"    Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

        except Exception as e:
            last_exception = e
            print(f"  - Unexpected Error (Attempt {attempt + 1}/{max_retries}) on model '{model_identifier}': {e.__class__.__name__}")
            if attempt == max_retries - 1:
                break

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"    Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

    error_message = f"API_ERROR: Call failed after {max_retries} attempts. Last error on model '{model_identifier}': {last_exception}"
    print(f"Error: {error_message}")
    return error_message