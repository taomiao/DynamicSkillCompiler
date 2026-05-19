import os
import time
import re
from openai import OpenAI
import json
_CLIENT = None
_CLIENT_CONFIG = None


def _get_timeout_seconds():
    return float(os.environ.get("EXPERIMENT_LLM_TIMEOUT_SECONDS", "90"))


def _get_retry_attempts():
    return int(os.environ.get("EXPERIMENT_LLM_RETRY_ATTEMPTS", "3"))


def _get_retry_delay_seconds():
    return float(os.environ.get("EXPERIMENT_LLM_RETRY_DELAY_SECONDS", "3"))


def _get_client():
    global _CLIENT, _CLIENT_CONFIG
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "API_KEY and BASE_URL (or OPENAI_API_KEY and OPENAI_BASE_URL) must be set before calling the experiment LLM."
        )
    client_config = (api_key, base_url, _get_timeout_seconds())
    if _CLIENT is None or _CLIENT_CONFIG != client_config:
        _CLIENT = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=client_config[2],
            max_retries=0,
        )
        _CLIENT_CONFIG = client_config
    return _CLIENT


def chat_completion(messages, model="gpt-4o"):
    attempts = max(_get_retry_attempts(), 1)
    delay_seconds = max(_get_retry_delay_seconds(), 0.0)
    last_error = None
    for attempt in range(attempts):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=messages,
            )
            return response
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1:
                raise
            sleep_seconds = min(delay_seconds * (2 ** attempt), 20.0)
            print(
                f"[WARN] LLM request failed on attempt {attempt + 1}/{attempts}: {exc}. "
                f"Retrying in {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)
    raise last_error


def get_llm_response(messages, is_string=False, model="gpt-4o"):
    response = chat_completion(messages=messages, model=model)
    if not hasattr(response, "error"):
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("LLM returned no choices in response.")
        ans = choices[0].message.content
        if is_string:
            return ans
        else:
            cleaned_text = ans.strip("`json\n").strip("`\n").strip("```\n")
            ans = json.loads(cleaned_text)
            return ans
    else:
        raise Exception(response.error.message)


def strip_code_fences(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    return cleaned.strip()


def extract_tagged_text(text, tag):
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    matches = re.findall(pattern, text or "", flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return ""
    return strip_code_fences(matches[-1])


def extract_tagged_json(text, tag, default=None):
    content = extract_tagged_text(text, tag)
    if not content:
        return [] if default is None else default
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [] if default is None else default
