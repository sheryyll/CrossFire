import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# Known free/local providers and their OpenAI-compatible base URLs.
PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}

# Env var each provider expects its API key under, if it needs one at all.
PROVIDER_API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,  # local Ollama needs no key
}


class LLMClientError(Exception):
    """Raised when a call ultimately fails after retries. Callers (the agent
    layer) are expected to catch this and fall back to a safe default action,
    per the reliability contract - a failed LLM call should never crash a game."""
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_seconds: float


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 2,
        timeout_seconds: float = 30.0,
    ):
        if provider not in PROVIDER_BASE_URLS and base_url is None:
            raise ValueError(
                f"Unknown provider '{provider}'. Pass one of "
                f"{list(PROVIDER_BASE_URLS)} or supply a custom base_url."
            )

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        resolved_base_url = base_url or PROVIDER_BASE_URLS[provider]
        env_var = PROVIDER_API_KEY_ENV.get(provider)
        resolved_key = api_key or (os.environ.get(env_var) if env_var else None)

        if env_var and not resolved_key:
            raise ValueError(
                f"No API key found for provider '{provider}'. "
                f"Set the {env_var} environment variable or pass api_key= explicitly."
            )

        self._client = OpenAI(
            api_key=resolved_key or "not-needed-for-local-ollama",
            base_url=resolved_base_url,
            timeout=timeout_seconds,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Sends one chat completion request and returns the raw text response.
        Raw text, NOT parsed JSON - parsing/validation happens one layer up
        (agents/parsing.py), per the agent decision contract's fallback-parser
        design (Section 6)."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                latency = time.monotonic() - start
                text = completion.choices[0].message.content or ""
                return LLMResponse(
                    text=text, provider=self.provider, model=self.model,
                    latency_seconds=latency,
                )
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))  # simple backoff
                    continue
        raise LLMClientError(
            f"LLM call failed after {self.max_retries + 1} attempts "
            f"(provider={self.provider}, model={self.model}): {last_error}"
        )


def list_available_models(provider: str, api_key: Optional[str] = None) -> list:
    """Hits the provider's /models endpoint directly. Model lineups on Groq and
    OpenRouter change over time (deprecations, renames, new releases), so
    rather than hardcode a model ID and hope it's still valid, this asks the
    account itself what's currently accessible."""
    base_url = PROVIDER_BASE_URLS[provider]
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    resolved_key = api_key or (os.environ.get(env_var) if env_var else None)
    if env_var and not resolved_key:
        raise ValueError(f"Set {env_var} to list models for provider '{provider}'.")
    client = OpenAI(api_key=resolved_key or "not-needed", base_url=base_url)
    return [m.id for m in client.models.list().data]


def build_client_from_config(config: dict) -> LLMClient:
    """Convenience factory so agent/tournament configs can be plain dicts
    (easy to log as tournament metadata, per Evaluation Spec Section 6)."""
    return LLMClient(
        provider=config["provider"],
        model=config["model"],
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 500),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
    )
