"""LLM access for translation and proofreading.

Two client families:

* ``anthropic`` - Claude through the official ``anthropic`` Python SDK
  (``pip install anthropic``; credentials from ``ANTHROPIC_API_KEY`` or an
  ``ant auth login`` profile).
* OpenAI-compatible chat-completions endpoints over plain HTTP - covers
  OpenAI, DeepSeek, Qwen (DashScope), Moonshot/Kimi, Zhipu GLM, local Ollama /
  vLLM / LM Studio, or any custom ``base_url``.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

# provider -> (base_url, default model, api-key env var)
OPENAI_COMPATIBLE: Dict[str, tuple] = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", "DASHSCOPE_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-8k", "MOONSHOT_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-plus", "ZHIPUAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", "OLLAMA_API_KEY"),
    "custom": (None, None, "LLM_API_KEY"),
}

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    pass


class LLMClient:
    """``complete(system, user, schema=None) -> str`` (JSON text when a schema is given)."""

    name = "base"

    def complete(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> Any:
        return parse_json_loose(self.complete(system, user, schema))


@dataclass
class AnthropicClient(LLMClient):
    model: str = ANTHROPIC_DEFAULT_MODEL
    api_key: Optional[str] = None
    effort: Optional[str] = None          # low | medium | high | xhigh | max
    max_tokens: int = 32000
    fallbacks: bool = True                # server-side refusal fallback
    name: str = "anthropic"

    def __post_init__(self):
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover - optional
            raise ImportError("pip install anthropic") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()

    def complete(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> str:
        anthropic = self._anthropic
        kwargs: Dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      messages=[{"role": "user", "content": user}])
        output_config: Dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config
        if self.fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["extra_body"] = {"fallbacks": "default"}
        try:
            # streaming keeps long translations clear of HTTP timeouts
            with self._client.beta.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except anthropic.BadRequestError as e:
            raise LLMError(f"Claude API rejected the request: {e.message}") from e
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude API authentication failed - set ANTHROPIC_API_KEY") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"cannot reach the Claude API: {e}") from e
        if msg.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if msg.stop_reason == "max_tokens":
            raise LLMError("response truncated (max_tokens); use a smaller batch size")
        return text


@dataclass
class OpenAICompatClient(LLMClient):
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.2
    timeout: float = 300.0
    retries: int = 3
    json_mode: bool = True
    name: str = "openai-compatible"

    def complete(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> str:
        body: Dict[str, Any] = {"model": self.model, "temperature": self.temperature,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]}
        if schema is not None and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code == 400 and "response_format" in body:
                    body.pop("response_format")       # provider without JSON mode
                    continue
                if e.code in (408, 409, 429) or e.code >= 500:
                    last = LLMError(f"HTTP {e.code}: {detail}")
                else:
                    raise LLMError(f"HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
            time.sleep(min(30, 2 ** attempt))
        raise LLMError(f"LLM request failed: {last}")


def make_client(provider: str = "anthropic", model: Optional[str] = None, base_url: Optional[str] = None,
                api_key: Optional[str] = None, **kw) -> LLMClient:
    provider = provider.lower()
    if provider in ("anthropic", "claude"):
        return AnthropicClient(model=model or ANTHROPIC_DEFAULT_MODEL, api_key=api_key,
                               **{k: v for k, v in kw.items() if k in ("effort", "max_tokens", "fallbacks")})
    if provider not in OPENAI_COMPATIBLE:
        raise ValueError(f"unknown LLM provider {provider!r}; choose anthropic or one of {sorted(OPENAI_COMPATIBLE)}")
    url, default_model, env = OPENAI_COMPATIBLE[provider]
    url = base_url or os.environ.get("LLM_BASE_URL") or url
    if not url:
        raise ValueError("provider 'custom' requires --llm-base-url")
    m = model or default_model
    if not m:
        raise ValueError("provider 'custom' requires --llm-model")
    key = api_key or os.environ.get(env) or os.environ.get("LLM_API_KEY", "")
    return OpenAICompatClient(model=m, base_url=url, api_key=key,
                              **{k: v for k, v in kw.items() if k in ("temperature", "timeout", "json_mode")})


def parse_json_loose(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences / surrounding prose."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = t.find(open_c), t.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"model did not return valid JSON: {text[:200]!r}")
