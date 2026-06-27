"""Unified LLM client. One place that talks to OpenAI / Anthropic / local OpenAI-compat endpoint.

Every module that needs the LLM imports from here so behaviour is consistent.
Provider + model come from .env (LLM_PROVIDER, LLM_MODEL).

    from llm import chat, stream_chat, describe_image
    text = chat([{"role": "user", "content": "Hallo"}])
    data = chat_json([...])          # when you need a JSON object back
    for tok in stream_chat([...]): …  # token-by-token streaming
    caption = describe_image(png_bytes)  # vision LLM (OpenAI / Anthropic / local)

LLM_PROVIDER=local  uses LLM_BASE_URL + LLM_API_KEY via the OpenAI SDK — works with
Ollama (/v1), vLLM, LM Studio, or any OpenAI-compatible server.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from config import get_settings

Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


def chat(
    messages: list[Message],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    model: str | None = None,
) -> str:
    """Send a chat completion and return the assistant text. Dispatches by provider."""
    s = get_settings()
    provider = s.llm_provider.lower()
    model = model or s.llm_model

    if provider in ("openai", "local"):
        return _openai_compat(messages, temperature, max_tokens, model)
    if provider == "anthropic":
        return _anthropic(messages, temperature, max_tokens, model)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def chat_json(messages: list[Message], **kwargs: Any) -> Any:
    """Like chat() but parse the reply as JSON. Falls back to extracting the first
    {...} or [...] block if the model wraps it in prose."""
    raw = chat(messages, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = min((raw.find(c) for c in "[{" if raw.find(c) != -1), default=-1)
        end = max(raw.rfind("]"), raw.rfind("}"))
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


# --- providers -------------------------------------------------------------

def _openai_compat(messages, temperature, max_tokens, model) -> str:
    """OpenAI SDK pointed at LLM_BASE_URL — works for openai, local Ollama /v1, vLLM, etc."""
    from openai import OpenAI

    s = get_settings()
    client = OpenAI(api_key=s.llm_api_key or s.openai_api_key, base_url=s.llm_base_url)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content or ""


def _anthropic(messages, temperature, max_tokens, model) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        system=system or None,
        messages=convo,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return "".join(block.text for block in resp.content if block.type == "text")


# --- streaming -----------------------------------------------------------------

def stream_chat(
    messages: list[Message],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    model: str | None = None,
) -> Iterator[str]:
    """Yield text tokens as they arrive from the LLM."""
    s = get_settings()
    provider = s.llm_provider.lower()
    model = model or s.llm_model

    if provider in ("openai", "local"):
        yield from _openai_compat_stream(messages, temperature, max_tokens, model)
    elif provider == "anthropic":
        yield from _anthropic_stream(messages, temperature, max_tokens, model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _openai_compat_stream(messages, temperature, max_tokens, model) -> Iterator[str]:
    from openai import OpenAI

    s = get_settings()
    client = OpenAI(api_key=s.llm_api_key or s.openai_api_key, base_url=s.llm_base_url)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _anthropic_stream(messages, temperature, max_tokens, model) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] != "system"]

    with client.messages.stream(
        model=model,
        system="\n".join(system_parts) if system_parts else "",
        messages=convo,
        temperature=temperature,
        max_tokens=max_tokens,
    ) as s:
        yield from s.text_stream


# --- vision --------------------------------------------------------------------

def describe_image(
    image_bytes: bytes,
    prompt: str = (
        "This is a technical diagram from a two-stroke engine manual. "
        "Describe what you see: components, labels, measurements, and relationships."
    ),
) -> str:
    """Send image bytes to the vision-capable LLM and return a text description.

    Uses LLM_VISION_MODEL (default llama3.2:3b) via the same LLM_BASE_URL endpoint.
    Supports local (Ollama /v1), openai, and anthropic providers.
    """
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    s = get_settings()
    provider = s.llm_provider.lower()
    model = s.llm_vision_model  # dedicated vision model, separate from chat model

    if provider in ("openai", "local"):
        from openai import OpenAI

        client = OpenAI(api_key=s.llm_api_key or s.openai_api_key, base_url=s.llm_base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    if provider == "anthropic":
        import anthropic as _anthropic

        client = _anthropic.Anthropic(api_key=s.anthropic_api_key)
        resp = client.messages.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    raise ValueError(
        f"describe_image: provider {provider!r} does not support vision. "
        "Set LLM_PROVIDER=openai, local, or anthropic."
    )
