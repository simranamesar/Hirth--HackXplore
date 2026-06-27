"""Unified LLM client. One place that talks to OpenAI / Anthropic / Ollama.

Every module that needs the LLM imports from here so behaviour is consistent.
Provider + model come from .env (LLM_PROVIDER, LLM_MODEL).

    from llm import chat, stream_chat, describe_image
    text = chat([{"role": "user", "content": "Hallo"}])
    data = chat_json([...])          # when you need a JSON object back
    for tok in stream_chat([...]): …  # token-by-token streaming
    caption = describe_image(png_bytes)  # vision LLM (OpenAI / Anthropic only)
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

    if provider == "openai":
        return _openai(messages, temperature, max_tokens, model)
    if provider == "anthropic":
        return _anthropic(messages, temperature, max_tokens, model)
    if provider == "ollama":
        return _ollama(messages, temperature, max_tokens, model)
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

def _openai(messages, temperature, max_tokens, model) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=get_settings().openai_api_key)
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


def _ollama(messages, temperature, max_tokens, model) -> str:
    import httpx

    s = get_settings()
    url = f"{s.ollama_base_url}/api/chat"
    payload = {
        "model": s.ollama_model if model == s.llm_model else model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = httpx.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


# --- streaming -----------------------------------------------------------------

def stream_chat(
    messages: list[Message],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    model: str | None = None,
) -> Iterator[str]:
    """Yield text tokens as they arrive from the LLM.

    OpenAI and Anthropic support true streaming.
    Ollama: emits the full response as a single chunk (no streaming API used).
    """
    s = get_settings()
    provider = s.llm_provider.lower()
    model = model or s.llm_model

    if provider == "openai":
        yield from _openai_stream(messages, temperature, max_tokens, model)
    elif provider == "anthropic":
        yield from _anthropic_stream(messages, temperature, max_tokens, model)
    else:
        yield _ollama(messages, temperature, max_tokens, model)


def _openai_stream(messages, temperature, max_tokens, model) -> Iterator[str]:
    from openai import OpenAI

    client = OpenAI(api_key=get_settings().openai_api_key)
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

    Supported providers: openai, anthropic.
    Raises ValueError for Ollama (no vision model wired up by default).
    """
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    s = get_settings()
    provider = s.llm_provider.lower()
    model = s.llm_model

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=s.openai_api_key)
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
        "Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic."
    )
