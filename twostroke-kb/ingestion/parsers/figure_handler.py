"""STRETCH — figure & diagram intelligence.

Extract embedded figures from PDFs, caption them with a vision-LLM (searchable text),
and store captions as chunks for semantic retrieval.

Gated by ENABLE_FIGURE_HANDLER=true in .env. When disabled, returns an empty ParsedDoc.
Requires: PyMuPDF (fitz), Pillow. Vision LLM: openai or anthropic (not ollama).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from ..types import ParsedDoc

log = logging.getLogger(__name__)


def extract_and_caption(path: str | Path) -> ParsedDoc:
    """Extract images from every PDF page and caption each with the vision LLM.

    Returns a ParsedDoc whose text is the joined figure captions (one per line),
    and whose images list holds the raw PNG bytes for each figure.
    When ENABLE_FIGURE_HANDLER is false, immediately returns an empty ParsedDoc.
    """
    from config import get_settings

    settings = get_settings()
    path = Path(path)

    if not settings.enable_figure_handler:
        return ParsedDoc(
            text="",
            metadata={"filename": path.name, "type": "figure"},
            source_ref={"filename": path.name},
        )

    import fitz  # PyMuPDF
    from PIL import Image
    from llm import describe_image

    pdf = fitz.open(str(path))
    captions: list[str] = []
    image_bytes_list: list[bytes] = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = pdf.extract_image(xref)
                img_bytes = base_image["image"]
            except Exception:
                continue

            # Normalise to RGB PNG for consistent LLM input
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                if pil_img.mode not in ("RGB", "L"):
                    pil_img = pil_img.convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
            except Exception:
                log.debug("figure_handler: could not decode image xref=%d on p.%d", xref, page_num + 1)
                continue

            try:
                caption = describe_image(png_bytes)
                captions.append(f"[Figure p.{page_num + 1}/{img_index + 1}]: {caption}")
                image_bytes_list.append(png_bytes)
            except Exception as exc:
                log.debug("figure_handler: vision LLM failed for xref=%d: %s", xref, exc)

    pdf.close()

    return ParsedDoc(
        text="\n\n".join(captions),
        images=image_bytes_list,
        metadata={
            "filename": path.name,
            "type": "figure",
            "figure_count": len(captions),
        },
        source_ref={"filename": path.name},
    )
