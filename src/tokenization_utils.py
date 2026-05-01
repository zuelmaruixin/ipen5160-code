from __future__ import annotations

from typing import Any, Dict

from transformers import AutoTokenizer


def build_aspect_pair_text(aspect_name: object, aspect_description: object) -> str:
    name = str(aspect_name).strip()
    description = str(aspect_description).strip()
    if not description:
        return name
    if description == name:
        return name
    return f"{name} {description}".strip()


def encode_aspect_example(
    tokenizer: Any,
    review_text: object,
    aspect_name: object,
    aspect_description: object,
    *,
    max_length: int,
    padding: str | bool = "max_length",
    truncation: str | bool = "only_first",
    return_tensors: str | None = "pt",
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    return tokenizer(
        str(review_text),
        build_aspect_pair_text(aspect_name, aspect_description),
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
        add_special_tokens=add_special_tokens,
    )


def load_tokenizer_with_local_fallback(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except OSError:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
