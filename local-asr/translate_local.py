from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get(
        "FLUENTFRAME_TRANSFORMERS_TRANSLATOR_MODEL",
        str(PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct"),
    )
)


def main() -> None:
    payload = json.loads(sys.stdin.read())
    if isinstance(payload, dict):
        lines = payload.get("lines", [])
        target = payload.get("target", "zh")
    else:
        lines = payload
        target = "zh"
    if not isinstance(lines, list) or not lines:
        print(json.dumps({"translations": []}, ensure_ascii=False))
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    instruction = (
        "你是中译英字幕翻译器。把每条中文准确、自然地翻译为英文。保留专有名词和术语的准确含义，例如‘八字命理’译为‘Bazi destiny analysis’或‘Bazi astrology’，不要删减信息。"
        if target == "en" else
        "你是英译中字幕翻译器。把每条英文准确、自然、简洁地翻译为简体中文，不要删减信息。"
    )
    messages = [
        {
            "role": "system",
            "content": instruction + "保持数量和顺序完全一致。只输出 JSON 数组，不要代码块，不要解释。",
        },
        {"role": "user", "content": json.dumps(lines, ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    max_new_tokens = min(4096, max(256, len(lines) * 48))
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    start = text.find("[")
    end = text.rfind("]")
    translations = json.loads(text[start : end + 1]) if start >= 0 and end > start else []
    if not isinstance(translations, list):
        translations = []
    print(json.dumps({"translations": [str(item) for item in translations]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
