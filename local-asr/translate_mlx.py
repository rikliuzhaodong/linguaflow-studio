from __future__ import annotations

import json
import sys
from pathlib import Path

from mlx_lm import generate, load


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-3B-Instruct-4bit"


def main() -> None:
    payload = json.loads(sys.stdin.read())
    lines = payload.get("lines", []) if isinstance(payload, dict) else payload
    target = payload.get("target", "zh") if isinstance(payload, dict) else "zh"
    if not isinstance(lines, list) or not lines:
        print(json.dumps({"translations": []}, ensure_ascii=False))
        return

    model, tokenizer = load(str(MODEL_PATH))
    if target == "en":
        instruction = (
            "Translate every Chinese subtitle into accurate, natural English. Preserve all information and specialist terms; "
            "translate 八字命理 as Bazi astrology or Bazi destiny analysis."
        )
    else:
        instruction = "把每条英文字幕准确、自然、完整地翻译为简体中文。"
    translations = []
    for line in lines:
        messages = [
            {"role": "system", "content": instruction + " Output only the translated subtitle, without explanation or quotation marks."},
            {"role": "user", "content": str(line)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = generate(model, tokenizer, prompt=prompt, max_tokens=192, verbose=False).strip()
        translations.append(result.strip('"\' '))
    print(json.dumps({"translations": [str(item) for item in translations]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
