"""Simple smoke test for ModelScope DeepSeek chat completion."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CONFIG
from tabs.special_symbols.background import ModelScopeClient


def main() -> int:
    llm_config: dict[str, Any] = dict(CONFIG.get("llm", {}))
    api_key = llm_config.get("modelscope_api_key")
    model = llm_config.get("modelscope_model", "deepseek-ai/DeepSeek-V3.1")
    base_url = llm_config.get("modelscope_base_url", "https://api-inference.modelscope.cn/api/v1")

    if not api_key:
        print("ModelScope API key missing in CONFIG['llm']['modelscope_api_key']", file=sys.stderr)
        return 1

    client = ModelScopeClient(api_key=api_key, model=model, base_url=base_url)
    prompt = "请用简体中文简要介绍你自己，并说明你来自 ModelScope DeepSeek-V3.1 服务。"
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"num_ctx": 40001},
    )

    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
