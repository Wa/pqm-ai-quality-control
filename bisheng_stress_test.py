"""Standalone Bisheng stress test script.

Sends multiple concurrent requests to the specified Bisheng flow to
help evaluate throughput and error handling. Configure the base URL,
flow ID, and API key with constants below or environment variables.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import string
import sys
from typing import Dict

import requests

BASE_API_URL = os.getenv("BISHENG_BASE_API_URL", "http://10.31.60.11:3001/api/v1/process")
FLOW_ID = os.getenv("BISHENG_FLOW_ID", "d5f4ffd5568b4e8b803acabb7f499fe5")
API_KEY = os.getenv("BISHENG_API_KEY")
CALL_COUNT = int(os.getenv("BISHENG_CALL_COUNT", "30"))
TIMEOUT_SECONDS = int(os.getenv("BISHENG_TIMEOUT", "120"))

INPUT_NODE_ID = os.getenv("BISHENG_INPUT_NODE_ID", "RetrievalQA-f0f31")
TWEAKS: Dict[str, Dict[str, object]] = {
    "MixEsVectorRetriever-J35CZ": {},
    "Milvus-cyR5W": {},
    "PromptTemplate-bs0vj": {},
    "BishengLLM-768ac": {},
    "ElasticKeywordsSearch-1c80e": {},
    "RetrievalQA-f0f31": {},
    "CombineDocsChain-2f68e": {},
}

PROMPT = (
    "你是一名资深APQP质量工程师，负责检查历史问题是否已在当前项目的DFMEA、PFMEA、控制计划中得到预防。\n\n"
    "你的任务：\n"
    "1. 阅读“历史问题”的描述，理解其失效模式、根本原因、预防措施、检测措施。\n"
    "2. 阅读提供的当前项目文档片段（DFMEA/PFMEA/控制计划）。\n"
    "3. 判断该历史问题的预防与检测措施是否已经被覆盖。\n"
    "4. 若未覆盖，请建议在哪个文档（DFMEA/PFMEA/控制计划）中增加何种控制。\n\n"
    "请仅输出以下JSON格式：\n"
    "{\n"
    '  "status": "已覆盖 | 部分覆盖 | 未覆盖",\n'
    '  "where_covered": [\n'
    '    {"doc_type": "PFMEA", "row_ref": "PFMEA-R12", "说明": "已有UV固化时间5s及拉力测试控制"}\n'
    "  ],\n"
    '  "建议更新": [\n'
    '    {"目标文件": "控制计划", "建议内容": "增加100%拉力测试≥5N", "理由": "对应历史问题NTC探头松脱"}\n'
    "  ]\n"
    "}\n\n"
    "历史问题详情：\n"
    "- 失效模式: 售后市场\n"
    "- 根因: 1.NTC UV保护胶吸水率较大,车辆在湿度环境存储后吸水使NTC处于较大湿度工作环境; 2.FPC制程生产工艺湿度管控不到位,封胶环境湿度大; 3.NTC点胶高度控制不良,密封性不好; 4.NTC零件前期验证不充分; 流出:FPC及模组出厂测试时,因NTC未长期处于电场环境,还未出现失效.\n"
    "- 预防措施: 供应商横展\n"
    "- 检测措施: （未提供）\n"
    "- 严重度: 1\n"
    "- 发生度: D148N50F D148N50D\n"
    "  \"record\": {\n"
    "    \"failure_mode\": \"售后市场\",\n"
    "    \"root_cause\": \"1.NTC UV保护胶吸水率较大,车辆在湿度环境存储后吸水使NTC处于较大湿度工作环境; 2.FPC制程生产工艺湿度管控不到位,封胶环境湿度大; 3.NTC点胶高度控制不良,密封性不好; 4.NTC零件前期验证不充分; 流出:FPC及模组出厂测试时,因NTC未长期处于电场环境,还未出现失效.\",\n"
    "    \"prevention_action\": \"供应商横展\",\n"
    "    \"detection_action\": \"\",\n"
    "    \"sheet_name\": \"乘用车\",\n"
    "    \"serial_number\": \"1\",\n"
    "    \"severity\": \"1\",\n"
    "    \"occurrence\": \"D148N50F D148N50D\",\n"
    "    \"detection\": null,\n"
    "    \"risk_priority\": null,\n"
    "    \"responsible\": null,\n"
    "    \"due_date\": null,\n"
    "    \"status\": null,\n"
    "    \"remarks\": null,\n"
    "    \"header_labels\": {\n"
    "      \"serial_number\": \"序号\",\n"
    "      \"occurrence\": \"产品型号 (手写or关联PLM)\",\n"
    "      \"failure_mode\": \"问题来源 (下拉选择)\",\n"
    "      \"root_cause\": \"原因分析 (人、机、料、法、环、测)\",\n"
    "      \"prevention_action\": \"预防措施&改善经验\",\n"
    "      \"severity\": \"规避部门/SQE\"\n"
    "    }"
    "  }"
)


HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"


def _random_session_id() -> str:
    token = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"stress-{token}"


def build_payload() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "inputs": {"query": PROMPT, "id": INPUT_NODE_ID},
        "history_count": 0,
        "session_id": _random_session_id(),
    }
    if TWEAKS:
        payload["tweaks"] = TWEAKS
    return payload


def invoke_once(index: int) -> str:
    url = f"{BASE_API_URL.rstrip('/')}/{FLOW_ID}"
    payload = build_payload()
    try:
        response = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=TIMEOUT_SECONDS)
        status = response.status_code
        if status == 200:
            return f"#{index:02d}: 200 OK"
        return f"#{index:02d}: {status} {response.reason or 'Unknown'}"
    except requests.Timeout:
        return f"#{index:02d}: TIMEOUT after {TIMEOUT_SECONDS}s"
    except requests.RequestException as error:
        return f"#{index:02d}: ERROR {error}"


def main() -> int:
    if not BASE_API_URL or not FLOW_ID:
        print("Missing Bisheng base URL or flow ID", file=sys.stderr)
        return 1

    print(f"Dispatching {CALL_COUNT} calls to {BASE_API_URL.rstrip('/')}/{FLOW_ID}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(CALL_COUNT, 30)) as executor:
        futures = [executor.submit(invoke_once, i + 1) for i in range(CALL_COUNT)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
