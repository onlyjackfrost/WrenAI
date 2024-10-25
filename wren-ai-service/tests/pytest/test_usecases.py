import asyncio
import base64
import json
import os
import time
import uuid

import aiohttp
import orjson
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

GLOBAL_DATA = {
    "semantics_preperation_id": str(uuid.uuid4()),
    "query_id": None,
}

BASE_URL = "http://localhost:5556"


def _replace_wren_engine_env_variables(engine_type: str, data: dict):
    with open("config.yaml", "r") as f:
        configs = list(yaml.safe_load_all(f))

        for config in configs:
            if config["type"] == "engine" and config["provider"] == engine_type:
                for key, value in data.items():
                    config[key] = value
            if "pipes" in config:
                for i, pipe in enumerate(config["pipes"]):
                    if "engine" in pipe:
                        config["pipes"][i]["engine"] = engine_type

    with open("config.yaml", "w") as f:
        yaml.safe_dump_all(configs, f, default_flow_style=False)


def _deploy_mdl(mdl_str, id):
    response = requests.post(
        url=f"{BASE_URL}/v1/semantics-preparations",
        json={
            "mdl": mdl_str,
            "id": id,
        },
    )

    assert response.status_code == 200
    assert response.json()["id"] == id

    status = "indexing"

    while status == "indexing":
        response = requests.get(url=f"{BASE_URL}/v1/semantics-preparations/{id}/status")

        assert response.status_code == 200
        assert response.json()["status"] in ["indexing", "finished", "failed"]
        status = response.json()["status"]

    assert status == "finished"


async def test_hubspot_questions():
    semantics_preperation_id = GLOBAL_DATA["semantics_preperation_id"]

    with open("tests/data/hubspot/mdl.json", "r") as f:
        mdl_json = json.load(f)
        mdl_str = orjson.dumps(mdl_json).decode("utf-8")

    _replace_wren_engine_env_variables(
        "wren_ibis",
        {
            "manifest": base64.b64encode(orjson.dumps(mdl_json)).decode(),
            "source": "bigquery",
            "connection_info": base64.b64encode(
                orjson.dumps(
                    {
                        "project_id": os.getenv("bigquery.project-id"),
                        "dataset_id": os.getenv("bigquery.dataset-id"),
                        "credentials": os.getenv("bigquery.credentials-key"),
                    }
                )
            ).decode(),
        },
    )

    time.sleep(10)

    _deploy_mdl(mdl_str, semantics_preperation_id)

    with open("tests/data/hubspot/questions.json", "r") as f:
        questions = json.load(f)["questions"]

    async def ask_question(question, id):
        async with aiohttp.request(
            "POST", f"{BASE_URL}/v1/asks", json={"query": question, "id": id}
        ) as response:
            return await response.json()

    async def get_ask_result(id):
        while True:
            async with aiohttp.request(
                "GET", f"{BASE_URL}/v1/asks/{id}/result"
            ) as respnose:
                result = await respnose.json()
                if result.get("status") in ("failed", "finished"):
                    return result
                await asyncio.sleep(1)

    responses = await asyncio.gather(
        *[ask_question(q, semantics_preperation_id) for q in questions]
    )

    response_ids = [response["query_id"] for response in responses]

    final_results = await asyncio.gather(
        *[get_ask_result(response_id) for response_id in response_ids]
    )

    print(final_results)


asyncio.run(test_hubspot_questions())
