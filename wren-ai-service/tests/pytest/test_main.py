import json
import uuid

import orjson
import pytest
from fastapi.testclient import TestClient
from pytest_lazy_fixtures import lf

from src.__main__ import app

GLOBAL_DATA = {
    "mdl_hash": str(uuid.uuid4()),
    "query_id": None,
}


@pytest.fixture
def valid_mdl():
    with open("tests/data/book_2_mdl.json", "r") as f:
        return orjson.dumps(json.load(f)).decode("utf-8")


@pytest.fixture
def invalid_mdl():
    return "invalid-mdl"


@pytest.fixture
def valid_semantics_preparation_data(
    valid_mdl: str,
):
    return {
        "mdl": valid_mdl,
        "mdl_hash": GLOBAL_DATA["mdl_hash"],
    }


@pytest.fixture
def invalid_mdl_semantics_preparation_data(
    invalid_mdl: str,
):
    return {
        "mdl": invalid_mdl,
        "mdl_hash": str(uuid.uuid4()),
    }


@pytest.fixture
def no_mdl_semantics_preparation_data():
    return {
        "mdl_hash": str(uuid.uuid4()),
    }


@pytest.fixture
def no_mdl_hash_semantics_preparation_data(
    valid_mdl: str,
):
    return {
        "mdl": valid_mdl,
    }


@pytest.mark.parametrize(
    "data,status_code,expected_status",
    [
        pytest.param(
            lf("valid_semantics_preparation_data"),
            200,
            "finished",
            marks=pytest.mark.dependency(name="valid_semantics_prep"),
        ),
        (lf("invalid_mdl_semantics_preparation_data"), 200, "failed"),
        (lf("no_mdl_semantics_preparation_data"), 400, None),
        (lf("no_mdl_hash_semantics_preparation_data"), 400, None),
    ],
)
@pytest.mark.dependency()
def test_semantics_preparations(data, status_code, expected_status):
    with TestClient(app) as client:
        response = client.post(
            url="/v1/semantics-preparations",
            json=data,
        )

        if "mdl" in data and "mdl_hash" in data:
            assert response.status_code == 200
            assert response.json()["id"] == data["mdl_hash"]

            status = "indexing"

            while status == "indexing":
                response = client.get(
                    url=f"/v1/semantics-preparations/{data["mdl_hash"]}/status"
                )

                assert response.status_code == status_code
                assert response.json()["status"] in ["indexing", "finished", "failed"]
                status = response.json()["status"]

            assert status == expected_status
        else:
            assert response.status_code == status_code


@pytest.fixture
def valid_ask_data():
    return {
        "query": "how many books",
        "mdl_hash": GLOBAL_DATA["mdl_hash"],
    }


@pytest.fixture
def invalid_ask_data():
    return {
        "query": "xxxxxx",
        "mdl_hash": GLOBAL_DATA["mdl_hash"],
    }


@pytest.fixture
def no_query_ask_data():
    return {
        "mdl_hash": GLOBAL_DATA["mdl_hash"],
    }


@pytest.mark.parametrize(
    "data,status_code,expected_status",
    [
        (lf("valid_ask_data"), 200, "finished"),
        (lf("invalid_ask_data"), 200, "failed"),
        (lf("no_query_ask_data"), 400, None),
    ],
)
@pytest.mark.dependency(depends=["valid_semantics_prep"])
def test_asks(data, status_code, expected_status):
    with TestClient(app) as client:
        response = client.post(
            url="/v1/asks",
            json=data,
        )

        if "query" in data and "mdl_hash" in data:
            assert response.status_code == 200
            assert response.json()["query_id"] != ""

            query_id = response.json()["query_id"]
            GLOBAL_DATA["query_id"] = query_id

            response = client.get(url=f"/v1/asks/{query_id}/result")
            while (
                response.json()["status"] != "finished"
                and response.json()["status"] != "failed"
            ):
                response = client.get(url=f"/v1/asks/{query_id}/result")

            # TODO: we'll refactor almost all test case with a mock server, thus temporarily only assert the status is finished or failed.
            assert response.status_code == status_code
            assert response.json()["status"] == expected_status
        else:
            assert response.status_code == status_code


def test_stop_asks():
    with TestClient(app) as client:
        query_id = GLOBAL_DATA["query_id"]

        response = client.patch(
            url=f"/v1/asks/{query_id}",
            json={
                "status": "stopped",
            },
        )

        assert response.status_code == 200
        assert response.json()["query_id"] == query_id

        response = client.get(url=f"/v1/asks/{query_id}/result")
        while response.json()["status"] != "stopped":
            response = client.get(url=f"/v1/asks/{query_id}/result")

        assert response.status_code == 200
        assert response.json()["status"] == "stopped"


def test_ask_details():
    with TestClient(app) as client:
        response = client.post(
            url="/v1/ask-details",
            json={
                "query": "How many books are there?",
                "sql": "SELECT COUNT(*) FROM book",
                "summary": "Retrieve the number of books",
            },
        )

        assert response.status_code == 200
        assert response.json()["query_id"] != ""

        query_id = response.json()["query_id"]
        response = client.get(url=f"/v1/ask-details/{query_id}/result")
        while response.json()["status"] != "finished":
            response = client.get(url=f"/v1/ask-details/{query_id}/result")

        assert response.status_code == 200
        assert response.json()["status"] == "finished"
        assert response.json()["response"]["description"] != ""
        assert len(response.json()["response"]["steps"]) >= 1

        for i, step in enumerate(response.json()["response"]["steps"]):
            assert step["sql"] != ""
            assert step["summary"] != ""
            if i < len(response.json()["response"]["steps"]) - 1:
                assert step["cte_name"] != ""
            else:
                assert step["cte_name"] == ""


def test_sql_regenerations():
    with TestClient(app) as client:
        response = client.post(
            url="/v1/sql-regenerations",
            json={
                "description": "This query identifies the customer who bought the most products within a specific time frame.",
                "steps": [
                    {
                        "sql": 'SELECT * FROM "customers"',
                        "summary": "Selects all columns from the customers table to retrieve customer information.",
                        "cte_name": "customer_data",
                        "corrections": [],
                    },
                    {
                        "sql": 'SELECT * FROM "orders" WHERE "PurchaseTimestamp" >= \'2023-01-01\' AND "PurchaseTimestamp" < \'2024-01-01\'',
                        "summary": "Filters orders based on the purchase timestamp to include only orders within the specified time frame.",
                        "cte_name": "filtered_orders",
                        "corrections": [
                            {
                                "before": {
                                    "type": "filter",
                                    "value": "('PurchaseTimestamp' >= '2023-01-01') AND ('PurchaseTimestamp' < '2024-01-01')",
                                },
                                "after": {
                                    "type": "nl_expression",
                                    "value": "change the time to 2022 only",
                                },
                            }
                        ],
                    },
                    {
                        "sql": 'SELECT * FROM "order_items"',
                        "summary": "Selects all columns from the order_items table to retrieve information about the products in each order.",
                        "cte_name": "order_items_data",
                        "corrections": [],
                    },
                    {
                        "sql": """
SELECT "c"."Id", COUNT("oi"."ProductId") AS "TotalProductsBought"
FROM "customer_data" AS "c"
JOIN "filtered_orders" AS "o" ON "c"."Id" = "o"."CustomerId"
JOIN "order_items_data" AS "oi" ON "o"."OrderId" = "oi"."OrderId"
GROUP BY "c"."Id"
""",
                        "summary": "Joins customer, order, and order item data to count the total products bought by each customer.",
                        "cte_name": "product_count_per_customer",
                        "corrections": [],
                    },
                    {
                        "sql": """
SELECT "Id",
       "TotalProductsBought"
FROM "product_count_per_customer"
ORDER BY "TotalProductsBought" DESC
LIMIT 1
""",
                        "summary": "Orders the customers based on the total products bought in descending order and limits the result to the top customer.",
                        "cte_name": "",
                        "corrections": [
                            {
                                "before": {
                                    "type": "sortings",
                                    "value": "('TotalProductsBought' DESC)",
                                },
                                "after": {
                                    "type": "nl_expression",
                                    "value": "sort by 'TotalProductsBought' ASC",
                                },
                            }
                        ],
                    },
                ],
            },
        )

        assert response.status_code == 200
        assert response.json()["query_id"] != ""

        query_id = response.json()["query_id"]
        response = client.get(url=f"/v1/sql-regenerations/{query_id}/result")
        while (
            response.json()["status"] != "finished"
            and response.json()["status"] != "failed"
        ):
            response = client.get(url=f"/v1/sql-regenerations/{query_id}/result")

        assert response.status_code == 200
        assert response.json()["status"] == "finished" or "failed"


def test_web_error_handler():
    with TestClient(app) as client:
        response = client.post(
            url="/v1/asks",
            json={},
        )

        assert response.status_code == 400
        assert response.json()["detail"] != ""
