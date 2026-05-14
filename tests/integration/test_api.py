"""Integration tests for the FastAPI semantic router service."""

from __future__ import annotations


def test_post_routes_creates_route_and_get_routes_returns_it(api_client) -> None:
    response = api_client.post(
        "/routes",
        json={
            "name": "returns",
            "utterances": ["start a return", "refund my order"],
            "description": "Returns and refunds",
            "threshold": 0.8,
            "metadata": {"team": "support"},
        },
    )
    assert response.status_code == 201

    list_response = api_client.get("/routes")

    assert list_response.status_code == 200
    names = [item["name"] for item in list_response.json()]
    assert "returns" in names


def test_post_route_with_known_query_returns_correct_match(api_client) -> None:
    response = api_client.post(
        "/route",
        json={"query": "play some jazz tonight", "include_scores": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] == "music"
    assert payload["all_scores"]["music"] >= payload["all_scores"]["travel"]


def test_post_route_with_gibberish_returns_null_match(api_client) -> None:
    response = api_client.post(
        "/route",
        json={"query": "zqxj blorb snazzle", "include_scores": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] is None
    assert payload["score"] is None


def test_delete_route_then_get_returns_404(api_client) -> None:
    create_response = api_client.post(
        "/routes",
        json={"name": "shipping", "utterances": ["track my package"], "metadata": {}},
    )
    assert create_response.status_code == 201

    delete_response = api_client.delete("/routes/shipping")
    get_response = api_client.get("/routes/shipping")

    assert delete_response.status_code == 200
    assert get_response.status_code == 404


def test_post_batch_route_returns_list_of_correct_length(api_client) -> None:
    response = api_client.post(
        "/batch-route",
        json={
            "queries": ["book a flight", "what is the weather", "unknown blorb text"],
            "include_scores": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 3
