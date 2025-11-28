import json

import api.main as main
import api.controllers.system as system_controller


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["redis"] == "healthy"


def test_example(client):
    res = client.get("/example")
    assert res.status_code == 200
    data = res.json()
    assert data["message"] == "Hello from DevStack API!"
    assert data["status"] == "success"
    assert "timestamp" in data


def test_cache_set_and_get(client):
    key = "greeting"
    payload = {"value": "hello", "ttl": 5}
    res_set = client.post(f"/cache/{key}", json=payload)
    assert res_set.status_code == 200
    body_set = res_set.json()
    assert body_set["success"] is True
    assert body_set["key"] == key
    assert body_set["value"] == payload["value"]

    res_get = client.get(f"/cache/{key}")
    assert res_get.status_code == 200
    body_get = res_get.json()
    assert body_get["found"] is True
    assert body_get["value"] == payload["value"]


def test_visitors_empty(client):
    res = client.get("/visitors")
    assert res.status_code == 200
    data = res.json()
    assert data["active_count"] == 0
    assert isinstance(data["active_visitors"], list)
    assert isinstance(data["recent_visits"], list)


def test_system_with_mocked_subprocess(client, monkeypatch):
    class FakeCompleted:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, capture_output, text, timeout):
        if args[:2] == ["docker", "ps"]:
            return FakeCompleted(
                0, "api|Up 5 minutes|myimage:1.0\nredis|Up 10 minutes|redis:7"
            )
        if args[:2] == ["docker", "stats"]:
            return FakeCompleted(
                0,
                "api|12.3%|50.0MiB / 1.0GiB|5.0%\nredis|1.0%|20.0MiB / 1.0GiB|2.0%",
            )
        return FakeCompleted(1, "")

    monkeypatch.setattr(system_controller, "run", fake_run, raising=False)
    monkeypatch.setattr(system_controller.subprocess, "run", fake_run, raising=False)

    res = client.get("/system")
    assert res.status_code == 200
    data = res.json()
    assert data["total_containers"] == 2
    names = {s["name"] for s in data["services"]}
    assert names == {"api", "redis"}
    api_service = next(s for s in data["services"] if s["name"] == "api")
    assert api_service["cpu_percent"] == 12.3
    assert api_service["memory_mb"] == 50.0
    assert api_service["memory_percent"] == 5.0


