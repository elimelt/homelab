import json
import time
import asyncio

import api.main as main


def test_websocket_join_broadcast_and_leave(client, monkeypatch):
    headers = {"x-forwarded-for": "203.0.113.10"}

    class DummyPubSub:
        def __init__(self, data_text: str):
            self._data_text = data_text

        async def subscribe(self, _channel: str):
            return True

        async def unsubscribe(self, _channel: str):
            return True

        async def close(self):
            return True

        async def aclose(self):
            return True

        async def listen(self):
            yield {"type": "message", "data": self._data_text}
            await asyncio.sleep(0)  # allow cancellation

    payload = {"type": "broadcast", "payload": {"hello": "world"}}
    monkeypatch.setattr(
        main.redis_client, "pubsub", lambda: DummyPubSub(json.dumps(payload))
    )

    with client.websocket_connect("/ws/visitors", headers=headers) as ws:
        # After connect, visitor should appear in /visitors
        visitors_resp = client.get("/visitors")
        assert visitors_resp.status_code == 200
        visitors = visitors_resp.json()
        assert visitors["active_count"] >= 1
        ips = [v["ip"] for v in visitors["active_visitors"]]
        assert "203.0.113.10" in ips

        # Websocket should receive broadcast forwarded by DummyPubSub.listen()
        msg = ws.receive_text()
        received = json.loads(msg)
        assert received["type"] == "broadcast"
        assert received["payload"]["hello"] == "world"

    # After disconnect, visitor key should be cleaned up shortly
    for _ in range(10):
        visitors_resp = client.get("/visitors")
        assert visitors_resp.status_code == 200
        if visitors_resp.json()["active_count"] == 0:
            break
        time.sleep(0.05)
    assert visitors_resp.json()["active_count"] == 0


