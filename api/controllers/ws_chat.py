import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.producers.chat_producer import build_chat_message, publish_chat_message
from api.bus import EventBus
from api import state

router = APIRouter()


@router.websocket("/ws/chat/{channel}")
async def websocket_chat(websocket: WebSocket, channel: str):
    await websocket.accept()

    client_ip = websocket.headers.get("x-forwarded-for", websocket.client.host)
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()
    sender = f"{client_ip}:{id(websocket)}"

    pubsub = state.redis_client.pubsub()
    await pubsub.subscribe(EventBus.chat_channel(channel))

    async def send_updates():
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await websocket.send_text(message["data"])
        except Exception:
            pass

    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(25)
                await websocket.send_text(json.dumps({"type": "ping"}))
        except Exception:
            pass

    update_task = asyncio.create_task(send_updates())
    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                text = payload.get("text")
                if not text:
                    continue
            except Exception:
                continue
            event = build_chat_message(channel=channel, sender=sender, text=text)
            await publish_chat_message(state.event_bus, channel, event)
    except WebSocketDisconnect:
        pass
    finally:
        update_task.cancel()
        heartbeat_task.cancel()
        await pubsub.unsubscribe(EventBus.chat_channel(channel))
        if hasattr(pubsub, "aclose"):
            await pubsub.aclose()
        else:
            await pubsub.close()


