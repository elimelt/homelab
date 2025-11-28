import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.producers.visitor_producer import join_visitor, heartbeat as hb, leave_visitor
from api import state

router = APIRouter()


@router.websocket("/ws/visitors")
async def websocket_visitors(websocket: WebSocket):
    await websocket.accept()

    client_ip = websocket.headers.get("x-forwarded-for", websocket.client.host)
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    visitor_id = f"visitor:{client_ip}:{id(websocket)}"

    visitor_data = await join_visitor(state.redis_client, state.event_bus, state.geoip_reader, client_ip, visitor_id)

    pubsub = state.redis_client.pubsub()
    await pubsub.subscribe("visitor_updates")

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
                await asyncio.sleep(10)
                await hb(state.redis_client, visitor_id, visitor_data)
                await websocket.send_text(json.dumps({"type": "ping"}))
        except Exception:
            pass

    update_task = asyncio.create_task(send_updates())
    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            data = await websocket.receive_text()
            if data == "pong":
                continue
    except WebSocketDisconnect:
        pass
    finally:
        update_task.cancel()
        heartbeat_task.cancel()
        await pubsub.unsubscribe("visitor_updates")
        if hasattr(pubsub, "aclose"):
            await pubsub.aclose()
        else:
            await pubsub.close()
        await leave_visitor(state.redis_client, state.event_bus, client_ip, visitor_id)


