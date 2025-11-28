import asyncio
import logging
import os
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.producers.visitor_producer import join_visitor, heartbeat as hb, leave_visitor
from api import state

router = APIRouter()

_logger = logging.getLogger("api.ws.visitors")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    _handler.setFormatter(_fmt)
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO if os.getenv("WS_DEBUG", "0") == "1" else logging.WARNING)
_logger.propagate = False


@router.websocket("/ws/visitors")
async def websocket_visitors(websocket: WebSocket):
    await websocket.accept()

    client_ip = websocket.headers.get("x-forwarded-for", websocket.client.host)
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    origin = websocket.headers.get("origin", "-")
    user_agent = websocket.headers.get("user-agent", "-")
    logger = _logger
    max_per_ip = int(os.getenv("WS_VISITORS_MAX_PER_IP", "50"))

    # Track and optionally cap per-IP connections
    async with state.ws_visitors_lock:
        current = state.active_ws_visitors_by_ip.get(client_ip, 0) + 1
        state.active_ws_visitors_by_ip[client_ip] = current
        if current > max_per_ip:
            logger.info("ws_visitors.reject ip=%s reason=per_ip_limit current=%s limit=%s origin=%s ua=%s",
                        client_ip, current, max_per_ip, origin, user_agent)
            await websocket.close(code=1008)
            # Decrement since we rejected
            state.active_ws_visitors_by_ip[client_ip] = current - 1
            return
    logger.info("ws_visitors.accept ip=%s origin=%s ua=%s active_per_ip=%s",
                client_ip, origin, user_agent, state.active_ws_visitors_by_ip.get(client_ip))

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
        # Decrement counter
        async with state.ws_visitors_lock:
            if client_ip in state.active_ws_visitors_by_ip:
                state.active_ws_visitors_by_ip[client_ip] = max(0, state.active_ws_visitors_by_ip[client_ip] - 1)
                if state.active_ws_visitors_by_ip[client_ip] == 0:
                    del state.active_ws_visitors_by_ip[client_ip]
        logger.info("ws_visitors.close ip=%s", client_ip)


