"""Gestor de eventos en tiempo real (SSE y WebSockets) para el panel web."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class RealtimeHub:
    def __init__(self) -> None:
        self._sse_subscribers: set[asyncio.Queue[str]] = set()
        self._ws_clients: set[WebSocket] = set()

    def subscribe_sse(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._sse_subscribers.add(q)
        return q

    def unsubscribe_sse(self, q: asyncio.Queue[str]) -> None:
        self._sse_subscribers.discard(q)

    async def register_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        self._ws_clients.add(ws)

    def unregister_ws(self, ws: WebSocket) -> None:
        self._ws_clients.discard(ws)

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        payload = json.dumps({"event": event_type, "data": data}, ensure_ascii=False, default=str)

        for q in list(self._sse_subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

        dead_ws = []
        for ws in list(self._ws_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead_ws.append(ws)
        for ws in dead_ws:
            self._ws_clients.discard(ws)


hub = RealtimeHub()
