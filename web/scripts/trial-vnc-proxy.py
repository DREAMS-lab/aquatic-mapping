#!/usr/bin/env python3
"""
Dynamic Trial VNC Proxy
Routes trialX.bharathdesikan.com to localhost:608X
"""

import re
import asyncio
import aiohttp
from aiohttp import web, WSMsgType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trial-proxy")

BASE_VNC_PORT = 6080  # trial1 = 6081, trial2 = 6082, etc.

async def extract_trial_number(request):
    """Extract trial number from Host header"""
    host = request.headers.get('Host', '')
    match = re.match(r'trial(\d+)\.', host)
    if match:
        return int(match.group(1))
    return None

async def proxy_handler(request):
    """Proxy HTTP requests to the appropriate trial VNC port"""
    trial_num = await extract_trial_number(request)

    if trial_num is None:
        return web.Response(text="Invalid trial URL. Use trialX.bharathdesikan.com", status=400)

    target_port = BASE_VNC_PORT + trial_num
    target_url = f"http://localhost:{target_port}{request.path_qs}"

    logger.info(f"Proxying trial {trial_num} -> localhost:{target_port}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=target_url,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ('host', 'content-length')},
                data=await request.read() if request.body_exists else None,
                allow_redirects=False
            ) as resp:
                body = await resp.read()
                return web.Response(
                    body=body,
                    status=resp.status,
                    headers={k: v for k, v in resp.headers.items() if k.lower() not in ('content-encoding', 'transfer-encoding', 'content-length')}
                )
    except aiohttp.ClientConnectorError:
        return web.Response(
            text=f"Trial {trial_num} VNC not running (port {target_port})",
            status=503
        )

async def websocket_handler(request):
    """Proxy WebSocket connections for noVNC"""
    trial_num = await extract_trial_number(request)

    if trial_num is None:
        return web.Response(text="Invalid trial URL", status=400)

    target_port = BASE_VNC_PORT + trial_num
    target_ws_url = f"ws://localhost:{target_port}{request.path_qs}"

    logger.info(f"WebSocket proxy trial {trial_num} -> localhost:{target_port}")

    ws_client = web.WebSocketResponse()
    await ws_client.prepare(request)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(target_ws_url) as ws_server:
                async def forward_to_server():
                    async for msg in ws_client:
                        if msg.type == WSMsgType.BINARY:
                            await ws_server.send_bytes(msg.data)
                        elif msg.type == WSMsgType.TEXT:
                            await ws_server.send_str(msg.data)
                        elif msg.type == WSMsgType.CLOSE:
                            await ws_server.close()
                            break

                async def forward_to_client():
                    async for msg in ws_server:
                        if msg.type == WSMsgType.BINARY:
                            await ws_client.send_bytes(msg.data)
                        elif msg.type == WSMsgType.TEXT:
                            await ws_client.send_str(msg.data)
                        elif msg.type == WSMsgType.CLOSE:
                            await ws_client.close()
                            break

                await asyncio.gather(forward_to_server(), forward_to_client())
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    return ws_client

def create_app():
    app = web.Application()
    app.router.add_route('GET', '/websockify', websocket_handler)
    app.router.add_route('*', '/{path:.*}', proxy_handler)
    return app

if __name__ == '__main__':
    print("=" * 50)
    print("  Trial VNC Dynamic Proxy")
    print("  Listening on port 6099")
    print("  Routes trialX.domain.com -> localhost:608X")
    print("=" * 50)
    app = create_app()
    web.run_app(app, host='0.0.0.0', port=6099)
