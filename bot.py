#作者：Narei~
#版本：1.0-20261021
#配合server.py使用
#可修改config区

import asyncio
import json
import time
import logging
from typing import Optional

import httpx
import websockets

# ===================== config =====================
APP_ID = "你的qq机器人id"
APP_SECRET = "密钥"
LLM_API_BASE = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "local-qwen"      # 需与 FastAPI 中注册的模型名一致
LLM_MAX_TOKENS = 512
LLM_TIMEOUT = 60.0
API_BASE = "https://api.sgroup.qq.com"
#===================================================
INTENTS = (1 << 25) | (1 << 26)

RECONNECT_BASE_DELAY = 2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("qq_openapi_bot")


class QQBotAuth:
    """QQ Bot Access Token 自动刷新"""
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token: Optional[str] = None
        self.expires_at: float = 0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            # 提前 5 分钟刷新 Token
            if self.access_token and time.time() < self.expires_at - 300:
                return self.access_token

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://bots.qq.com/app/getAppAccessToken",
                    json={
                        "appId": self.app_id,
                        "clientSecret": self.app_secret,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            self.access_token = data["access_token"]
            self.expires_at = time.time() + int(data["expires_in"])
            logger.info("已刷新  Access Token")
            return self.access_token


class LLMClient:
    """轻量级 LLM 客户端"""
    def __init__(
        self,
        api_base: str = LLM_API_BASE,
        model: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: float = LLM_TIMEOUT,
    ):
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.client = httpx.AsyncClient(timeout=timeout)

    async def chat(self, content: str) -> str:

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            resp = await self.client.post(self.api_base, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("LLM 请求失败: %s", exc)
            return f"[AI 服务异常: {type(exc).__name__}]"


class QQBotOpenAPI:
    """QQ Bot OpenAPI WebSocket 网关与 REST 消息发送器"""
    def __init__(self, auth: QQBotAuth):
        self.auth = auth
        self.http = httpx.AsyncClient()
        self._session_id: Optional[str] = None

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def send_group_message(self, group_openid: str, content: str, msg_id: str) -> dict:
        token = await self.auth.get_token()
        url = f"{API_BASE}/v2/groups/{group_openid}/messages"
        resp = await self.http.post(
            url, headers=self._headers(token),
            json={"content": content, "msg_type": 0, "msg_id": msg_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def send_c2c_message(self, user_openid: str, content: str, msg_id: str) -> dict:
        token = await self.auth.get_token()
        url = f"{API_BASE}/v2/users/{user_openid}/messages"
        resp = await self.http.post(
            url, headers=self._headers(token),
            json={"content": content, "msg_type": 0, "msg_id": msg_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_gateway(self) -> dict:
        token = await self.auth.get_token()
        resp = await self.http.get(
            f"{API_BASE}/gateway/bot",
            headers={"Authorization": f"QQBot {token}"},
        )
        resp.raise_for_status()
        return resp.json()

    async def run(self, llm: LLMClient):
        gateway_info = await self.get_gateway()
        wss_url = gateway_info["url"]
        logger.info("获取到网关地址: %s", wss_url)

        while True:
            try:
                await self._connect(wss_url, llm)
            except Exception as exc:
                logger.error("WebSocket 异常，准备重连: %s", exc)
                await asyncio.sleep(RECONNECT_BASE_DELAY)

    async def _connect(self, wss_url: str, llm: LLMClient):
        async with websockets.connect(wss_url) as ws:
            await self._identify(ws)
            heartbeat_task: Optional[asyncio.Task] = None

            async for raw in ws:
                event = json.loads(raw)
                op = event.get("op")

                if op == 10:  # Hello
                    interval = event.get("d", {}).get("heartbeat_interval", 45000)
                    logger.info("收到 Hello，心跳间隔: %d ms", interval)
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws, interval))
                    continue

                if op == 11:  # Heartbeat ACK
                    continue

                if op == 0:  # Dispatch
                    session_id = event.get("s")
                    if session_id:
                        self._session_id = str(session_id)

                    event_type = event.get("t")
                    data = event.get("d", {})

                    if event_type == "GROUP_AT_MESSAGE_CREATE":
                        await self._handle_group_message(data, llm)
                    elif event_type == "C2C_MESSAGE_CREATE":
                        await self._handle_c2c_message(data, llm)
                    else:
                        logger.debug("收到未处理事件: %s", event_type)

                elif op == 7:  # Reconnect
                    logger.warning("收到重连指令，将重新 Identify")
                    await self._identify(ws)

                elif op == 9:  # Invalid Session
                    logger.error("Identify 无效，将断开并重连")
                    break

    async def _identify(self, ws: websockets.ClientConnection):
        token = await self.auth.get_token()
        identify = {
            "op": 2,
            "d": {
                "token": f"QQBot {token}",
                "intents": INTENTS,
                "shard": [0, 1],
                "properties": {
                    "$os": "linux",
                    "$browser": "qq_openapi_bot",
                    "$device": "qq_openapi_bot",
                },
            },
        }
        await ws.send(json.dumps(identify))
        logger.info("已发送 Identify，订阅 intents=%d", INTENTS)

    async def _heartbeat_loop(self, ws: websockets.ClientConnection, interval_ms: int):
        interval = interval_ms / 1000
        while True:
            await asyncio.sleep(interval)
            try:
                await ws.send(json.dumps({"op": 1, "d": self._session_id}))
            except Exception:
                break

    async def _handle_group_message(self, data: dict, llm: LLMClient):
        group_openid = data.get("group_openid")
        msg_id = data.get("id")
        content = (data.get("content") or "").strip()

        if not all([group_openid, msg_id, content]):
            return

        logger.info("群聊 @ 消息 [%s]: %s", group_openid, content)
        reply = await llm.chat(content)

        try:
            await self.send_group_message(group_openid, reply, msg_id)
            logger.info("已回复群聊消息")
        except Exception as exc:
            logger.error("回复群聊消息失败: %s", exc)

    async def _handle_c2c_message(self, data: dict, llm: LLMClient):
        author = data.get("author") or {}
        user_openid = author.get("user_openid")
        msg_id = data.get("id")
        content = (data.get("content") or "").strip()

        if not all([user_openid, msg_id, content]):
            return

        logger.info("私聊消息 [%s]: %s", user_openid, content)
        reply = await llm.chat(content)

        try:
            await self.send_c2c_message(user_openid, reply, msg_id)
            logger.info("已回复私聊消息")
        except Exception as exc:
            logger.error("回复私聊消息失败: %s", exc)


async def main():
    auth = QQBotAuth(APP_ID, APP_SECRET)
    llm = LLMClient()  # 不再传入 system_prompt
    bot = QQBotOpenAPI(auth)
    await bot.run(llm)


if __name__ == "__main__":
    asyncio.run(main())
