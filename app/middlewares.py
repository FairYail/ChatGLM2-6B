import json
import time

from fastapi import Request
from starlette.types import Message

from base_log import llog
from utils.otlp import global_tracer


# 日志中间件
class RequestLoggerMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # 打印请求信息到日志
        ownReq = Request(scope, receive)

        llog.info(f"Received request: {ownReq.method} {ownReq.url}")

        with global_tracer.start_as_current_span("request middleware") as span:
            body = await ownReq.body()
            span.add_event("request", {"body": body})

            async def receive() -> Message:
                return {"type": "http.request", "body": body}

            start_time = time.perf_counter()  # 记录请求开始时间

            async def custom_send(message):
                if message["type"] == "http.response.start":
                    # 打印响应状态码到日志
                    llog.info(f"Received request: {ownReq.method} {ownReq.url} "
                              f"status code: {message['status']} "
                              f"time: {time.perf_counter() - start_time:.6f} seconds")
                await send(message)

            # 继续处理请求
            ownResp = await self.app(scope, receive, custom_send)
            return ownResp
