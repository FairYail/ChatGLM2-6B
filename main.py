#!/data/pyenv/customer-robot/bin/python3

# -*- coding: utf-8 -*-
import argparse
import time

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.config import Config
from app.exception_config import exception_handler, validation_exception_handler
from app.middlewares import RequestLoggerMiddleware
from base_log import llog
from router.test_router import test_routers
from router.health_router import health_routers
from service.test_service import TestService


# 注册关闭事件信息
def on_shutdown():
    llog.info("on_shutdown ...")
    Config().close_db()


def start_app():
    # 初始化 FastAPI 应用
    app = FastAPI()

    # 允许跨域请求的来源
    origins = [
        "http://43.159.41.241/",
    ]

    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_event_handler("shutdown", on_shutdown)
    # 添加http中间件
    app.add_middleware(RequestLoggerMiddleware)

    # 添加异常处理
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, exception_handler)

    # 注册路由方法
    app.include_router(health_routers)
    app.include_router(test_routers, prefix="/test")
    llog.info("server注册路由方法")

    # 获取所有路由信息
    # 打印路由信息
    for route in app.routes:
        llog.info(f"Method: {route.methods}，Path: {route.path}")
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="运行环境，dev，test，prod", default="local")
    args = parser.parse_args()
    envParam = args.env
    llog.info(f"server项目启动环境:{str(args.env)}")
    # 读取启动参数，初始化配置文件
    # Config().Prepare(envParam)
    llog.info("配置文件初始化完成")

    # 初始化service
    TestService()

    # 启动应用
    startApp = start_app()
    llog.info("server start...")

    # 启动应用
    uvicorn.run(app=startApp, host="", port=19396)


if __name__ == "__main__":
    main()
