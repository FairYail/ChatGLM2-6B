#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from base_log import llog
from fastapi import APIRouter
from vo.response import Response

health_routers = APIRouter()


@health_routers.get('/ping')
async def ping():
    llog.info("ping ...")
    return Response.success("success")


@health_routers.get('/')
async def health():
    llog.info("health...")
    return Response.success("success")


@health_routers.head('/pong')
async def pong():
    llog.info("pong...")
    return Response.success("success")
