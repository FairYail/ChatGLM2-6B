from fastapi import APIRouter, Request
from service.test_service import TestService
from service.text_consts import prompt
from vo.response import Response

test_routers = APIRouter()


@test_routers.get('/GetAnswer/{key}')
async def GetAnswer(key: int):
    # msg = "想要增强大语言模型的上下文窗口，可以使用哪些技术手段？"
    # if key == 1:
    msg = prompt + "\n 根据这篇文章内容，请回答我的问题：想要增强大语言模型的上下文窗口，可以使用哪些技术手段？"
    return Response.success(TestService().display_answer(msg))


@test_routers.get('/GetCommentType')
async def GetAnswer(request: Request):
    return Response.success(TestService().get_comments(request))
