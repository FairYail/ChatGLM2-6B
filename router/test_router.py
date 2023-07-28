from fastapi import APIRouter

from service.test_service import TestService
from service.text_consts import prompt
from vo.response import Response

test_routers = APIRouter()


# 获取答案
@test_routers.get('/GetAnswer/{id}')
async def GetAnswer(id: int):
    msg = "想要增强大语言模型的上下文窗口，可以使用哪些技术手段？"
    if id == 1:
        msg = prompt + "\n" + msg
    return Response.success(TestService().display_answer(msg))
