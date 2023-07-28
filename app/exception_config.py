from base_log import llog

from fastapi.exceptions import RequestValidationError

from consts.code_resp import Err_System
from vo.customer_err import CustomError
from vo.response import Response
from fastapi import Request
from fastapi.responses import JSONResponse


# 所有异常捕获
async def exception_handler(request: Request, exc: Exception):
    # 打印堆栈错误信息到日志文件
    llog.error("exception_handler", exc_info=True)
    if isinstance(exc, CustomError):
        resp = Response.fail(exc.code, exc.message)
    else:
        resp = Response.fail(Err_System.code, f"{type(exc).__name__}:{str(exc)}")

    try:
        return JSONResponse(status_code=200, content=resp.__dict__)
    except Exception as e:
        llog.error("exception_handler_resp", exc_info=True)
        return JSONResponse(status_code=500, content=Err_System)


# 所有参数校验异常捕获
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 打印堆栈错误信息到日志文件
    llog.error("validation_exception", exc_info=True)

    if isinstance(exc, CustomError):
        resp = Response.fail(exc.code, exc.message)
    else:
        resp = Response.fail(Err_System.code, str(exc))

    try:
        return JSONResponse(status_code=200, content=resp.__dict__)
    except Exception as e:
        llog.error("validation_exception_resp", exc_info=True)
        return JSONResponse(status_code=500, content=Err_System)
