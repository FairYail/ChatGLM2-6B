from pydantic import BaseModel

from consts.code_resp import Err_Param_Info


class CommentDto(BaseModel):
    prompt: str

    # 参数校验
    def Validator(self):
        if self.prompt == "":
            raise Err_Param_Info

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "history": self.history,
        }
