from pydantic import BaseModel

from consts.code_resp import Err_Param_Info


class CommentDto(BaseModel):
    prompt: str
    history: list
    maxLength: int
    top_p: float
    temperature: float

    # 参数校验
    def Validator(self):
        if self.prompt == "":
            raise Err_Param_Info
        if self.maxLength == 0.0:
            self.maxLength = 2048
        if self.top_p == 0.0:
            self.top_p = 0.7
        if self.temperature == 0.0:
            self.temperature = 0.95

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "history": self.history,
            "maxLength": self.maxLength,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
