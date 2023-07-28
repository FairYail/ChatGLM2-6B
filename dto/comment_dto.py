import json

from pydantic import BaseModel
from dataclasses import dataclass

from consts.code_resp import Err_Param_Info


@dataclass
class CommentDto(BaseModel):
    prompt: str
    history: []
    maxLength: int
    top_p: float
    temperature: float

    # 参数校验
    def Validator(self):
        if self.prompt == "":
            raise Err_Param_Info
        if self.history is None:
            self.history = []
        if self.maxLength is None or self.maxLength == 0.0:
            self.maxLength = 2048
        if self.top_p is None or self.top_p == 0.0:
            self.top_p = 0.7
        if self.temperature is None or self.temperature == 0.0:
            self.temperature = 0.95

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "history": self.history,
        }
