class CustomError(Exception):
    def __init__(self, err_code: int, err_msg: str):
        self.code = err_code
        self.message = err_msg
        super().__init__(self.message)

    def ParseErrInfo(self, info=str):
        self.message = self.message.format(info)
        return self

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message
        }
