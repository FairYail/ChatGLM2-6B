# 定义问题结构体
class CommentVo:
    def __init__(self, commentName: str, commentType: str, score: float):
        self._commentName = commentName
        self._commentType = commentType
        self._score = score

    @property
    def commentName(self):
        return self._commentName

    @commentName.setter
    def commentName(self, value):
        self._commentName = value

    @property
    def commentType(self):
        return self._commentType

    @commentType.setter
    def commentType(self, value):
        self._commentType = value

    @property
    def score(self):
        return round(self._score, 4)

    @score.setter
    def score(self, value):
        self._score = round(value, 4)

    @staticmethod
    def sort_list_by_score(lst):
        return sorted(lst, key=lambda x: x.score, reverse=True)

    def to_dict(self):
        return {
            "commentName": self._commentName,
            "commentType": self._commentType,
            "score": self._score
        }
