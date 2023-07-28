import json


# CustomEncoder 自定义json编码器,需要实现to_dict方法
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        return super().default(obj)


def str_to_json(json_str=str):
    # 将 JSON 字符串解析为 Python 字典
    try:
        obj = json.loads(json_str)
    except Exception as e:
        return e
    return obj
