from transformers import AutoTokenizer, AutoModel
# 使用 Markdown 格式打印模型输出
from IPython.display import display, Markdown, clear_output

from utils import load_model_on_gpus


class TestService:
    _instance = None
    _initialized = False
    model_2b = None
    tokenizer_2b = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.init_model()

    # 加载模型
    @classmethod
    def init_model(cls):
        # 加载模型
        model_path = "/data/chatglm2-6b"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = load_model_on_gpus(model_path, num_gpus=4)
        model = model.eval()
        cls.tokenizer_2b = tokenizer
        cls.model_2b = model

    @classmethod
    def display_answer(cls, query, history=[]):
        for response, history in cls.model_2b.stream_chat(
                cls.tokenizer_2b, query, history=history):
            clear_output(wait=True)
            display(Markdown(response))
        return history
