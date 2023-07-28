import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer
# 使用 Markdown 格式打印模型输出
from IPython.display import display, Markdown, clear_output

from base_log import llog
from consts.code_resp import Err_Embedder_Info
from consts.public_consts import commentMap
from dto.comment_dto import CommentDto
from utils import load_model_on_gpus
from text2vec import SentenceModel, semantic_search

from vo.comment_vo import CommentVo

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class TestService:
    _instance = None
    _initialized = False

    # 大语言模型
    model_2b = None
    tokenizer_2b = None

    # 文本向量化匹配
    embedder = None
    embeddingList = []
    embeddingNameList = []

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
        # 加载大预言模型模型
        model_path = "/data/chatglm2-6b"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = load_model_on_gpus(model_path, num_gpus=4)
        model = model.eval()
        cls.tokenizer_2b = tokenizer
        cls.model_2b = model

        # 加载向量匹配模型
        cls.embedder = SentenceModel(
            model_name_or_path="/data/embedding-model/text2vec-large-chinese",
            device="cuda"
        )

        # 加载向量化数据信息
        eList = []
        for name in commentMap:
            qE = cls.embedder.encode([name])
            cls.embeddingNameList.append(name)
            eList.extend(np.array(qE, dtype=np.float32))
        cls.embeddingList = np.array(eList)

    @classmethod
    def display_answer(cls, query, history=[]):
        for response, history in cls.model_2b.stream_chat(
                cls.tokenizer_2b, query, history=history):
            clear_output(wait=True)
            display(Markdown(response))
        return history

    # 检查评论类型
    @classmethod
    def check_comments(cls, param: CommentDto):
        prompt = '''你是一个游戏公司的客服，后面会给你发一些语句，你需要做出一些判断 \n''' + param.prompt + '''\n这一句话是什么情感方向的言论。你以下三个选项：正能量的、中性的、负能量的，不要有多余发言'''

        history = gr.State([])
        past_key_values = gr.State(None)
        max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
        top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        response, history = cls.model_2b.chat(cls.tokenizer_2b,
                                              prompt,
                                              history=history,
                                              past_key_values=past_key_values,
                                              max_length=max_length,
                                              top_p=top_p,
                                              temperature=temperature)
        torch_gc()
        return response

    # 向量胡匹配
    @classmethod
    def matchEmbedderQName(cls, qName):
        if cls.embedder is None:
            raise Err_Embedder_Info
        qE = cls.embedder.encode([qName])
        llog.info(f"向量化数据：{len(cls.embeddingList)}")
        return semantic_search(qE, cls.embeddingList, top_k=10)

    # 使用向量模型检验最终返回值
    def get_comments(self, param=CommentDto):
        llog.info(f"请求数据：{param.to_dict()}")
        # 参数校验
        param.Validator()
        # 检查评论情感类型
        resp = self.check_comments(param)
        hits = self.matchEmbedderQName(resp)

        lst = []
        # 返回结果
        for hit in hits[0]:
            score = hit['score']
            commentName = self.embeddingNameList[hit['corpus_id']]
            commentType = commentMap.get(commentName, "UNKNOWN")
            lst.append(CommentVo(commentName, commentType, score))

        llog.info(f"AI分析：{resp}")
        # 打印结果
        for val in lst:
            llog.info(val.__dict__)
        lst = CommentVo.sort_list_by_score(lst)
        if len(lst) == 0:
            return "UNKNOWN"
        return lst[0].commentType
