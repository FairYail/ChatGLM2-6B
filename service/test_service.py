import json

from fastapi import Request

import torch
from transformers import AutoTokenizer
# 使用 Markdown 格式打印模型输出
from IPython.display import display, Markdown, clear_output

from base_log import llog
from consts.code_resp import Err_Embedder_Info
from consts.public_consts import commentMap
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
        count = 0
        for name, _ in commentMap:
            qE = cls.embedder.encode([name])
            cls.embeddingNameList.append(name)
            cls.embeddingList.append(qE)

    @classmethod
    def display_answer(cls, query, history=[]):
        for response, history in cls.model_2b.stream_chat(
                cls.tokenizer_2b, query, history=history):
            clear_output(wait=True)
            display(Markdown(response))
        return history

    # 检查评论类型
    @classmethod
    def check_comments(cls, json_post):
        json_post_list = json.loads(json_post)
        prompt = json_post_list.get(
            'prompt') + '''\n 你是一个游戏公司的客服，根据上面的评论语句，对这句话做出评论，你只需要回复我:积极地、正向的、不好的、普通的、中性的、负向的、不好的, 词组中的一个，你回复我三个字，不要有多余发言'''
        history = json_post_list.get('history')
        max_length = json_post_list.get('max_length')
        top_p = json_post_list.get('top_p')
        temperature = json_post_list.get('temperature')
        response, history = cls.model_2b.chat(cls.tokenizer_2b,
                                              prompt,
                                              history=history,
                                              max_length=max_length if max_length else 2048,
                                              top_p=top_p if top_p else 0.7,
                                              temperature=temperature if temperature else 0.95)
        torch_gc()
        return response

    # 向量胡匹配
    @classmethod
    def matchEmbedderQName(cls, qName):
        if cls.embedder is None:
            raise Err_Embedder_Info
        qE = cls.embedder.encode([qName])
        return semantic_search(qE, cls.embeddingList, top_k=10)

    # 使用向量模型检验最终返回值
    @classmethod
    def get_comments(cls, request: Request) -> {}:
        json_post_raw = await request.json()
        json_post = json.dumps(json_post_raw)

        # 检查评论情感类型
        resp = cls.check_comments(json_post)
        hits = cls.matchEmbedderQName(resp)

        lst = []
        # 返回结果
        for hit in hits[0]:
            score = hit['score']
            commentName = cls.embeddingNameList[hit['corpus_id']]
            commentType = commentMap.get(commentName, "UNKNOWN")
            lst.append(CommentVo(commentName, commentType, score))

        llog.info(f"请求数据：{json_post}")
        llog.info(f"AI分析：{resp}")
        # 打印结果
        for val in lst:
            llog.info(val.__dict__)
        lst = CommentVo.sort_list_by_score(lst)
        if len(lst) == 0:
            return "UNKNOWN"
        return lst[0].commentType
