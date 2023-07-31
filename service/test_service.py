import numpy as np
import torch
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
        model = load_model_on_gpus(model_path, num_gpus=1)
        model = model.eval()
        cls.tokenizer_2b = tokenizer
        cls.model_2b = model

        # 加载向量匹配模型
        # cls.embedder = SentenceModel(
        #     model_name_or_path="/data/embedding-model/text2vec-large-chinese",
        #     device="cuda"
        # )
        #
        # # 加载向量化数据信息
        # eList = []
        # for name in commentMap:
        #     qE = cls.embedder.encode([name])
        #     cls.embeddingNameList.append(name)
        #     eList.extend(np.array(qE, dtype=np.float32))
        # cls.embeddingList = np.array(eList)

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
        prompt = '''你是一个游戏公司的客服，请对以下语句进行等级评分，评分范围对应着1到10分，只需要回答多少分，不要提供额外回答。该语句是：\n
        ''' + param.prompt
        response, history = cls.model_2b.chat(cls.tokenizer_2b,
                                                     prompt,
                                                     history=[],
                                                     max_length=2048,
                                                     top_p=0.7,
                                                     temperature=0.4)
        # torch_gc()
        llog.info(f"prompt：{prompt}")
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
        return resp
        # hits = self.matchEmbedderQName(resp)

        # lst = []
        # # 返回结果
        # for hit in hits[0]:
        #     score = hit['score']
        #     commentName = self.embeddingNameList[hit['corpus_id']]
        #     commentType = commentMap.get(commentName, "UNKNOWN")
        #     lst.append(CommentVo(commentName, commentType, score))
        #
        # llog.info(f"AI分析：{resp}")
        # # 打印结果
        # for val in lst:
        #     llog.info(val.__dict__)
        # lst = CommentVo.sort_list_by_score(lst)
        # if len(lst) == 0:
        #     return "UNKNOWN"
        # return lst[0].commentType

        # 检查评论类型

    @classmethod
    def check_comments_type_dg(cls, param: CommentDto):
        prompt = '''你的角色是是游戏公司客服人员。请问下面的语句属于其中哪一种(只需要回答选项，不需要说其他信息):：\n
         ''' + param.prompt + '''A、玩法咨询\n B、注销账号\n C、充值未到账\n D、功能异常\n E、玩法吐槽\n F、误触找回\n G、未成年人退款'''
        response, history = cls.model_2b.chat(cls.tokenizer_2b,
                                              prompt,
                                              history=[],
                                              max_length=2048,
                                              top_p=0.5,
                                              temperature=0.3)
        torch_gc()
        llog.info(f"prompt：{prompt}")
        return response

    # 向量胡匹配
    @classmethod
    def matchEmbedderQName(cls, qName):
        if cls.embedder is None:
            raise Err_Embedder_Info
        qE = cls.embedder.encode([qName])
        llog.info(f"向量化数据：{len(cls.embeddingList)}")
        return semantic_search(qE, cls.embeddingList, top_k=10)
