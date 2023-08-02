#!/data/pyenv/customer-robot/bin/python3

# -*- coding: utf-8 -*-
import argparse
import os
import time

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.config import Config
from app.exception_config import exception_handler, validation_exception_handler
from app.middlewares import RequestLoggerMiddleware
from base_log import llog
from router.test_router import test_routers
from router.health_router import health_routers
from service.test_service import TestService

# This example requires environment variables named "LANGUAGE_KEY" and "LANGUAGE_ENDPOINT"
language_key = os.environ.get('LANGUAGE_KEY')
language_endpoint = os.environ.get('LANGUAGE_ENDPOINT')

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


# Authenticate the client using your key and endpoint
def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=language_endpoint,
        credential=ta_credential)
    return text_analytics_client


client = authenticate_client()


# Example method for detecting sentiment and opinions in text
def sentiment_analysis_with_opinion_mining_example(client):
    documents = [
        "祝贵公司的前程也能和视频一样 ，抓住所有的坏机遇，而且颗粒无收[玫瑰][玫瑰][玫瑰][比心][比心][比心]同时也希望贵公司的前程也和视频一样，在离成功一步之遥的时候功亏一篑，没用的东西",
        "真是飒必啊[鼓掌][鼓掌][赞][赞]趣赤石吧[鼓掌][鼓掌][赞][赞][赞][赞]",
        "策划一天一个还不错来",
        "已购，穿着很好看用着和真人一样，敏感肌也能用，隔壁小孩也爱吃，下次会回购[比心]",
        "祝贵公司的前程也能和视频一样 ，抓住所有的坏机遇，而且颗粒无收[玫瑰][玫瑰][玫瑰][比心][比心][比心]同时也希望贵公司的前程也和视频一样，在离成功一步之遥的时候功亏一篑，没用的东西",
        "把咸鱼之王开发组全部宰了喂鱼 做的游戏跟坨屎一样 [感谢][感谢][感谢][小鼓掌][小鼓掌][小鼓掌]",
        "希望贵公司的前程也能和视频一样 ，完美避开所有的好机遇，年底同样颗粒无收而公司破产，加油，没用的东西[玫瑰][玫瑰][玫瑰][比心][比心][比心][比心]",
        "每次看到这种视频我都感慨，国内做游戏的这帮人真的不知道游戏最重要的是好玩",
        "祝愿贵司经营前途也似你们视频一样途中充满坎坷 回首看待每次机遇尽是错过 结局充斥灰色与失败[感谢][感谢][感谢]",
        "祝贵公司的前程也能和视频一样 ，抓住所有的坏机遇，而且颗粒无收[玫瑰][玫瑰][玫瑰][比心][比心][比心]同时也希望贵公司的前程也和视频一样，在离成功一步之遥的时候功亏一篑，没用的东西",
        "祝贵公司的前程也能和视频一样 ，抓住所有的坏机遇，而且颗粒无收[玫瑰][玫瑰][玫瑰][比心][比心][比心]同时也希望贵公司的前程也和视频一样，在离成功一步之遥的时候功亏一篑，没用的东西",
        "[666][666][666]",
        "帅哥，我拿策划老婆换你手中的西瓜，可以不[感谢][感谢][感谢]",
        "好看 举报了",
        "这丝袜好看[看]",
        "好多版本[比心][比心][比心][送心]越来越疯了",
        "祝贵公司的前程也能和视频一样 ，抓住所有的坏机遇，而且颗粒无收[玫瑰][玫瑰][玫瑰][比心][比心][比心]同时也希望贵公司的前程也和视频一样，在离成功一步之遥的时候功亏一篑，没用的东西",
        "咸鱼之王你花这些功夫拍电影，绝对能得奥斯卡",
        "@y 我是土狗我爱玩",
        "@GGZZ 笑死我了",
        "我只想说，Bgm好评",
        "贺知章一样",
        "顶多B[捂脸][捂脸]",
        "玉足[绿帽子]",
        "搞笑",
        "我只想说，Bgm好评",
        "这广告真服了",
        "买了孩子很爱喝[赞]",
        "你就值这点",
        "美女挺好看",
        "美女挺好看",
        "@黑夜没有猩：祝贵公司的前程也能和视频一样 ，抓住所有的坏机遇，而且颗粒无收[玫瑰][玫瑰][玫瑰][比心][比心][比心]同时也希望贵公司的前程也和视频一样，在离成功一步之遥的时候功亏一篑，没用的东西",
        "就算难得来一个正常的，我还是不玩[看]",
        "已到货，很好用，跟真人一样[害羞]",
        "你们的广告做的，真的是跟💩一样啊，厉害厉害",
        "真是大聪明",
        "儿子加油",
        "已下单，咸鱼很好吃[舔屏]",
        "加油没用的东西[送心]",
        "这美女可以",
        "这个是d，我吃下去[感谢]",
        "好看爱看[色]",
        "我照样吃",
        "穿了两天，一点不臭脚",
        "谢谢你[白眼]",
        "爱玩[送心]",
        "@wenzezeze 好看爱看",
        "真的啊，我都在玩了",
        "看挺仔细[看]",
        "瓜子不错挺好吃[看]",
        "好玩",
        "其实挺好玩的[看]",
        "表情已拿[看]给你点赞",
        "支持[呲牙]",
        "哈哈哈",
        "见一次骂一次！策划你妹妹[鼓掌]",
        "好吧，我十天了400多万战斗力",
        "未下单期待发货[送心]",
        "方案给你:屏蔽你就好啦[舔屏][舔屏]",
        "这游戏很不错，我不玩",
        "我下载了 每天都玩",
    ]

    result = client.analyze_sentiment(documents, show_opinion_mining=True)
    doc_result = [doc for doc in result if not doc.is_error]

    positive_reviews = [doc for doc in doc_result if doc.sentiment == "positive"]
    negative_reviews = [doc for doc in doc_result if doc.sentiment == "negative"]

    positive_mined_opinions = []
    mixed_mined_opinions = []
    negative_mined_opinions = []

    for document in doc_result:
        print("Document Sentiment: {}".format(document.sentiment))
        print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
            document.confidence_scores.positive,
            document.confidence_scores.neutral,
            document.confidence_scores.negative,
        ))
        for sentence in document.sentences:
            print("Sentence: {}".format(sentence.text))
            print("Sentence sentiment: {}".format(sentence.sentiment))
            print("Sentence score:\nPositive={0:.10f}\nNeutral={1:.10f}\nNegative={2:.10f}\n".format(
                sentence.confidence_scores.positive,
                sentence.confidence_scores.neutral,
                sentence.confidence_scores.negative,
            ))
            for mined_opinion in sentence.mined_opinions:
                target = mined_opinion.target
                print("......'{}' target '{}'".format(target.sentiment, target.text))
                print("......Target score:\n......Positive={0:.2f}\n......Negative={1:.2f}\n".format(
                    target.confidence_scores.positive,
                    target.confidence_scores.negative,
                ))
                for assessment in mined_opinion.assessments:
                    print("......'{}' assessment '{}'".format(assessment.sentiment, assessment.text))
                    print("......Assessment score:\n......Positive={0:.2f}\n......Negative={1:.2f}\n".format(
                        assessment.confidence_scores.positive,
                        assessment.confidence_scores.negative,
                    ))
            print("\n")
        print("\n")


sentiment_analysis_with_opinion_mining_example(client)
