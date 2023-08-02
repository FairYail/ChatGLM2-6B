#!/data/pyenv/customer-robot/bin/python3

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 下载VADER情感分析器所需数据
nltk.download()

# 初始化情感分析器
analyzer = SentimentIntensityAnalyzer()

# 输入的语句
input_sentence = "这部电影真的很棒，我非常喜欢！"

# 进行情感分析
sentiment_score = analyzer.polarity_scores(input_sentence)
# 输出情感分析结果
print(sentiment_score)

# 情感得分中的'compound'项表示综合情感极性，可用于计算匹配度
compound_score = sentiment_score['compound']

# 假设预期的情感极性为积极（假设大于等于0.05为积极）
expected_sentiment = "positive"
if compound_score >= 0.05:
    predicted_sentiment = "positive"
elif compound_score <= -0.05:
    predicted_sentiment = "negative"
else:
    predicted_sentiment = "neutral"

# 计算匹配度
if predicted_sentiment == expected_sentiment:
    match_percentage = 100.0
else:
    match_percentage = 0.0

# 输出匹配度
print(f"匹配度：{match_percentage}%")
