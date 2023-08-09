#!/data/pyenv/customer-robot/bin/python3
import time
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import joblib

from base_log import llog


# -*- coding: utf-8 -*-
def dataParse():
    # 读取指定工作表和列的 Excel 文件
    df = pd.read_excel("./字节评论.xlsx")
    # 处理可能的 NaN 值
    df = df.fillna('')

    # 存储每一行的数据
    all_rows_data = df.values.tolist()
    csv_data = []
    for row in all_rows_data:
        single_val = [row[1]]
        temp = row[3]
        if row[3] == "POSITIVE" and len(row) > 4:
            temp = row[4]
        single_val.append(temp)
        csv_data.append(single_val)

    # 保存csv
    csv_df = pd.DataFrame(csv_data)
    # 保存为CSV文件
    csv_filename = '字节评论.csv'  # 文件名
    csv_df.to_csv(csv_filename, index=False)  # index=False 表示不保存行索引


def main():
    # 读取数据集
    data = pd.read_csv('./字节评论.csv')  # 假设数据在名为 'data.csv' 的文件中
    data.dropna(subset=['text_column'], inplace=True)  # 删除含有 NaN 值的行

    # 分割特征和标签
    X = data['text_column']  # 假设文本数据在 'text_column' 列中
    y = data['label_column']  # 假设标签在 'label_column' 列中
    # 假设 df 是包含文本数据的DataFrame

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 文本分词和特征提取
    vectorizer = CountVectorizer()  # 使用词袋模型进行特征提取
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 创建逻辑回归模型实例
    model = LogisticRegression()

    start_time = datetime.now()
    startStr = start_time.strftime('%Y-%m-%d %H:%M:%S')
    llog.info(f'训练模型,开始时间:{startStr}')

    # 训练模型
    model.fit(X_train_vec, y_train)
    end_time = datetime.now()
    endStr = end_time.strftime('%Y-%m-%d %H:%M:%S')
    llog.info(f'训练模型,结束时间:{endStr}')
    llog.info(f'训练模型,花费总时间:{(end_time - start_time).total_seconds()}')

    # 在测试集上进行预测
    y_pred = model.predict(X_test_vec)

    start_time = datetime.now()
    startStr = start_time.strftime('%Y-%m-%d %H:%M:%S')
    llog.info(f'计算准确率作为模型评估指标,开始时间:{startStr}')
    # 计算准确率作为模型评估指标
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy:", accuracy)
    endStr = end_time.strftime('%Y-%m-%d %H:%M:%S')
    llog.info(f'算准确率作为模型评估指标,结束时间:{endStr}')
    llog.info(f'算准确率作为模型评估指标,花费总时间:{(end_time - start_time).total_seconds()}')

    llog.info("开始保存模型到文件")
    # 保存模型到文件
    model_filename = 'trained_model.joblib'
    joblib.dump(model, model_filename)
    llog.info("开始结束模型到文件")

    # 加载模型
    # loaded_model = joblib.load(model_filename)

    # 使用加载的模型进行预测
    # new_data = ...  # 准备新的数据进行预测
    # new_data_vec = vectorizer.transform(new_data)  # 对新数据进行特征提取
    # predictions = loaded_model.predict(new_data_vec)
    # print("Predictions for new data:", predictions)


if __name__ == "__main__":
    main()
    # dataParse()
    # 获取当前时间
    # current_time = datetime.now()
    #
    # # 格式化为默认字符串
    # default_format = current_time.strftime('%Y-%m-%d %H:%M:%S')
    # print("current_time:", current_time)
    # print("default_format:", default_format)
