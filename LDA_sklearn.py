import os
import re
import jieba
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import warnings
from sklearn.exceptions import FitFailedWarning

# 忽略交叉验证的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# 定义参数
UNITS = ["word", "char"]
K_VALUES = [20, 100, 500, 1000, 3000]  # 段落长度
T_VALUES = [5, 10, 15, 20]  # 主题数量
CORPUS_DIR = "D:\语料库"
STOPWORDS_PATH = "D:\cn_stopwords.txt"
N_SAMPLES = 1000  # 抽取的段落数量
CV_FOLDS = 10  # 交叉验证折数


# 加载停用词
def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f])
    return stopwords


# 构建数据集
def build_dataset(corpus_dir, k, stopwords, n_samples, unit):

    paragraphs = []
    labels = []
    # 清除\n -> 清除非中文词 -> 清除停用词 -> 分词 -> 构建数据集

    for novel_file in os.listdir(corpus_dir):
        if not novel_file.endswith(".txt") or novel_file == "inf.txt":
            continue
        with open(os.path.join(corpus_dir, novel_file), "r", encoding="gb18030") as f:
            text = f.read().replace("\n", "")

            text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
            if unit == "word":
                tokens = list(
                    filter(lambda char: char not in stopwords, jieba.lcut(text))
                )
            elif unit == "char":
                tokens = list(filter(lambda char: char not in stopwords, list(text)))
            current_paragraphs = [
                tokens[i * k : (i + 1) * k] for i in range(len(tokens) // k)
            ]
            paragraphs += current_paragraphs
            labels += [novel_file] * len(current_paragraphs)  # 用当前生成段落数生成标签

        # randomly sample paragraphs and labels

    if n_samples > len(paragraphs):
        n_samples = len(paragraphs)  # 或抛出 ValueError
    selected_paragraphs = random.sample(list(zip(paragraphs, labels)), n_samples)
    sampled_paragraphs, sampled_labels = zip(*selected_paragraphs)  # 解压为独立列表

    paragraphs = list(sampled_paragraphs)
    labels = list(sampled_labels)
    return paragraphs, labels


# 主函数
def main():
    print("加载停用词...")
    stopwords = load_stopwords(STOPWORDS_PATH)
    results = []

    for unit in UNITS:
        print(f"\n分词单位: {unit}")

        for k in K_VALUES:
            print(f"\n段落长度 K = {k}")
            paragraphs, labels = build_dataset(
                CORPUS_DIR, k, stopwords, N_SAMPLES, unit
            )
            paragraphs = [
                "".join(paragraph) if unit == "char" else " ".join(paragraph)
                for paragraph in paragraphs
            ]
            print(f"总段落数量: {len(paragraphs)}, 总标签数量: {len(labels)}")
            for t in T_VALUES:
                print(f"主题数量 T = {t}")

                # 创建特征提取器
                vectorizer = CountVectorizer(max_df=0.95, min_df=2)

                # 创建并训练LDA模型
                lda = LatentDirichletAllocation(
                    n_components=t,
                    max_iter=50,
                    learning_method="online",
                    random_state=0,
                )

                # 创建分类模型
                classifier = RandomForestClassifier(n_estimators=100, random_state=0)

                # 构建pipeline
                # pipeline = Pipeline([("vectorizer", vectorizer), ("lda", lda)])

                # 提取特征
                # X = pipeline.fit_transform(paragraphs)

                # 使用10折分层交叉验证
                skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
                accuracies = []
                pipeline = Pipeline(
                    [
                        ("vectorizer", vectorizer),
                        ("lda", lda),
                        ("classifier", classifier),  # 整合分类器
                    ]
                )
                for train_index, test_index in skf.split(
                    paragraphs, labels
                ):  # 传入原始文本数据
                    X_train = [paragraphs[i] for i in train_index]
                    X_test = [paragraphs[i] for i in test_index]
                    y_train = [labels[i] for i in train_index]
                    y_test = [labels[i] for i in test_index]
                    pipeline.fit(X_train, y_train)  # 每次仅训练训练集
                    y_pred = pipeline.predict(X_test)

                    # for train_index, test_index in skf.split(X, labels):
                    #     X_train, X_test = X[train_index], X[test_index]
                    #     y_train, y_test = [labels[i] for i in train_index], [
                    #         labels[i] for i in test_index
                    #     ]

                    #     # 训练分类器
                    #     classifier.fit(X_train, y_train)

                    #     # 预测
                    #     y_pred = classifier.predict(X_test)

                    # 计算准确率
                    acc = accuracy_score(y_test, y_pred)
                    accuracies.append(acc)

                # 计算平均准确率
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)

                print(f"平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

                # 存储结果
                results.append(
                    {
                        "Unit": unit,
                        "K": k,
                        "T": t,
                        "Mean_Accuracy": mean_accuracy,
                        "Std_Accuracy": std_accuracy,
                    }
                )

    # 转换结果为DataFrame并输出
    results_df = pd.DataFrame(results)
    print("\n所有结果:")
    print(results_df)

    # 找出最佳参数组合
    best_result = results_df.loc[results_df["Mean_Accuracy"].idxmax()]
    print(f"\n最佳参数组合:")
    print(f"分词单位: {best_result['Unit']}")
    print(f"段落长度 K: {best_result['K']}")
    print(f"主题数量 T: {best_result['T']}")
    print(
        f"准确率: {best_result['Mean_Accuracy']:.4f} ± {best_result['Std_Accuracy']:.4f}"
    )

    # 保存结果
    results_df.to_csv("lda_results.csv", index=False)
    print("\n结果已保存到 lda_results.csv")


if __name__ == "__main__":
    main()
