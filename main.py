import numpy as np
import pandas as pd

# 读取数据及预处理
df = pd.read_csv('./data/heart_2020_cleaned.csv')
df = df
print(df.info())
df = df[df.columns].replace({'Yes': 1, 'No': 0,
                             'Male': 1, 'Female': 0,
                             'No, borderline diabetes': 0, 'Yes (during pregnancy)': 1})  # 注意Diabetic 有特殊的数值
BMI_bin = [0, 18.5, 24, 28, 100]
df['BMI'] = pd.cut(df['BMI'], BMI_bin, labels=False)

SleepTime_bin = [0, 6, 8, 24]
df['SleepTime'] = pd.cut(df['SleepTime'], SleepTime_bin, labels=False)

Health_bin = [-1, 10, 20, 30]
df['MentalHealth'] = pd.cut(df['MentalHealth'], Health_bin, labels=False)
df['PhysicalHealth'] = pd.cut(df['PhysicalHealth'], Health_bin, labels=False)
df = pd.get_dummies(df, drop_first=True)  # 转换为虚拟变量。这是计量经济学中常用的一个技巧

# 处理类别不均衡的问题。先对数据集进行划分，然后对训练集进行平衡
from imblearn.over_sampling import SMOTE
# Set Training and Testing Data
from sklearn.model_selection import train_test_split

X = df.drop(columns=['HeartDisease'], axis=1)
y = df['HeartDisease']
X_res, X_test, y_res, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=5)  # 划分训练集和测试集

X_train, y_train = SMOTE().fit_resample(X_res, y_res)  # 进行类别均衡操作

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of testing label:', y_test.shape)

from collections import Counter

print(Counter(y_train), Counter(y_test))

# 使用自己的随机森林模型进行预测
from RandomForestClassification import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100,
                             max_depth=5,
                             min_samples_split=6,
                             min_samples_leaf=2,
                             min_split_gain=0.0,
                             colsample_bytree="sqrt",
                             subsample=1,
                             random_state=66,
                             oob_score=True)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, average_precision_score

print('ACC of training set:', accuracy_score(y_train, clf.predict(X_train)))

y_pred, y_prob = clf.predict(X_test, return_prob=True)
print('ACC of test set:', accuracy_score(y_test, y_pred))
print('AUPRC', average_precision_score(y_test, y_prob))
oob_error = clf.oob_errors(y_train)
print(f'oob_error:{oob_error}')
