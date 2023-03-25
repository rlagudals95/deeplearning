import tensorflow as tf
import pandas as pd  # 액셀 데이터 정제
import numpy as np

data = pd.read_csv('gpascore.csv')
# data.isnull().sum() # 액셀의 데이터가 빈 곳을 세어준다.
data = data.dropna()  # 빈데이터가 있는 행을 없애버림
# data.fillna(100) 빈칸을 원하는 값으로 바꿔줌
# data['gpa'].min() # gpa 라는 열의 최솟값을 알려줌 / 최댓값 max / conut 행의 갯수

y데이터 = data['admit'].values # 결과값
x데이터 = []

for i, rows in data.iterrows(): # pandas 데이터 프레임 데이터를 한행씩 출력함
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])

# 1. 모델만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="tanh"),  # 히든레이어 노드의 갯수라고 생각하자
    tf.keras.layers.Dense(128, activation="tanh"),  # 그리고 각각 노드에는 활성함수가 필요하겠지요?
    tf.keras.layers.Dense(128, activation="tanh"),  # 그리고 각각 노드에는 활성함수가 필요하겠지요?
    # 결과값은 1개이기 때문에 마지막 노드는 1개, 0과 1의 확률을 알고 싶으면 sigmoid
    tf.keras.layers.Dense(1, activation="sigmoid"),
])  # keras를 이용한 딥러닝 모델만들기

# 2. 모델컴파일

# optimizer란 learning_rate를 이용해 기울기 값을 적절하게 계산해줌
# 결과가 0 ~ 1의 확률을 출력하는 모델이라면 binary_crossentropy 손실함수를 사용하는 것이 정확도가 좋다.
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])

# x는 학습데이터
# y는 실제값
# ephochs = 학습횟수
model.fit(np.array(x데이터), np.array(y데이터), epochs=1000)


# 예측 최적의 w값을 찾았으니 신규 데이터로 예측해보자
예측값 = model.predict([[750,3.70, 3],[400, 2.2, 1]])
print(예측값)
