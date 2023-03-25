# 뉴럴 네트워크에 집어 넣을 수 있는 것은 무조건 숫자!
import tensorflow as tf
import numpy as np

# ((), ()) 튜플형태의 이미지 데이터
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0 # 작은 숫자로 재가공 안해도됨
testX = testX / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover'
               'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

trainX = trainX.reshape((trainX.shape[0]), 28, 28, 1)  # 데이터 재가공 1을 추가해서 reshape 괄호가 쳐짐
textX = testX.reshape((testX.shape[0]), 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same",  # 이미지의 패딩값
                           activation="relu", input_shape=(28, 28, 1)),  # 데이터 하나의 모양 input_shape 마지막 값이 3이여야 컬러값 1이면 흑백
    # 이미지 학습을 위한 컨볼루션 레이어
    # 32개의 다른 feature를 생성해라
    # 커널 사이즈는 3x3
    # 이미지 데이터는 0~255임 그래서 활성함수로 relu를 사용한다!
    # tf.keras.layers.Dense(128, input_shape=(28, 28), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)), # maxpooling으로 사이즈를 줄여주고 중요한 부분만 추려낸다.
    tf.keras.layers.Flatten(),  # 행렬을 1차원으로 바꿔줌 [28, 28]의 행렬로 계산이 불가능하기 때문
    tf.keras.layers.Dense(64, activation="relu"),  # 음수의 확률은 0으로
    tf.keras.layers.Dense(
        10, activation="softmax")  # 각각 제품 사진일 확률
])

# model.summary() 모델의 요약본 출력 / input_shape 집어 넣을 데이터의 모양을 넣어줘야함

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=['accuracy'])
# sparse_categorical_crossentropy
# 위의 손실함수는 여러가지 카테고리가 있을때 자주 쓴다.
model.fit(trainX, trainY, validation_data=(textX, testY) , epochs=5)
# validation_data epochs마다 테스트 데이터로 채점함 

# 학습까지 완료된 모델을 평가함
# score =  model.evaluate(testX, testY)
# print(score)
model.predict()