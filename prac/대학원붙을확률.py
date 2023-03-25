import tensorflow as tf


# 1. 모델만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="tanh"), # 히든레이어 노드의 갯수라고 생각하자
    tf.keras.layers.Dense(128, activation="tanh"), # 그리고 각각 노드에는 활성함수가 필요하겠지요?
    tf.keras.layers.Dense(1, activation="sigmoid"), # 결과값은 1개이기 때문에 마지막 노드는 1개, 0과 1의 확률을 알고 싶으면 sigmoid 
]) # keras를 이용한 딥러닝 모델만들기

# 2. 모델컴파일

# optimizer란 learning_rate를 이용해 기울기 값을 적절하게 계산해줌
# 결과가 0 ~ 1의 확률을 출력하는 모델이라면 binary_crossentropy 손실함수를 사용하는 것이 정확도가 좋다.
model.complie(optimizer="adam", losses="binary_crossentropy", metric=["accuracy"])

# x는 학습데이터
# y는 실제값
# ephochs = 학습횟수
model.fit(x데이터, y데이터, epochs=10)