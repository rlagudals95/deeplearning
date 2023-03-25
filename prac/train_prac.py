import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

# x에다가 뭔짓을 해야 y의 행렬이 나올까?
# 실은 2를 곱하고 1을 더하면 됨

# 모델 만들기

a = tf.Variable(0.1)
b = tf.Variable(0.1)



opt = tf.keras.optimizers.Adam(learning_rate = 0.01)

def 손실함수():
    예측_y = train_x * a + b # List 자료형에 연산이 된다.. 각각의 요소에 연산이 적용됨
    return tf.keras.losses.mse(train_y, 예측_y) # mean squared error (예측1 - 실제1)^2 + (에측2 - 실제2)^2 ....

for i in range(5000):
    opt.minimize(손실함수, var_list = [a, b])
    print('가중치 프린트 :', a.numpy(), b.numpy())