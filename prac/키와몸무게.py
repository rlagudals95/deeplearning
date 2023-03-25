import tensorflow as tf

# 키 = [170, 180, 175, 160]
# 신발 = [260, 270, 265, 255]

# y = 키, x는 신발사이즈
# y = ax + b
# a와 b의 가중치를 구하는 것이 목적
# 키를 대입했을때 신발 사이즈를 측정하기

키 = 170
신발 = 260

# 신발 = 키 * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

# keras.optimizer -> 경사하강법을 이용해 위의 변수들을 업데이트 시켜주는 함수!
# gradient를 알아서 적절하게 업데이트 해줌 -> 적절하게 가중치를 업데이트
opt = tf.keras.optimizers.Adam(learning_rate = 0.1)

def 손실함수():
    예측값 = 키*a + b # 음수가 나오지 않게 제곱으로 손실값 리턴
    return tf.square(신발 - 예측값)

for i in range(300):
    opt.minimize(손실함수, var_list = [a, b])
    print('가중치 프린트 :', a, b)


