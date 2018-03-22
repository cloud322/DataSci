#MNIST
#MNIST 이미지 데이터집합 다운로드
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

#mnist 데이터 다운로드
(x_train, y_train),(x_test, y_test) = mnist.load_data()

mnist = input_data.read_data_sets('MNIST/', one_hot=True)

#손글씨를 픽셀화 한 결과를 저장할 심블릭 변수 선언
#손글씨 이미지는 28x28 크기, 따라서 총 784 픽셀이 필요
x = tf.placeholder(tf.float32, [None, 784])  #784개의 벡터생성

W = tf.Variable(tf.zeros([784, 10]))  #가중치 필드
b = tf.Variable(tf.zeros([10]))       #은닉층 필드 (bais)

y = tf.nn.softmax(tf.matmul(x, W) + b)    #행렬곱, 벡터합


y_ = tf.placeholder(tf.float32, [None, 10])
#크로스엔트로피 연산을 위한 변수 선언

cross_entropy = tf.reduce_mean(
       -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#각 원소y들 간의 log 계산하고 합한 후 평균 계산

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#딥러닝 학습시 오차와 비용을 최소화 하기 위해 경사하강법 사용

init = tf.global_variables_initializer()
#위에서 생성한 변수들을 초기화함

sess = tf.Session()
sess.run(init)
#텐서플로 작업 생성 및 실행

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#무작위로 선별된 데이터로 딥러닝 학습 - 1000번

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#테스트 데이터를 통해 예측 테스트

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#테스트 결과값들을 평균으로 계산

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#테스트 정확도 평가