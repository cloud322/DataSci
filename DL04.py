from keras.datasets import mnist

import numpy as np
import sys
import tensorflow as tf
import matplotlib.pylab as plt
from keras.utils import np_utils

# see값 설정
seed=20180322
np.random.seed(seed)
tf.set_random_seed(seed)

#MNIST 데이터 불러오기
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print('학습셋 이미지수 : %d' % (x_train.shape[0]))
print('testset 이미지수 : %d' % (x_test.shape[0]))

# 첫번쨰 데이터 그래프로 확인

# plt.imshow(x_train[20],cmap='Greys')
# plt.show()


#이미지를 컴퓨터는
#이미지는 28x28 chd 784픽셀로 이루어저잇음
#각픽셀의 밝기 정도에 따라 0~255 까지 숫자중 하나로 표기
#픽셀단위로 확인
for x in x_train[1112]:
    for i in x:
        sys.stdout.write('%d\t'%i)
    sys.stdout.write('\n')

#이렇게 변환된 픽셀 들의 집합을 고유의 숫자집합으로 바꿔야함 즉
# 784개의 속성을 이용 10개의 결과집합이나오도록해야함
# 입력이미지(3) [0,0,0,1,0,0,0,0,0,0,0,0,0,0]=3
# 입력이미지(9) [0,0,0,0,0,0,0,0,0,1,0,0,0,0]=9
#
# 차원변환
# 1차원으로 변환된 데이터값들은 다시 0,1 사이값으로 변환
x_train =  x_train.reshape(x_train.shape[0],784)
x_train = x_train.astype('float64')
    # 0.1로 변환 하기위해 255로나누려면
    # 정수 데이터를 실수 데이터로 바꿈
x_train = x_train/255   #0,1 중 하나로변환

x_test = x_test.reshape(x_test.shape[0],784).astype('float64')/255

#선택한 데이터의 결과값 미리확인
print('결과값:%d'%(y_train[1122]))

#이미지 인식결과확인
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

print(y_train[1122])