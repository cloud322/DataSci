#deep learning
# 딥러닝 기초 텐서플로 케라스
# 딥러닝 개발환경
# 1#파이썬 3.5이상, 아나콘다 4.4 이상
# 2#파이참 파이썬 venv 환경에서 실행하도록 설정
# 3#머신러닝 딥러닝 관련 패키지 설치
# numpy scipy ma`tplotlib pandas scikit-learn
# spyder seaborn h5py pillow
# tensorflow keras

# 텐서프롤우 GPU 지원사이트
# CUDA - developer.nvidia.com/cuda-downloads
# cuDNN - developer.nvidia.com/cudnn

# 설치확인
# 파이참 - '파이썬 콘솔'에서 다음 실행
import tensorflow as tf
print(tf.__version__) #tensorflow 버전확인

import keras
hello=tf.constant('hello,Tensor')
sess=tf.Session()      #작업생성
print(sess.run(hello))  #작업실행

# 인공지능 -관념적으로 컴퓨터가 인간이 사고를 모방하는 것 즉 기계가 인간처럼 사고하고 행동하개 하는 것
# 머신러닝 -주어진 데이터를 통해 컴퓨터가 스스로 학습하는 것
#         학습 : 데이터 입력해서 패턴분석하는 과정
# 딥러닝 - 인공신경망을 이용해 컴퓨터가 스스로 학습
#         인공신경망  : 인간의 뇌의 동장방식 착안
# 2012 Image.net 에서 제공하는 1000개의 카테고리로
# 분류된 100만개의 이미지를 인식하여 정확성을 겨루는 ILSVRC아는 이미지인식
# 대회에서 84.7%라는 인식률을 달성 그전에는 75% 현재 97%
#
# 인공신경망은 이미 1940년 부터 연구
# 그전에는 여러문제
# 빅데이터와 GPGU 발전 딥러닝 알고리즘의 발명 덕택
# 급격히발전 (간단한 수식을 효율적으로 실행해주는 알고리즘)

# 케라스 카페 토치 MXNet 체이너 CNTK
# 텐서플로우를 좀더 사용하기 쉽게 만들어주는 보조 라이브러리

# 인공신경망 구동 예제
x1='데이트요청'
w1='너무좋다 : 7'
w1='싸움 :0.1'

x2='눈옴 ㅜㅜ'
w2='눈좋다:5'
w2='눈ㄴㄴ:1'

x3='배고파:1'
w3='배불러:1'
w3='배고플듯:1'

y=0:'집',1:'나간다'

y= 0.5x0.1 + 1 = 5.5

# 딥러닝 구동 필요 케라스 함수불러옴
from keras.models import Sequential
from keras.layers import Dense

# 머신러닝 관련 라이브러리 불러옴
import numpy as np
import tensorflow as tf

# 난수 생성을 위한 초기화
seed = 9563
np.random.seed(seed)
tf.set_random_seed(seed)

#준비된 환자정보를 불러옴
#종양 유형 폐활량 호흡곤란여부 고통정도 기침 흡연 천식여부
data_set = np.loadtxt('data/ThoraricSurgery.csv',delimiter=',')

#환자기록과 수술결과를 x y 구분지정
x=data_set[:,0:17]
y=data_set[:,17]

#딥러닝 실행 방식을 결정 (모델 설정 및 실행방법 정의)

# https://tykimos.github.io/2017/01/27/MLP_Layer_Talk

model= Sequential()
model.add(Dense(30,input_dim=17,activation='relu'))
#입력데이터 17 은닉층 갯수는30 적용 알고리즘 함수relu

model.add(Dense(1,activation='sigmoid'))
#출력 데이터 1 적용 알고리즘 함수 sigmoid

#딥러닝실행
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=30,batch_size=10)
# loss 오차값추적방식 optimizer 는 오차수정함수

#결과검증 및 출력
print('정확도 : %.4f'%(model.evaluate(x,y)[1]))