#ML01

# 규칙기반(1950) - 신경망시대(1960) - 통계학기반머신러닝(1990)- 빅데이터(2010)->딥러닝(2013)

# 머신러닝기반 데이터 분석 기법의 유형

# 지도 학습 (Supervised Learning)
# - 설명 혹은 예측하고자 하는 목적변수(혹은 반응변수, 종속변수)의 형태가
#   수치형(양적변수) 인가 범주형(질적변수)인가에 따라 분류와 수치예측 방법
#   ex)현재주식시장변화학습-> 미래주식시장변화예측
#
#   회기:숫자값예측
#       분류 - 입력된데이터를 주어진 항복으로 나눔
#       순위/추천 - 선호도 예측
# - K-최근접 이웃(K-Nearest Neighbors) - 선형 회귀(Linear Regression)
# - 로지스틱 회귀(Logistic Regression)
# - 확장된 회귀분석(ex : 다항회귀, 비선형 회귀, 벌점화 회귀 등)
# - 인공 신경망 분석(Artificial Neural Network) - 인공 신경망 분석(Artificial Neural Network)
# - 의사결정트리(Decision Tree) - 의사결정트리(Decision Tree)
# - 서포트 벡터 머신(Support Vector Machine)
# - 서포트 벡터 머신(회귀) (Support Vector Machine (Regression))
# - 나이브 베이즈(Naive Bayes) - PLS(Partial Least Squares)
# - 앙상블 기법(랜덤 포레스트 등) - 앙상블 기법(랜덤 포레스트 등)


# 비지도학습(UnsupervisedLearning)
# -사전정보가 없는 상태에서 유용한 정보나 패턴을 탐색적으로 발견
#   ex)군집화 이상검출
#
#   밀도추정-분포예측
#   자원축소-차원간추림

# 강화학습
# 장기적 이득을 최대화하도록 하는학습방법
# 현재수에서 다음수를 선택하는 것은 지도학습에 속함

# 딥러닝 신경망을 층층이쌓아 문제를 해결하는 기법 총칭 데이터양에 의존하는
# 기법으로 다양한 패턴과 경우에 유연하게 대응하는 구조 많은 데이터를 이용해야
# 능력이 향상되는 구조 채택

# 비즈니스이해및 문제정의
# 데이터수집
# 데이터전치리 탐색
# 데이터 모델훈련 - 결측치 이상치 처리
# 모델 성능평가 - train/test 데이터분리
# 모델 성능 향상

# 데이터분할 방법 1 연속적인순서대로나눔
# iris_train<-iris[1:105]
# iris_test<-iris[106:150]
#
# 데이터분할방법 2 무작위추출로 나눔
# idx<-sample(1:nrow(iris),size=nrow(iris)*0.7,replace=F)
# iris_train<-iris[idx,]
# iris_test<-iris[-idx,]

# 지도학습 모델 적용
# K-최근접 이웃(K-Nearest Neighbors)
# 해당 데이터 세트와 가장 유사한 주변 데이터 세트의 범주로 지정
# 데이터 간의 유사성을 측정하는 방식은 여러가지
# 상품및 서비스 추천

# iris - 1953 통계학자의 유사성분류에사용된 데이터
# python 에서ML 구현하려면
# numpy scipy matplotlib ipython
# scikit-learn pandas
# ML 위한 데이터집합 라이브러리 sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
iris = load_iris()
print(iris)

print(iris.keys())  #iris datakey 출력
print(iris['data'][:5]) #iris 처음 5행
print(iris['target'][:5])   # 타겟 품정 처음 5행
print(iris['target_names']) # iris 타겟 품정 출력

# scikit-learn 에 train_test_spkit 함수 이용
# 데이터집합을 일정비율로 나눠 train/test 로작성
from sklearn.model_selection import train_test_split

print(iris.keys())
print(iris['data'].shape)   #iris 데이터크기
print(iris['target'].shape) #iris 타겟크기

x_train,x_test,y_train, y_test= train_test_split(iris['data'],iris['target'],random_state=0)

print('학습데이터크기',x_train.shape)
print('평가데이터크기',x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 머신러닝 모델 만들기전에 머신러닝을 이용해서
# 풀어도되는 문제인가 데이터에 이상은 없는지 여부확인 위해 시각화도구이용
# 대표적 시각화도구 :산포도

# 단 데이터가 행렬로 작성되어 있기 때문에
# dataframe 으로 변환필요
from pandas.plotting import scatter_matrix
iris_df = pd.DataFrame(x_train)
scatter_matrix(iris_df,c=y_train,figsize=(15,15),marker='o',s=60,alpha=0.8)
plt.show()

#k-최근접 알고리즘에서 k는 가장가까운 이웃 하나 가아니라
# 훈련데이터중 새로운 데이터와 가장 가까운 k개의 이웃을 찾는다의미
# 머신러닝의 모든 모델은 scikit-learn 의 Esimator 클래스에구현
# KNN 알고리즘은 neighbor 모듈의 KNeighborsClassifier 클래스에 구현

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=1)    #k=1 로설정
knn.fit(x_train,y_train)    # train data 로 분류모델 학습

x_new = np.array([[5,2.9,1,0.2]])   #예측을 위한 데이터생성

prediction = knn.predict(x_new)     #예측값조사
print('예측결과',prediction)
print('예측결과 대비 품종',iris['target_names'][prediction])

# 예측모델평가 -신뢰성확인
# 앞서만든 test데이터집합이용
y_pred=knn.predict(x_test)

print('test 데이터 이용예측값',y_pred)
print('test 데이터대비 에측값 정확도',knn.score(x_test,y_pred))

# 대부분의 지도학습 알고리즘(의사결정나무 SVM 나이브베이지안)이 그렇듯 주어진
# 학습자료들로부터 모형을 추정하여 새로운 실증자료가 주어지면 모형에 적합하여 예측값을 산출함 이러한
# 학습방법을 eager 방식이라함

#하지만 knn은 학습자료가 주어져도 아무런 움직임이없다가 실증자료가 주어저야만
#그떄부터 움직이기시작함 이러한 학습방법을
#lazy 방식이라함

#지도학습 분류방법 중 가장 간단한 방법 KNN
#많은 메모리 소요 (대용량 데이터일때 불리함)
#대안 로지스틱회귀 딥러닝 이용

# 분류의개념
# 미리정의되 가능성 있는 여러 클래스 결과값 중하나를 예측
# iris의 경우 결과값은 모두 3가지 setosa versicolor virginica
# 이진분류 질문의답중 하나를 에측(스펨)
# 다항분류 3가지이상 질문의답중 하나를예측

# 일반화 과대적합 과소적합
# 일반화 - 모델을통해 처음보는 데이터에 대해 정확히예측할수 잇는경우
# 과대적합 - 나이가 45이상 자녀가 3이상
#           이혼하지 않은 고객은 요트를 구매할 것이다
#           너무많은 특성을 이용해서 복잡한모댈만드는경우
# 과소적합 - 애완견이있는 고객은 요트를 구매
#           너무적은 특성이용 단순한 모델

# KNN 알고리즘 k값의 변화에 따른 산점도 비교
# mglearn 패키지
import mglearn as mg
x,y = mg.datasets.make_forge() #데이터집합생성
mg.discrete_scatter(x[:,0],x[:,1],y)    #산점도작성
plt.show()

#k값이 1일때 KNN 알고리즘 산점도
mg.plots.plot_knn_classification(n_neighbors=1)
plt.show()
mg.plots.plot_knn_classification(n_neighbors=3)
plt.show()
mg.plots.plot_knn_classification(n_neighbors=5)
plt.show()

#값을 3으로 설정후 KNN알고리즘 적용
#데이터집합생성
x,y =mg.datasets.make_forge()
x_train,x_test,y_train, y_test= train_test_split(x,y,random_state=0)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)    #train 데이터로 학습
print(knn.predict(x_test))  #test 데이터로 예측
print('정확도',knn.score(x_test,y_test)) #예측평가



