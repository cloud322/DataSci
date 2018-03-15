# numpy
# 선형대수(배열 행렬), 연산에 효과적인 함수 제공
# 리스트 자료구조와 유사 파이썬 리스트보다 속도가 빠름
# randn(행 열) - 다차원배열 정규분포를 따르는 난수 생성
# array(리스트) - 다차원배열 생성
# arrange - 0~n-1 정수생성

from Numpy import genfromtxt    #텍스트 파일을 배열로 생성

data=np.random.rand(3,4) #3*4난수생성
print(data)

lotto=np.random.randint(46,size=(3,6))   #3x6행렬에 난수생성
print('lotto',lotto)

list = [3,4.1,5,6.3,7, 8.6]

arr=np.array(list)
print('평균',arr.mean())
print('합',arr.sum())
print('max',arr.max())
print('min',arr.min())
print('var',arr.var())
print('sd',arr.std())

list=[[9,8,7,6,5],[1,2,3,4,5]]
arr=np.array(list)
print(arr)
print(arr[0,2]) #7
print(arr[1,3]) #4

print(arr[0,:]) #98765 행전체
print(arr[:,1]) #82    열전체

#자동으로 채워지는 행렬 생성
zarr=np.zeros((3,5)) #0으로 채워지는 3x5행렬생성
print(zarr)
#정수생성
cnt=0
for i in np.arange(3):
    for j in np.arange(5):
        cnt+=1
        zarr[i,j]=cnt
print(zarr)

# 외부 csv 파일 읽어 배열 생성
phone = np.genfromtxt('c:/java/phone-01.csv', delimiter=',')
print(phone)
print(np.mean(phone[:,2])) #화면크기따른 평균출력
print(np.median(phone[:,2]))
print('총갯수',len(phone))

p_col13=phone[:,2]
print(np.percentile(p_col13,0))     #사분위 최소
print(np.percentile(p_col13,25))    #1사분위
print(np.percentile(p_col13,50))    #2사분위
print(np.percentile(p_col13,75))    #3사분위
print(np.percentile(p_col13,100))   #사분위ㅣ 최대

#scipy에는 여러가지 기술통계 한번에 계산해주는 describe 함수 있음
from scipy.stats import describe
print(describe(p_col13))
