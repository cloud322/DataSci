 # DL03
# 딥러닝 기초 예제 - MINST 데이터 집합을 이용한 손글씨 식별
# 미국 국립표준기술원(MIST)에서 고등학생, 인구조사국 직원등이
# 쓴 손글씨를 이용한 데이터들로 구성 - 7만개 글자(0-9) 이미지

# 그리고, 각 데이터는 55,000개의 학습 데이터( mnist.train ),
# 10,000개의 테스트 데이터( mnist.text ),
# 5,000개의 검증 데이터( mnist.validation ) 등으로 구성

# 먼저, 이미지를 배열로 표현하는 방법에 대해 알아보자
# 즉, 이미지를 average hash 방식으로 변환해서 배열로 표현

from PIL import Image
import numpy as np

# 이미지를 avhash 형태로 변환하는 함수
#  이미지를 0 1 형태의 배열로 나타내기위한 방법
# 알고리즘이 간단하고 변환속도가 빨라서
#  이미지비교시 자주사용되는기법
#  처리순서
#  이미지를 흑백으로바꾸고 크기를 줄인후
#  픽셀 평균값 을 계산한 후 그것에 따라 0,1

# 이미지를 avhash 형태로 변환하는 함수
def average_hash(fname, size=16):
    img = Image.open(fname) # 이미지 파일을 읽기
    img = img.convert('L') # 이미지를 흑백으로 변환
    img = img.resize((size,size), Image.ANTIALIAS) # 이미지크기 변환

    pixel_data = img.getdata() # 픽셀 데이터 가져옴
    pixels = np.array(pixel_data) # 픽셀 데이터를 numpy 배열로 생성
    pixels = pixels.reshape((size, size)) # 2차원배열로 변환
    avg = pixels.mean() # 평균값 구하기
    diff = 1 * (pixels > avg) # 평균보다 크면 1, 작으면 0으로
    return diff

# 바이너리를 hash로 변환하는 함수
# 두 이미지를 유사성을 검사할때 픽셀 하나하나 비교하는것은 비효율적
# 따라서 이미지를 변환한 전체 픽셀에대해 hash code 를 생성
그것으로 오차범위내에서 비교하는것이 편리
def np2hash(n):
    bhash=[]
    for n1 in n.tolist():
        s1=[str(i) for i in n1]
        s2="".join(s1)
        i=int(s2,2)
        bhash.append("%04x"%i)
    return "".join(bhash)

#예제용 이미지를 이용해서 이진코드로 구성된 배열 출력
ahash = average_hash('data/tower.jpg')
print(ahash)
print(np2hash(ahash))

ahash = average_hash('data/tower.jpg',size=24)
print(ahash)
print(np2hash(ahash))

ahash = average_hash('data/etherum_200px.png',size=30)
print(ahash)
print(np2hash(ahash))