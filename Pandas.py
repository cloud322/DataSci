import numpy as np
import pandas as pd
import matplotlib as plt
import scipy.stats as stats

df=pd.read_excel('c:/java/sungjuk.xlsx')
print(df)
# xlrd 패키지 팔요
# 총점 평균 계산후 df애 ㅊ가
subj = ['국어','영어','수학','과학']
df['총점']= df[subj].sum(axis=1)
df['평균']= df['총점']/len(subj)
df.sort_values(['평균'],ascending=[False]) #평균으로정렬

import matplotlib as mpl
mpl.rc('font',family='Malgun Gothic')   #그래프한글설정

sj=df.sort_values(['평균'],ascending=[False])
sj.index = sj['이름']
sj['평균'].plot(kind='bar',figsize=(8,4))

# DataFrame 객체생성 :{'key':['val','val',....]}
data= {'이름':['수지','혜교','지현'],
       '국어':['99','99','99'],
       '영어': ['98', '98', '98'],
       '수학': ['97', '97', '97']}
sj=pd.DataFrame(data,columns=['이름','국어','영어','수학'])
print(sj)
print(sj['이름']) #특정컬럼

#Series :1차원자료구조 df에서 특정 컬럼 추출시 Series 생성

data=[4,5,6,7,8,9,10]
print(data)             #그냥 1차월배열
print(pd.Series(data))  #행번호가있는 다차원배열

#ㄷ데이터프레인ㅁ에 새로운컬럼추가
name=['수지','혜교','지현']
kor=['99','99','99']
eng=['98', '98', '98']
mat=['97', '97', '97']

data={'이름':name,'국어':kor,'영어':eng,'수학':mat}
sj =pd.DataFrame(data,columns=['이름','국어','영어','수학'])
print(sj)

gender =pd.Series(['남','여','여'])
sj['성별']=gender
print(sj)

sj = sj.drop('성별', axis=1)
print(sj)
