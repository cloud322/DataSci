# matplotlib
# python 에서 데이터과학관련 시각화 패키지

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#   %matplotlib inline    #주피터 노트북에서 show() 없이 호출가능
data = np.arange(10)
plt.plot(data)
plt.show()

#산점도 - 100의 표준 정규분포 난수 생성
list =[]
for i in range(100):
    x=np.random.normal(0.1)
    y= x+0.1,0.2 + np.random.normal(0.1)
    list.append([x,y])
x_data = [v[0] for v in list]   #x=[x,]
y_data = [v[1] for v in list]   #y=[,y]

plt.plot(x_data,y_data,'ro')
plt.show()
print(list)

############################################
df=pd.read_excel('c:/java/sungjuk.xlsx')
print(df)
# xlrd 패키지 팔요
# 총점 평균 계산후 df애 ㅊ가
subj = ['국어','영어','수학','과학']
df['총점']= df[subj].sum(axis=1)
df['평균']= df['총점']/len(subj)
df.sort_values(['평균'],ascending=[False])

import matplotlib as mpl
mpl.rc('font',family='Malgun Gothic')

sj=df.sort_values(['평균'],ascending=[False])
sj.index = sj['이름']
sj['평균'].plot(kind='bar',figsize=(8,4))

# 성적비교 어느반이 잘햇나
c1=df[df['반']==1]
c2=df[df['반']==2]
c1_mean=c1['총점'].sum()/24
c2_mean=c2['총점'].sum()/24
print(c1_mean)
print(c2_mean)

#두집단간의 평균은 유의미하게 차이나는 것인가(t검증)

import scipy.stats as stats
result = stats.ttest_ind(c1['평균'],c2['평균'])
print(result)
    # statistic=0.319960228209846, pvalue=0.755583336185639

# 과목별 평균운 차이가 ?
for sub in subj:
    print(sub,stats.ttest_ind(c1[sub],c2[sub]))

    # 국어 Ttest_indResult(statistic=-2.490140665442242, pvalue=0.031982494983816424)
    # 영어 Ttest_indResult(statistic=-0.6156907152631581, pvalue=0.5518533781528807)
    # 수학 Ttest_indResult(statistic=1.4961318778859336, pvalue=0.1654958420079056)
    # 과학 Ttest_indResult(statistic=4.328442555331755, pvalue=0.0014931977711732465)

#전체 성적 데이터에 대한 그래프 추ㅡㄹ력

sj[subj].plot(kind='bar', figsize=(10,6))
plt.show()

#과목별 점수분포 박스수염그래프 작성
df[subj].boxplot(return_type='axes')
plt.show()
#일반 이반 과목별 점수 분포

c1[subj].boxplot(return_type='axes')
c2[subj].boxplot(return_type='axes')

# 과목별 상관관계 - '수학:과학'  와 '국어:영어'
df.plot(kind='scatter', x='수학', y='과학')
print(stats.pearsonr(df['수학'],df['과학']))
    #(상관계수, p검증) (0.5632890597067751, 0.05650580486155532)

df.plot(kind='scatter', x='국어', y='영어')
print(stats.pearsonr(df['국어'],df['영어']))
    #(상관계수, p검증) (0.10566562777973997, 0.7437959551857836)