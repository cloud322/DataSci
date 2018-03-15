#파이썬으로 상관분석 회기분석테스트
import numpy as np
import pandas as pd

#csv 파일 읽어오기

hdr = ['V1','V2','V3','V4','V5','V6','V7','V8','V9']
df = pd.read_csv('c:/java/phone-02.csv', header=None,names=hdr)
print(df)

#상관분석
dfc=df.corr()
print(dfc)

# df97=df['V9'].corr(df['V7'])
# or
df97=df.V9.corr(df.V7)

#회기분석

from scipy import stats
lm=stats.linregress(df.V7,df.V9)
# 기울기 절편 상관도 오류지수
print(lm)

#회기식 - y=6.282598387861545x-272.0009483167378

#
from scipy import polyval
make = [3.24,3.57, 2.64, 4.63, 4.83, 3.82, 6.43, 2.55]
power= [7.24,4.57, 1.64, 8.63, 6.83, 1.82, 3.43, 1.55]

slope, intercept, rvalue, pvalue, stderr =stats.linregress(make,power)
mp = pd.DataFrame(make,power)

print(slope)
print(intercept)
#매출 4억 전기사용량은?

import matplotlib.pyplot as plt
import matplotlib

krfont={'family':'Malgun Gothic','weight':'bold','size':10}
matplotlib.rc('font',**krfont)

ry=polyval([slope,intercept],make)
plt.plot(make, power,'b*')    #파랑점(,o > ^ 1 s,p *)
plt.plot(make, ry,'r.-.')     #빨강점 실선
plt.title('회기분석결과')
plt.legend(['실제데이터','회기분석을 따르는 모델'])
plt.show()