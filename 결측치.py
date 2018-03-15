import numpy as np
import pandas as pd

#난수를 이용햐 5x3 데이터프래임 생성
df=pd.DataFrame(np.random.rand(5,3),columns=['col1','col2','col3'])
print(df)

#결측치 삽입
df.ix[0,0]=None
df.ix[1,['col1','col2']]= np.nan
df.ix[2,'col2']= np.nan
df.ix[3,'col2']= np.nan
df.ix[4,'col3']= np.nan

print(df)

#결측치 해결 2 fillna 을 이용 문자열로 채움

dfc=df.fillna('결측')
print(dfc)

#결측치 해결 3 fillna 를 이용 평균값으로 채움
df_mean = df.mean()
dfm=df.fillna(df_mean)
print(dfm)


