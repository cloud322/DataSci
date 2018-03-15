#판다스 데이터가공

#데이터 취득및 가중
import numpy as np
import pandas as pd

#데이터 읽어 들이기
gap = pd.read_csv('c:/java/GAP_5year.tsv',sep='\t')
print(gap.head())
print(gap.tail())
print(gap.info())
print(gap.describe())

#데이터 조회
#통계자료에서ㅗ 2007 한국데이터 조회
#
# R dplyr - filter(gap, country=='' & year==)
kor = gap.query('country=="Korea, Rep." & year==2007')
print(kor)

#정령해서 출력
# R dplyr - gap %>% arange(country,year)
sort= gap.sort_values(by=['year','country'])
print(sort.head())

#부분열 선택하기
#인구수 GDP

partcol =gap[['pop','gdpPercap']]
print(partcol.head())

#특정열 추가
#총GDP - pop*gdpPercap
#gdp_ratio - lifeExp/gdpPercap
#gdp_perc - gdp_ratio*100
gap['gdp']= gap['pop']*gap['gdpPercap']
# gap.gdp = gap.pop*gap.gdpPercap


gap['gdp_ratio']= gap['lifeExp']/gap['gdpPercap']
gap['gdp_perc']= gap['gdp_ratio']*100
print(gap.head())

# 통계량계사하기
print(gap.aggregate(['mean','median']))
print(gap.aggregate(['sum','min','max']))

# 표번 sampling
np.random.seed(6412)
print(gap.sample(n=10))

print(gap.country)
print(gap.year)
print(gap.year.unique())

print(gap.drop_duplicates(['country','year']).head())


# group by 연산
#2007 기준 대륙별 생활지수 평균값 분석
lifeExp2007=gap.query('year==2007').groupby('continent').agg({'lifeExp':'mean'})
print(lifeExp2007)

