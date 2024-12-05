# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 02:08:07 2023

@author: jaeb0
"""
from pybaseball import statcast
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import os

# MLB 2019/06/24-25 pbp 추출
data23 = statcast(start_dt="2023-03-30", end_dt="2023-10-02")
# mlb2023 csv 읽기
os.chdir(r'C:\Users\jaeb0\Desktop\baseball\mlb_pbp')
data23 = pd.read_csv('mlb_pbp_2023.csv')

L, R = -0.708333, +0.708333
Bot, Top = 1.5, 3.5

plt.plot([L, L], [Bot, Top], c='black', lw=1.5)
plt.plot([L, R], [Top, Top] ,c='black', lw=1.5)
plt.plot([R, R], [Top, Bot], c='black', lw=1.5)
plt.plot([R, L], [Bot, Bot], c='black', lw=1.5)

L_p = L+0.1
L_m = L-0.1
R_p = R+0.1
R_m = R-0.1
Center=0

S_height = 0
M_height = -0.6
E_height = -1.0

plt.plot([L_m, Center], [M_height, E_height], c = 'grey', lw = 0.5)
plt.plot([Center, R_p], [E_height, M_height], c = 'grey', lw = 0.5)
plt.plot([R_p, R_m], [M_height, S_height], c = 'grey', lw = 0.5)
plt.plot([R_m, L_p], [S_height, S_height], c = 'grey', lw = 0.5)
plt.plot([L_p, L_m], [S_height, M_height], c = 'grey', lw = 0.5)

ryu = data23[(data23['pitcher']==547943)]

S = ryu[(ryu['description']=='swinging_strike') | (ryu['description']=='swinging_strike_blocked')] # 헛스윙
B = ryu[ryu.description == 'called_strike'] # 스트라이크 콜

x = [R_m, L_p, L_m, Center, R_p, R_m]
y = [S_height, S_height, M_height, E_height, M_height, S_height]

x2 = [L, R, R, L, L]
y2 = [Top, Top, Bot, Bot, Top]

fig = plt.figure()
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.rc('font', family='Malgun Gothic')
plt.scatter(S['plate_x'], S['plate_z'], color = 'red')
plt.scatter(B['plate_x'], B['plate_z'], color = 'blue')
plt.plot(x, y, c='grey', lw = 0.5)
plt.plot(x2, y2, c='black', lw = 1.5)
plt.xlim([L-2.5, R+2.5])
plt.ylim([Bot-3, Top+2])
plt.title('{name} 투구 위치 그리기'.format(name='ryu'))

def strike_zone(player, alt):
    #player를 고정값이 아닌 변하는 값으로 설정
    data = data23[(data23['pitcher']==player)]
    # 스윙과 콜스트라이크 구분
    S = data.loc[(data.description == 'swinging_strike') | (data.description == 'swinging_strike_blocked')]  #헛스윙
    B = data.loc[data.description == 'called_strike']  #스트라이크 콜
    #로케이션 박스
    L, R = -0.708333, +0.708333
    Bot, Top = 1.5, 3.5

    plt.figure(figsize=(4,4), dpi=100)

    x2 = [L, R, R, L, L]
    y2 = [Top, Top, Bot, Bot, Top]

    #홈플레이트 (포수시점)
    L_p = L+0.1
    L_m = L-0.1
    R_p = R+0.1
    R_m = R-0.1
    Center=0

    S_height = 0
    M_height = -0.6
    E_height = -1.0

    x = [R_m, L_p, L_m, Center, R_p, R_m]
    y = [S_height, S_height, M_height, E_height, M_height, S_height]

    plt.rc('font', family='Malgun Gothic')
    #x,y 최대 범위
    plt.xlim([L-2.5, R+2.5])
    plt.ylim([Bot-3, Top+2])
    plt.title('투구 위치 그리기')


    #그래프 그리기 (scatter plot+박스+홈플레이트)
    plt.scatter(S['plate_x'], S['plate_z'], s=50, color='red',  alpha=alp, label='헛스윙')
    plt.scatter(B['plate_x'], B['plate_z'], s=50, color='blue', alpha=alp, label='콜스트라이크')
    plt.plot(x2, y2, c='black', lw=2.5)
    plt.plot(x, y, c='grey', lw=0.5)

    plt.show()
    plt.close()

pitcher=543037
#player="Gerrit Cole"
alp=0.2 #0~1t사이
strike_zone(pitcher, alp)

##################################
# hitmap
import numpy as np
import numpy.random
import matplotlib.pyplot as plt

ryu2 = ryu[~(ryu['plate_x'].isnull())]

a = ryu2['plate_x']
b = ryu2['plate_z']

H, xedges, yedges = np.histogram2d(a, b, bins = 50)

extent = [xedges[0]-1, xedges[-1]+1, yedges[0]-1, yedges[-1]+1]
# imshow()는 기본적으로 2차원 행렬을 그림으로 보여줌
plt.imshow(H.T, extent = extent, origin = 'lower')
plt.colorbar()
plt.xlim([L-2.5, R+2.5])
plt.ylim([Bot-3, Top+2])

plt.plot(x2, y2, c='white', lw=2.5)
plt.plot(x, y, c='grey', lw=0.5)

plt.show()

##################################
ryu['pitch_type'].unique()


fig=plt.figure()
plt.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False 

axes1=fig.add_subplot(2,3,1)
axes2=fig.add_subplot(2,3,2)
axes3=fig.add_subplot(2,3,3)
axes4=fig.add_subplot(2,3,4)
axes5=fig.add_subplot(2,3,5)

FF=ryu2.loc[ryu2.pitch_type == 'FF']  #직구
FC=ryu2.loc[ryu2.pitch_type == 'FC']  #커터
CU=ryu2.loc[ryu2.pitch_type == 'CU']  #커브
SI=ryu2.loc[ryu2.pitch_type == 'SI']  #싱커(투심)
CH=ryu2.loc[ryu2.pitch_type == 'CH']  #체인지업


axes1.scatter(FF['plate_x'],FF['plate_z'], c='#D22D49',alpha=0.3)
axes1.set_title("직구",c='#D22D49')
axes1.plot(x2, y2, c='black', lw=2.5)
axes1.plot(x, y, c='grey', lw=0.5)
axes1.set_xlim(L-2.5, R+2.5)
axes1.set_ylim(Bot-3, Top+2)

axes2.scatter(FC['plate_x'],FC['plate_z'], c='#933F2C',alpha=0.3)
axes2.set_title("커터",c='#933F2C')
axes2.plot(x2, y2, c='black', lw=2.5)
axes2.plot(x, y, c='grey', lw=0.5)
axes2.set_xlim(L-2.5, R+2.5)
axes2.set_ylim(Bot-3, Top+2)

axes3.scatter(CU['plate_x'],CU['plate_z'], c='#888888',alpha=0.3)
axes3.set_title("커브",c='#888888')
axes3.plot(x2, y2, c='black', lw=2.5)
axes3.plot(x, y, c='grey', lw=0.5)
axes3.set_xlim(L-2.5, R+2.5)
axes3.set_ylim(Bot-3, Top+2)

axes4.scatter(SI['plate_x'],SI['plate_z'], c='#FE9D00',alpha=0.3)
axes4.set_title("싱커",c='#FE9D00')
axes4.plot(x2, y2, c='black', lw=2.5)
axes4.plot(x, y, c='grey', lw=0.5)
axes4.set_xlim(L-2.5, R+2.5)
axes4.set_ylim(Bot-3, Top+2)

axes5.scatter(CH['plate_x'],CH['plate_z'], c='#1DBE3A',alpha=0.3)
axes5.set_title("첸접",c='#1DBE3A')
axes5.plot(x2, y2, c='black', lw=2.5)
axes5.plot(x, y, c='grey', lw=0.5)
axes5.set_xlim(L-2.5, R+2.5)
axes5.set_ylim(Bot-3, Top+2)


#겹치지 않게
fig.tight_layout()

####################################
# 등고선 그리기
import seaborn as sns

plt.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False 
#https://stackoverflow.com/questions/43875258/how-to-change-the-positions-of-subplot-titles-and-axis-labels-in-seaborn-facetgr

g = sns.FacetGrid(ryu2, col="pitch_type", col_wrap=5, height=2, ylim=(Bot-3, Top+2),xlim=(L-2.5, R+2.5))
g.map(sns.kdeplot, "plate_x", "plate_z", cmap="flare",shade=True, alpha=0.6)
g.map_dataframe(plt.plot, x2, y2, '-',linewidth=2)
g.map_dataframe(plt.plot, x, y, '-',linewidth=0.5)
g.fig.suptitle('류현진 구종 분포도') # fontsize=16
g.fig.subplots_adjust(top=0.7)
#cbar=True, shade_lowest=False,
#https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html

#겹치지 않게
fig.tight_layout()

###################################
# 좌우 스플릿 추가

#https://stackoverflow.com/questions/43875258/how-to-change-the-positions-of-subplot-titles-and-axis-labels-in-seaborn-facetgr
g = sns.FacetGrid(ryu2, col="pitch_type",row='stand',height=2, ylim=(Bot-3, Top+2),xlim=(L-2.5, R+2.5))
g.map(sns.kdeplot, "plate_x", "plate_z", cmap="flare",shade=True, alpha=0.6)
g.map_dataframe(plt.plot, x2, y2, '-',linewidth=2)
g.map_dataframe(plt.plot, x, y, '-',linewidth=0.5)
g.fig.suptitle('류현진 구종별 좌우 스플릿') # fontsize=16
g.fig.subplots_adjust(top=0.7)
#cbar=True, shade_lowest=False,
#https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html

#겹치지 않게
fig.tight_layout()

#####################################

