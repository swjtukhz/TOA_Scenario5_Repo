import seaborn as sns
import numpy as np
import math
import sympy
from pygments.lexers import configs
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import random

# 固定随机值
random.seed(0)
np.random.seed(0)
# 计算实际距离
def distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

# 计算RMSE
def root_mean_squared_error(hat, target):
    return np.sqrt(mean_squared_error(hat, target))

# 三基站，由D和A解算定位坐标
def triposition(xa, ya, da, xb, yb, db, xc, yc, dc):
    x, y = sympy.symbols('x y')
    f1 = 2 * x * (xa - xc) + np.square(xc) - np.square(xa) + 2 * y * (ya - yc) + np.square(yc) - np.square(ya) - (
                np.square(dc) - np.square(da))
    f2 = 2 * x * (xb - xc) + np.square(xc) - np.square(xb) + 2 * y * (yb - yc) + np.square(yc) - np.square(yb) - (
                np.square(dc) - np.square(db))
    result = sympy.solve([f1, f2], [x, y])
    locx, locy = result[x], result[y]
    return [locx, locy]

# NLOS-W的err计算公式
def err_w(w,sigma):
    err = w*(math.sqrt(sigma)-1)+np.random.normal(0,0.15)
    return err

def trilateration_3anchor(guess_, args_):
    x, y = guess_
    x1_, y1_, x2_, y2_, x3_, y3_, d1, d2, d3 = args_

    EQ1 = (x1_ - x) ** 2 + (y1_ - y) ** 2 - (d1) ** 2
    EQ2 = (x2_ - x) ** 2 + (y2_ - y) ** 2 - (d2) ** 2
    EQ3 = (x3_ - x) ** 2 + (y3_ - y) ** 2 - (d3) ** 2

    return (EQ1, EQ2, EQ3)

def ls_3anchors(x1, x2, x3,y1, y2, y3,ttw_r1_los, ttw_r2_los, ttw_r3_los):
        x0 = (sum([x1, x2, x3]) / 3, sum([y1, y2, y3]) / 3)
        opti_args = [x1, y1, x2, y2, x3, y3, ttw_r1_los, ttw_r2_los, ttw_r3_los]
        res = least_squares(trilateration_3anchor, x0, args=(opti_args,), method='lm')
        loc_los = [res.x[0], res.x[1]]
        return loc_los

def trilateration_4anchor(guess_, args_):
    x, y = guess_
    x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_, d1, d2, d3, d4 = args_

    EQ1 = (x1_ - x) ** 2 + (y1_ - y) ** 2 - (d1) ** 2
    EQ2 = (x2_ - x) ** 2 + (y2_ - y) ** 2 - (d2) ** 2
    EQ3 = (x3_ - x) ** 2 + (y3_ - y) ** 2 - (d3) ** 2
    EQ4 = (x4_ - x) ** 2 + (y4_ - y) ** 2 - (d4) ** 2
    return (EQ1, EQ2, EQ3, EQ4)

def ls_4anchors(x1, x2, x3, x4, y1, y2, y3, y4, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los):
        x0 = (sum([x1, x2, x3, x4]) / 4, sum([y1, y2, y3, y4]) / 4)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los]
        res = least_squares(trilateration_4anchor, x0, args=(opti_args,), method='lm')
        loc_los = [res.x[0], res.x[1]]
        return loc_los

def trilateration_5anchor(guess_, args_):
    x, y = guess_
    x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_, x5_, y5_, d1, d2, d3, d4, d5 = args_

    EQ1 = (x1_ - x) ** 2 + (y1_ - y) ** 2 - (d1) ** 2
    EQ2 = (x2_ - x) ** 2 + (y2_ - y) ** 2 - (d2) ** 2
    EQ3 = (x3_ - x) ** 2 + (y3_ - y) ** 2 - (d3) ** 2
    EQ4 = (x4_ - x) ** 2 + (y4_ - y) ** 2 - (d4) ** 2
    EQ5 = (x5_ - x) ** 2 + (y5_ - y) ** 2 - (d5) ** 2
    return (EQ1, EQ2, EQ3, EQ4, EQ5)

def ls_5anchors(x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los):
        x0 = (sum([x1, x2, x3, x4, x5]) / 5, sum([y1, y2, y3, y4, y5]) / 5)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los]
        res = least_squares(trilateration_5anchor, x0, args=(opti_args,), method='lm')
        loc_los = [res.x[0], res.x[1]]
        return loc_los

def trilateration_6anchor(guess_, args_):
    x, y = guess_
    x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_, x5_, y5_, x6_, y6_, d1, d2, d3, d4, d5, d6 = args_

    EQ1 = (x1_ - x) ** 2 + (y1_ - y) ** 2 - (d1) ** 2
    EQ2 = (x2_ - x) ** 2 + (y2_ - y) ** 2 - (d2) ** 2
    EQ3 = (x3_ - x) ** 2 + (y3_ - y) ** 2 - (d3) ** 2
    EQ4 = (x4_ - x) ** 2 + (y4_ - y) ** 2 - (d4) ** 2
    EQ5 = (x5_ - x) ** 2 + (y5_ - y) ** 2 - (d5) ** 2
    EQ6 = (x6_ - x) ** 2 + (y6_ - y) ** 2 - (d6) ** 2
    return (EQ1, EQ2, EQ3, EQ4, EQ5, EQ6)

def ls_6anchors(x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los):
        x0 = (sum([x1, x2, x3, x4, x5, x6]) / 6, sum([y1, y2, y3, y4, y5, y6]) / 6)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los]
        res = least_squares(trilateration_6anchor, x0, args=(opti_args,), method='lm')
        loc_los = [res.x[0], res.x[1]]
        return loc_los

def trilateration_7anchor(guess_, args_):
    x, y = guess_
    x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_, x5_, y5_, x6_, y6_, x7_, y7_, d1, d2, d3, d4, d5, d6, d7 = args_

    EQ1 = (x1_ - x) ** 2 + (y1_ - y) ** 2 - (d1) ** 2
    EQ2 = (x2_ - x) ** 2 + (y2_ - y) ** 2 - (d2) ** 2
    EQ3 = (x3_ - x) ** 2 + (y3_ - y) ** 2 - (d3) ** 2
    EQ4 = (x4_ - x) ** 2 + (y4_ - y) ** 2 - (d4) ** 2
    EQ5 = (x5_ - x) ** 2 + (y5_ - y) ** 2 - (d5) ** 2
    EQ6 = (x6_ - x) ** 2 + (y6_ - y) ** 2 - (d6) ** 2
    EQ7 = (x7_ - x) ** 2 + (y7_ - y) ** 2 - (d7) ** 2
    return (EQ1, EQ2, EQ3, EQ4, EQ5, EQ6, EQ7)

def ls_7anchors(x1, x2, x3, x4, x5, x6, x7, y1, y2, y3, y4, y5, y6, y7, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los):
        x0 = (sum([x1, x2, x3, x4, x5, x6, x7]) / 7, sum([y1, y2, y3, y4, y5, y6, y7]) / 7)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los]
        res = least_squares(trilateration_7anchor, x0, args=(opti_args,), method='lm')
        loc_los = [res.x[0], res.x[1]]
        return loc_los

def trilateration_8anchor(guess_, args_):
    x, y = guess_
    x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_, x5_, y5_, x6_, y6_, x7_, y7_, x8_, y8_, d1, d2, d3, d4, d5, d6, d7, d8 = args_

    EQ1 = (x1_ - x) ** 2 + (y1_ - y) ** 2 - (d1) ** 2
    EQ2 = (x2_ - x) ** 2 + (y2_ - y) ** 2 - (d2) ** 2
    EQ3 = (x3_ - x) ** 2 + (y3_ - y) ** 2 - (d3) ** 2
    EQ4 = (x4_ - x) ** 2 + (y4_ - y) ** 2 - (d4) ** 2
    EQ5 = (x5_ - x) ** 2 + (y5_ - y) ** 2 - (d5) ** 2
    EQ6 = (x6_ - x) ** 2 + (y6_ - y) ** 2 - (d6) ** 2
    EQ7 = (x7_ - x) ** 2 + (y7_ - y) ** 2 - (d7) ** 2
    EQ8 = (x8_ - x) ** 2 + (y8_ - y) ** 2 - (d8) ** 2
    return (EQ1, EQ2, EQ3, EQ4, EQ5, EQ6, EQ7, EQ8)

def ls_8anchors(x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, y5, y6, y7, y8, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los, ttw_r8_los):
        x0 = (sum([x1, x2, x3, x4, x5, x6, x7, x8]) / 8, sum([y1, y2, y3, y4, y5, y6, y7, y8]) / 8)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los, ttw_r8_los]
        res = least_squares(trilateration_8anchor, x0, args=(opti_args,), method='lm')
        loc_los = [res.x[0], res.x[1]]
        return loc_los

# 定位次数
n = 100
# 基站个数
m = 39
# 目标点个数
p = 400
# 信号范围
r = 200
# 基站排布
BS7_1=[0,600];BS7_2=[200,600];BS7_3=[400,600];BS7_4=[600,600];BS7_5=[800,600];BS7_6=[1000,600]
BS6_1=[100,500];BS6_2=[300,500];BS6_3=[500,500];BS6_4=[700,500];BS6_5=[900,500]
BS5_1=[0,400];BS5_2=[200,400];BS5_3=[400,400];BS5_4=[600,400];BS5_5=[800,400];BS5_6=[1000,400]
BS4_1=[100,300];BS4_2=[300,300];BS4_3=[500,300];BS4_4=[700,300];BS4_5=[900,300]
BS3_1=[0,200];BS3_2=[200,200];BS3_3=[400,200];BS3_4=[600,200];BS3_5=[800,200];BS3_6=[1000,200]
BS2_1=[100,100];BS2_2=[300,100];BS2_3=[500,100];BS2_4=[700,100];BS2_5=[900,100]
BS1_1=[0,0];BS1_2=[200,0];BS1_3=[400,0];BS1_4=[600,0];BS1_5=[800,0];BS1_6=[1000,0]

MS = [[0]*2 for i in range(p)]
# 在范围内随机生成100个目标点
x_true = [0]*20
y_true = [0]*20
for i in range(20):
    x_true[i] = random.randrange(0, 1000, 1)
    for j in range(20):
        y_true[j] = random.randrange(0, 600, 1)
        MS[i*20+j] = [x_true[i], y_true[j]]
ms = np.array(MS)
# 正态分布的方差
std_var=0.1
# A为2列39行
A = [BS1_1,BS1_2,BS1_3,BS1_4,BS1_5,BS1_6,BS2_1,BS2_2,BS2_3,BS2_4,BS2_5,BS3_1,BS3_2,BS3_3,BS3_4,BS3_5,BS3_6,BS4_1,BS4_2,BS4_3,BS4_4,BS4_5,BS5_1,BS5_2,BS5_3,BS5_4,BS5_5,BS5_6,BS6_1,BS6_2,BS6_3,BS6_4,BS6_5,BS7_1,BS7_2,BS7_3,BS7_4,BS7_5,BS7_6]
A0 = np.array(A)

p_true = [[0]*2 for i in range(n*p)]
p_hat = [[0]*2 for i in range(n*p)]
rmse = [[0]*n for i in range(p)]
d = [[0]*m for i in range(p)]
#D = [[0]*m for i in range(p)]
D = [0]*m

# 生成n位列表
k = [0]*n
for i in range(n):
    k[i] = i

# 生成真实坐标数组
for t in range(p):
    for i in range(n):
        for j in range(2):
            if j == 0:
                p_true[t*n+i][j] = MS[t][0]
            else:
                p_true[t*n+i][j] = MS[t][1]
# # 生成3900个误差数组
# e = [0]*m*p
# for i in range(3510):
#     e[i] = np.random.normal(0 , std_var)
# for i in range(195):
#     e[3510+i] = np.random.normal(0.1,0.1) + np.random.gamma(2,3.5) + 0.121
# for i in range(22):
#     e[3510+195+i] = err_w(0.47,6)
# for i in range(22):
#     e[3510+195+22+i] = err_w(0.47,9)
# for i in range(22):
#     e[3510+195+22+22+i] = err_w(0.47, 7.5)
# for i in range(22):
#     e[3510 + 195 +22 +22 +22+ i] = err_w(0.27, 6)
# for i in range(22):
#     e[3510 + 195 +22 +22 +22 +22 + i] = err_w(0.27, 9)
# for i in range(22):
#     e[3510 + 195 +22 +22 +22+ 22 + 22 + i] = err_w(0.27, 7.5)
# for i in range(21):
#     e[3510 + 195 +22 +22 +22+ 22 + 22 +22+ i] = err_w(0.32, 6)
# for i in range(21):
#     e[3510 + 195 +22 +22 +22+ 22 + 22 +22+21+ i] = err_w(0.32, 9)
# for i in range(21):
#     e[3510 + 195 +22 +22 +22+ 22 + 22 +22+21+21+ i] = err_w(0.32, 7.5)
# random.shuffle(e)


w = [0.47,0.27,0.32]
sigma = [6,9,7.5]
# 生成390000个误差数组
e = [0]*m*p*n
num_e_los = int(m*p*0.9)
num_e_nlos = int(m*p*0.05)

# for k in range(n):
#     for i in range(m*p*0.9):
#         e[m*p*k+i] = np.random.normal(0 , std_var)
#     for i in range(m*p*0.05):
#         e[m*p*k+m*p*0.9+i] = np.random.normal(0.1,0.1) + np.random.gamma(2,3.5) + 0.121
#     for i in range(22):
#         e[m*p*k+3510+195+i] = err_w(0.47,6)
#     for i in range(22):
#         e[m*p*k+3510+195+22+i] = err_w(0.47,9)
#     for i in range(22):
#         e[m*p*k+3510+195+22+22+i] = err_w(0.47, 7.5)
#     for i in range(22):
#         e[m*p * k + 3510 + 195 +22 +22 +22+ i] = err_w(0.27, 6)
#     for i in range(22):
#         e[m*p * k + 3510 + 195 +22 +22 +22 +22 + i] = err_w(0.27, 9)
#     for i in range(22):
#         e[m*p * k + 3510 + 195 +22 +22 +22+ 22 + 22 + i] = err_w(0.27, 7.5)
#     for i in range(21):
#         e[m*p * k + 3510 + 195 +22 +22 +22+ 22 + 22 +22+ i] = err_w(0.32, 6)
#     for i in range(21):
#         e[m*p * k + 3510 + 195 +22 +22 +22+ 22 + 22 +22+21+ i] = err_w(0.32, 9)
#     for i in range(21):
#         e[m*p * k + 3510 + 195 +22 +22 +22+ 22 + 22 +22+21+21+ i] = err_w(0.32, 7.5)
for k in range(n):
    for i in range(num_e_los):
        e[m*p*k+i] = np.random.normal(0 , std_var)
    for i in range(num_e_nlos):
        e[m*p*k+num_e_los+i] = np.random.normal(0.1,0.1) + np.random.gamma(2,3.5) + 0.121
    for i in range(num_e_nlos):
        w0 = random.choice(w)
        sigma0 = random.choice(sigma)
        e[m*p*k+num_e_los+num_e_nlos+i] = err_w(w0,sigma0)
random.shuffle(e)

# 计算真实距离
for t in range(p):
    for i in range(m):
        d[t][i]=distance(A[i][0],A[i][1],MS[t][0],MS[t][1])

# 计算p*n个定位坐标
for t in range(p):
    for i in range(n):
        a = 0
        for j in range(m):
            dis = d[t][j]+e[t*n*m+i*m+j]
            if(dis<r):
                D[j] = dis
                a = a+1
            else:
                D[j] = 0

        A_eff = [0]*a
        D_eff = [0]*a
        b = 0
        for q in range(m):
            if (D[q] != 0):
                A_eff[b] = A[q]
                D_eff[b] = D[q]
                b = b + 1
        # 解算得到定位节点坐标
        if(a==3):
            p_hat[t*n+i] = ls_3anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],D_eff[0], D_eff[1], D_eff[2])
        if(a==4):
            p_hat[t*n+i] = ls_4anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],A_eff[3][1],D_eff[0], D_eff[1], D_eff[2],D_eff[3])
        if(a==5):
            p_hat[t*n+i] = ls_5anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],A_eff[3][1],A_eff[4][1],D_eff[0], D_eff[1], D_eff[2],D_eff[3],D_eff[4])
        if(a==6):
            p_hat[t*n+i] = ls_6anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],A_eff[5][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],A_eff[3][1],A_eff[4][1],A_eff[5][1],D_eff[0], D_eff[1], D_eff[2],D_eff[3],D_eff[4],D_eff[5])
        if(a==7):
            p_hat[t*n+i] = ls_7anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],A_eff[5][0],A_eff[6][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],A_eff[3][1],A_eff[4][1],A_eff[5][1],A_eff[6][1],D_eff[0], D_eff[1], D_eff[2],D_eff[3],D_eff[4],D_eff[5],D_eff[6])
        if(a==8):
            p_hat[t*n+i] = ls_8anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],A_eff[5][0],A_eff[6][0],A_eff[7][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],A_eff[3][1],A_eff[4][1],A_eff[5][1],A_eff[6][1],A_eff[7][1],D_eff[0], D_eff[1], D_eff[2],D_eff[3],D_eff[4],D_eff[5],D_eff[6],D_eff[7])
        # if(a==9):
        #     p_hat[t*n+i] = ls_9anchors(A[0][0], A[0][1], D[t][0],A[1][0], A[1][1], D[t][1],A[2][0], A[2][1], D[t][2])
# rmse存放每一次定位的RMSE
for t in range(p):
    for i in range(n):
        rmse[t][i] = root_mean_squared_error(p_hat[t*n+i:t*n+i+1],p_true[t*n+i:t*n+i+1])

# g存放各个目标点的RMSE
g = [0]*p
for t in range(p):
    g[t] = root_mean_squared_error(p_hat[t*n:t*n+99],p_true[t*n:t*n+99])
    print("坐标",MS[t],"的RMSE为",g[t])

l = [[0]*20 for i in range(20)]
for i in range(20):
    for j in range(20):
        l[j][i] = g[i*20+j]

# 计算RMSE
RMSE = root_mean_squared_error(p_hat,p_true)
print("总RMSE为",RMSE)





# # CDF图
# kwargs = {'cumulative': True}
# sns.distplot(rmse, hist_kws=kwargs, kde_kws=kwargs)
# plt.title('方差10cm，基站下三角分布的误差累计分布图')
# plt.xlabel('误差，单位：m')
# plt.xlim(-0.2,5)
# #用来正常显示中文标签
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.show()




# # RMSE与测试次数的关系图
# #通用设置
# matplotlib.rc('axes', facecolor = 'white')
# matplotlib.rc('figure', figsize = (6, 4))
# matplotlib.rc('axes', grid = False)
# #数据及线属性
# plt.plot(k, rmse,color='red', linestyle='-')
# #标题设置
# plt.title('RMSE与测试次数的关系')
# plt.xlabel('测试次数')
# plt.ylabel('RMSE，单位：m')
# #用来正常显示中文标签
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.show()



# contour等高图
x = x_true
y = y_true
plt.xlim(0,1000)
plt.ylim(0,600)
plt.xlabel('x坐标轴')
plt.ylabel('y坐标轴')
# 等高线梯度
z = np.linspace(0.6, 2, 20)
C = plt.contourf(x,y,l,z)
# 绘制基站坐标
x_point = A0[:,0]
y_point = A0[:,1]
plt.scatter(x_point,y_point)
#plt.scatter(ms[:,0],ms[:,1])
plt.clabel(C, inline=True, fontsize=10)
plt.grid()
# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.show()

# # contour等高图
# x = ms[:,0]
# y = ms[:,1]
# plt.xlim(0,1000)
# plt.ylim(0,600)
# plt.xlabel('x坐标轴')
# plt.ylabel('y坐标轴')
# # 等高线梯度
# z = np.linspace(0, 2, 100)
# C = plt.contour(x,y,l,z)
# # 绘制基站坐标
# x_point = A0[:,0]
# y_point = A0[:,1]
# plt.scatter(x_point,y_point)
# #第一个参数为标记文本，第二个参数为标记对象的坐标，第三个参数为标记位置
# # plt.annotate('BS1', xy=BS1, xytext=(1,1))
# # plt.annotate('BS2', xy=BS2, xytext=(51,1))
# # plt.annotate('BS3', xy=BS3, xytext=(51,51))
# plt.clabel(C, inline=True, fontsize=10)
# plt.grid()
# #用来正常显示中文标签
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.show()

# #用来正常显示指数上的负号-
# mpl.rcParams.update(
# {
#     'text.usetex': False,
#     'font.family': 'stixgeneral',
#     'mathtext.fontset': 'stix',
# }
# )


# def draw_hist(e, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
#     plt.hist(e, 1000)
#     plt.xlabel(Xlabel)
#     plt.xlim(Xmin, Xmax)
#     plt.ylabel(Ylabel)
#     plt.ylim(Ymin, Ymax)
#     plt.title(Title)
#     plt.show()
#
#
# draw_hist(e, '在有NLOS情况下的测距误差分布', '频次', '误差', -1, 5, 0, 60000)  # 直方图展示


# u1 = 0.1  # 第一个高斯分布的均值
# sigma1 = 0.1  # 第一个高斯分布的标准差
# x = np.arange(-1, 5, 0.001)
# # 表示第一个高斯分布函数
# y = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决pythonmatplotlib绘图无法显示中文的问题
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.plot(x, y, 'b-', linewidth=2)
# plt.hist(e, 1000)
# plt.xlabel('误差')
# plt.xlim(-1, 5)
# plt.ylabel('频次')
# plt.ylim(0, 60000)
# plt.title('在有NLOS情况下的测距误差分布')
# plt.show()

# u1 = 0  # 第一个高斯分布的均值
# sigma1 = 0.1  # 第一个高斯分布的标准差
# x = np.arange(-20, 20, 0.01)
# # 表示第一个高斯分布函数
# y = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))
# # 设置两种绘图颜色
# c1='r'
# c2='b'
# lns=[] # 用于存储绘图句柄以合并图例的list
# # 创建画布并设置大小
# fig=plt.figure()
# # 通过 add_subplot 方式创建两个坐标轴，相当于在同一个子图上叠加了两对坐标系
# ax=fig.add_subplot(111, label="1")
# ax2=fig.add_subplot(111, label="2", frame_on=False)
# # 绘制图1并将绘图句柄返回，以便添加合并图例
# lns1=ax.plot(x,y,color=c1,label=c1)
# lns=lns1
# lns2=ax2.hist(e, 1000)
# lns+=lns2
# """图形美化"""
# # 调整第二对坐标轴的label和tick位置，以实现双X轴双Y轴效果
# ax2.xaxis.tick_top()
# ax2.yaxis.tick_right()
# ax2.xaxis.set_label_position('top')
# ax2.yaxis.set_label_position('right')
# plt.tight_layout()
# plt.show()