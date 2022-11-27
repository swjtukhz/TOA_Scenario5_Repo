import matplotlib
import numpy as np
import math
import sympy
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import random

# 固定随机值
random.seed(1)
np.random.seed(1)
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
def err_w(w,epsr):
    err = w*(math.sqrt(epsr)-1)+np.random.normal(0,0.15)
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

def ls_6anchors(x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los,
                ttw_r5_los, ttw_r6_los):
        x0 = (sum([x1, x2, x3, x4, x5, x6]) / 6, sum([y1, y2, y3, y4, y5, y6]) / 6)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, ttw_r1_los, ttw_r2_los, ttw_r3_los, ttw_r4_los,
                     ttw_r5_los, ttw_r6_los]
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

def ls_7anchors(x1, x2, x3, x4, x5, x6, x7, y1, y2, y3, y4, y5, y6, y7, ttw_r1_los, ttw_r2_los, ttw_r3_los,
                ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los):
        x0 = (sum([x1, x2, x3, x4, x5, x6, x7]) / 7, sum([y1, y2, y3, y4, y5, y6, y7]) / 7)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, ttw_r1_los, ttw_r2_los, ttw_r3_los,
                     ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los]
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

def ls_8anchors(x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, y5, y6, y7, y8, ttw_r1_los, ttw_r2_los,
                ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los, ttw_r8_los):
        x0 = (sum([x1, x2, x3, x4, x5, x6, x7, x8]) / 8, sum([y1, y2, y3, y4, y5, y6, y7, y8]) / 8)
        opti_args = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, ttw_r1_los, ttw_r2_los,
                     ttw_r3_los, ttw_r4_los, ttw_r5_los, ttw_r6_los, ttw_r7_los, ttw_r8_los]
        res = least_squares(trilateration_8anchor, x0, args=(opti_args,), method='lm')
        loc_los = [res.x[0], res.x[1]]
        return loc_los

# 定位次数
n = 10
# 基站个数
m = 40
# MS随机坐标个数
p = 1600
# 信号最大传播范围
r = 200
# 基站排布
BS8_1=[0,600];BS8_2=[150,600];BS8_3=[300,600];BS8_4=[450,600];BS8_5=[600,600];BS8_6=[750,600];BS8_7=[900,600];BS8_8=[1000,600]
BS7_1=[0,519];BS7_2=[300,519];BS7_3=[600,519];BS7_4=[900,519]
BS6_1=[150,432];BS6_2=[450,432];BS6_3=[750,259];BS6_4=[1000,259]
BS5_1=[0,346];BS5_2=[300,346];BS5_3=[600,346];BS5_4=[900,346]
BS4_1=[150,259];BS4_2=[450,259];BS4_3=[750,259];BS4_4=[1000,259]
BS3_1=[0,173];BS3_2=[300,173];BS3_3=[600,173];BS3_4=[900,173]
BS2_1=[150,86];BS2_2=[450,86];BS2_3=[750,86];BS2_4=[1000,86]
BS1_1=[0,0];BS1_2=[150,0];BS1_3=[300,0];BS1_4=[450,0];BS1_5=[600,0];BS1_6=[750,0];BS1_7=[900,0];BS1_8=[1000,0]

MS = [[0]*2 for i in range(p)]

# # 在范围内随机生成100个目标点
# for i in range(100):
#     x_true = random.randrange(0, 1000, 1)
#     y_true = random.randrange(0, 600, 1)
#     MS[i] = [x_true, y_true]
x_true = np.linspace(0,1000,40)
y_true = np.linspace(0,600,40)
for i in range(40):
    for j in range(40):
        MS[i*40+j] = [x_true[i], y_true[j]]
ms = np.array(MS)
# 正态分布的方差
std_var=0.1

# A为2列40行
A = [BS1_1,BS1_2,BS1_3,BS1_4,BS1_5,BS1_6,BS1_7,BS1_8,BS2_1,BS2_2,BS2_3,BS2_4,BS3_1,BS3_2,BS3_3,BS3_4,BS4_1,BS4_2,BS4_3,BS4_4,
     BS5_1,BS5_2,BS5_3,BS5_4,BS6_1,BS6_2,BS6_3,BS6_4,BS7_1,BS7_2,BS7_3,BS7_4,BS8_1,BS8_2,BS8_3,BS8_4,BS8_5,BS8_6,BS8_7,BS8_8]
A0 = np.array(A)

p_true = [[0]*2 for i in range(n*p)]
p_hat = [[0]*2 for i in range(n*p)]
rmse = [[0]*n for i in range(p)]
d = [[0]*m for i in range(p)]
#D = [[0]*m for i in range(p)]
D = [0]*m

# # 生成n位列表
# k = [0]*n
# for i in range(n):
#     k[i] = i

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
eps = [6, 9, 7.5]
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
        eps0 = random.choice(eps)
        e[m*p*k+num_e_los+num_e_nlos+i] = err_w(w0,eps0)
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
            p_hat[t*n+i] = ls_3anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[0][1], A_eff[1][1],
                                       A_eff[2][1],D_eff[0], D_eff[1], D_eff[2])
        if(a==4):
            p_hat[t*n+i] = ls_4anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[0][1],
                                       A_eff[1][1], A_eff[2][1],A_eff[3][1],D_eff[0], D_eff[1], D_eff[2],D_eff[3])
        if(a==5):
            p_hat[t*n+i] = ls_5anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],
                                       A_eff[0][1], A_eff[1][1], A_eff[2][1],A_eff[3][1],A_eff[4][1],
                                       D_eff[0], D_eff[1], D_eff[2],D_eff[3],D_eff[4])
        if(a==6):
            p_hat[t*n+i] = ls_6anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],
                                       A_eff[5][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],A_eff[3][1],
                                       A_eff[4][1],A_eff[5][1],D_eff[0], D_eff[1], D_eff[2],D_eff[3],D_eff[4],D_eff[5])
        if(a==7):
            p_hat[t*n+i] = ls_7anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],
                                       A_eff[5][0],A_eff[6][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],
                                       A_eff[3][1],A_eff[4][1],A_eff[5][1],A_eff[6][1],D_eff[0], D_eff[1],
                                       D_eff[2],D_eff[3],D_eff[4],D_eff[5],D_eff[6])
        if(a==8):
            p_hat[t*n+i] = ls_8anchors(A_eff[0][0], A_eff[1][0], A_eff[2][0],A_eff[3][0],A_eff[4][0],
                                       A_eff[5][0],A_eff[6][0],A_eff[7][0],A_eff[0][1], A_eff[1][1], A_eff[2][1],
                                       A_eff[3][1],A_eff[4][1],A_eff[5][1],A_eff[6][1],A_eff[7][1],D_eff[0], D_eff[1],
                                       D_eff[2],D_eff[3],
            D_eff[4],D_eff[5],D_eff[6],D_eff[7])
        # if(a==9):
        #     p_hat[t*n+i] = ls_9anchors(A[0][0], A[0][1], D[t][0],A[1][0], A[1][1], D[t][1],A[2][0], A[2][1], D[t][2])
# rmse存放每一次定位的RMSE
for t in range(p):
    for i in range(n):
        rmse[t][i] = root_mean_squared_error(p_hat[t*n+i:t*n+i+1],p_true[t*n+i:t*n+i+1])
        #print("第",t,"个点的第",i,"次rmse为",rmse[t][i])
# g存放各个目标点的RMSE
g = [0]*p
for t in range(p):
    g[t] = root_mean_squared_error(p_hat[t*n:t*n+99],p_true[t*n:t*n+99])
    print("坐标",MS[t],"的RMSE为",g[t])

l = [[0]*40 for i in range(40)]
for i in range(40):
    for j in range(40):
        l[j][i] = g[i*40+j]
#
# 计算RMSE
RMSE = root_mean_squared_error(p_hat,p_true)
print("总RMSE为",RMSE)

# contour等高图
x = x_true
y = y_true
plt.xlim(0,1000)
plt.ylim(0,600)
plt.xlabel('x坐标轴')
plt.ylabel('y坐标轴')
plt.title('RMSE等高图')
# 等高线梯度
z = np.linspace(0.6, 2.5, 100)
C = plt.contourf(x,y,l,z)
# 绘制基站坐标
x_point = A0[:,0]
y_point = A0[:,1]
plt.scatter(x_point,y_point,color='red')
#plt.scatter(ms[:,0],ms[:,1])
#第一个参数为标记文本，第二个参数为标记对象的坐标，第三个参数为标记位置
# plt.annotate('BS1', xy=BS1, xytext=(1,1))
# plt.annotate('BS2', xy=BS2, xytext=(51,1))
# plt.annotate('BS3', xy=BS3, xytext=(51,51))
plt.clabel(C, inline=True, fontsize=10)
plt.grid()
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.show()