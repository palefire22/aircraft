import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
#import scipy as sp
import math

def file_get(x0, pointer):
    grand_massive = []
    for q in x0:
        file_name = 'stat/'+ pointer + '_if_q0_equals' + str(round(q, 2))
        with open(file_name, 'r') as file:
            q = []
            for number in file:
                q.append(float(number[:-1]))
        grand_massive.append(q)
        file.close()
    return grand_massive

#непосредственно сама функция

def fy(x, i, j):
    v = v_grand_massive[j][i]
    q = q_grand_massive[j][i]
    funct = v * math.sin(q)  # значение функции в точке
    return funct

#Непосредственно сама функция, принимает угол q=x и его индекс в массиве
def fx(x,i, j):  # непосредственно сама функция
    v = v_grand_massive[j][i]
    q = q_grand_massive[j][i]
    funct = v * math.cos(q)# значение функции в точке
    return funct

#Видоизмененный под наши нужды метод Рунге-Кутта 4 порядка
def runge_kutta(f, y0, x0, j):
    #y0 = 0
    n = len(x0)
    x = x0
    y = np.zeros(n)
    y[0] = y0

    for i in range(1, n):
        h = x[i] - x[i-1]
        k1 = h * f(x[i - 1], i - 1, j)
        k2 = h * f(x[i - 1] + h / 2, i - 1, j)
        k3 = h * f(x[i - 1] + h / 2, i - 1, j)
        k4 = h * f(x[i - 1] + h, i - 1, j)
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y

# Запускает метод рунге-кутта для выданных значений, затем строит график решения q(t)
# !!!!Меняет координатные оси, т.е. по x будет t и возвращает массивы в порядке 'параметр' 'значение'
def runge_kutta_init(f, y0, grand_massive, i, pointer):
    x, y = runge_kutta(f, 0, t_grand_massive[i], i)
    plt.plot(x, y, label= pointer + '(t) при q0 = ' + str(round(x0[i], 2)))
    plt.title('Решение d' + pointer + '/dt = f(t) при q0 =' + str(round(x0[i], 2)))
    plt.xlabel('Время полёта t, с')
    plt.ylabel('Координата ' + pointer + ', м')
    plt.legend()
    plt.grid()
    plt.show()
    return x, y

def file_output(x, y, i):
    xfile_name = 'stat/x_if_q0_equals' + str(round(x0[i], 2))
    yfile_name = 'stat/y_if_q0_equals' + str(round(x0[i], 2))
    with open(xfile_name, "w") as xfile:
        for i in x:
            xfile.write(str(i) + '\n')
    with open(yfile_name, "w") as yfile:
        for j in y:
            yfile.write(str(j) + '\n')

    xfile.close()
    yfile.close()

#Начальные условия
y0 = 3.5  # начальное значение скорости(м/c)
x0_grad = [2.8, 6.8, 18.1, 23.4, 39.2, 48.6]  # начальное значение угла наклона траектории в градусах
x_end_grad = [-22, -30.9, -36.3, -44.5, -63.2, -64.6]  # конечное значение угла наклона траектории в градусах
x0 = list(map(lambda x: math.radians(x), x0_grad))  # начальное значение угла наклона траектории в радианах
x_end = list(map(lambda x: math.radians(x), x_end_grad))  # конечное значение угла наклона траектории в радианах
q_grand_massive = file_get(x0, 'q') #массив массивов для q
t_grand_massive = file_get(x0, 't') #массив массивов для t
v_grand_massive = file_get(x0, 'v') #массив массивов для t
######

length = len(x0)
for i in range(0, length):
    x1, y1 = runge_kutta_init(fx, 0, t_grand_massive[i], i, 'x')
    x2, y2 = runge_kutta_init(fy, 0, t_grand_massive[i], i, 'y')
    file_output(y1, y2, i)


