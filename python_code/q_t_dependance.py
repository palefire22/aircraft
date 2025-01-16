import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
#import scipy as sp
import math


#Получает значения из всех файлов, которые определяет pointer(файл q или v) и x0[i] элемент, возвращает двумерный массив типа float
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

#Непосредственно сама функция, принимает угол q=x и его индекс в массиве
def f(x,i, j):  # непосредственно сама функция
    cxa = 0.1  # коэфф лобового сопротивления
    cya = 0.1  # коэфф подъемной силы
    p = 1.23  # плотность воздуха (кг/м3)
    s = 128 * (10 ** (-4)) * 2  # площадь крыла(см2) * (10**(-4)) * 2 шт.
    m = 0.03 #масса самолета
    gg = m * 9.815  # масса самолета(кг) * уск. своб падения
    v = v_grand_massive[j][i]

    funct = ((-1) * cya * p *  s / 2 * (v**2) - gg * math.cos(x))  / (m * v)# значение функции в точке
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
def runge_kutta_init(f, y0, grand_massive, i):
    x, y = runge_kutta(f, 0, q_grand_massive[i], i)
    plt.plot(y, x, label='q(t) при q0 = ' + str(round(x0[i], 2)))
    plt.title('Решение dq/dt = f(q) при q0 =' + str(round(x0[i], 2)))
    plt.xlabel('Время полёта t, с')
    plt.ylabel('Угол наклона траектории q, рад')
    plt.legend()
    plt.grid()
    plt.show()
    return y, x

#Выводит в файлы с названиями q_if_q0_equals... и v_if_q0_equals... точку q[i] и v[i] соотв.
def file_output(x, y, i):
    xfile_name = 'stat/t_if_q0_equals' + str(round(x0[i], 2))
    yfile_name = 'stat/q_if_q0_equals' + str(round(x0[i], 2))
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
q_grand_massive = file_get(x0, 'q') # массив массивов для q
v_grand_massive = file_get(x0, 'v') # массив массивов для v
######


length = len(x0)
for i in range(0, length):
    x, y = runge_kutta_init(f, 0, q_grand_massive[i], i)
    file_output(x, y, i)


