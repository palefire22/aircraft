import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
#import scipy as sp
import math


def f(x, y):  # непосредственно сама функция
    cxa = 1  # коэфф лобового сопротивления
    cya = 1  # коэфф подъемной силы
    p = 1.25  # плотность воздуха (кг/м3)
    s = 128 * (10 ** (-4)) * 2  # площадь крыла(см2) * (10**(-4)) * 2 шт.
    gg = 0.03 * 9.815  # масса самолета(кг) * уск. своб падения
    funct = y * (cxa * p * s / 2 * (y ** 2) + gg * math.sin(x)) / (
                cya * p * s / 2 * y ** 2 + gg * math.cos(x))  # значение функции в точке
    return funct


# метод Рунге-Кутты 4-го порядка
def runge_kutta(f, y0, x0, x_end, h):
    n = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n)
    y = np.zeros(n)
    y[0] = y0

    for i in range(1, n):
        k1 = h * f(x[i - 1], y[i - 1])
        k2 = h * f(x[i - 1] + h / 2, y[i - 1] + k1 / 2)
        k3 = h * f(x[i - 1] + h / 2, y[i - 1] + k2 / 2)
        k4 = h * f(x[i - 1] + h, y[i - 1] + k3)
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y


# Запускает метод рунге-кутта для выданных значений, затем строит график решения
def runge_kutta_init(f, yo, x0, x_end, h):
    x, y = runge_kutta(f, y0, x0, x_end, h)

    # построение графика
    plt.plot(x, y, label='v(q); q0 = ' + str(round(x0, 2)) + 'рад')
    plt.title('Решение дифференциального уравнения dv/dq = f(v,q) при q = ' + str(round(x0, 2)) + 'рад')
    plt.xlabel('Угол наклона траектории q, рад')
    plt.ylabel('Модуль скорости ЛА v, м/c')
    plt.legend()
    plt.grid()
    plt.show()
    return x, y

#Выводит в файлы с названиями q_if_q0_equals... и v_if_q0_equals... точку q[i] и v[i] соотв.
def file_input(x, y, i):
    xfile_name = 'q_if_q0_equals' + str(round(x0[i], 2))
    yfile_name = 'v_if_q0_equals' + str(round(x0[i], 2))
    with open(xfile_name, "w") as xfile:
        for i in x:
            xfile.write(str(i) + '\n')
    with open(yfile_name, "w") as yfile:
        for j in y:
            yfile.write(str(j) + '\n')

    xfile.close()
    yfile.close()

# начальные условия
y0 = 3.5  # начальное значение скорости(м/c)
x0_grad = [2.8, 6.8, 18.1, 23.4, 39.2, 48.6]  # начальное значение угла наклона траектории в градусах
x_end_grad = [-22, -30.9, -36.3, -44.5, -63.2, -64.6]  # конечное значение угла наклона траектории в градусах
h = -0.01  # шаг

x0 = list(map(lambda x: math.radians(x), x0_grad))  # начальное значение угла наклона траектории в радианах
x_end = list(map(lambda x: math.radians(x), x_end_grad))  # конечное значение угла наклона траектории в радианах

length = len(x0_grad)
for i in range(0, length):
    x, y = runge_kutta_init(f, y0, x0[i], x_end[i], h)
    file_input(x, y, i) #Вывод в txt файлы всех точек
