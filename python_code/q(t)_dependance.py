import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
#import scipy as sp
import math


#Получает значения из всех файлов, которые определяет pointer(файл q или v) и x[i] элемент, возвращает двумерный массив типа float
def file_get(x, pointer):
    grand_massive = []
    for q in x:
        file_name = 'q(v)_stat/'+ pointer + '_if_q0_equals' + str(round(q, 2))
        with open(file_name, 'r') as file:
            q = []
            for number in file:
                q.append(float(number[:-1]))
        grand_massive.append(q)
        file.close()
    return grand_massive

#Начальные условия
y0 = 3.5  # начальное значение скорости(м/c)
x0_grad = [2.8, 6.8, 18.1, 23.4, 39.2, 48.6]  # начальное значение угла наклона траектории в градусах
x_end_grad = [-22, -30.9, -36.3, -44.5, -63.2, -64.6]  # конечное значение угла наклона траектории в градусах
x0 = list(map(lambda x: math.radians(x), x0_grad))  # начальное значение угла наклона траектории в радианах
x_end = list(map(lambda x: math.radians(x), x_end_grad))  # конечное значение угла наклона траектории в радианах
q_grand_massive = file_get(x0, 'q') #массив массивов для q
v_grand_massive = file_get(x0, 'v') #массив массивов для v
######








x = q_grand_massive[5]
y = v_grand_massive[5]

plt.plot(x, y, label='v(q); q0 = ' + str(round(x0[5], 2)) + 'рад')
plt.title('Решение дифференциального уравнения dv/dq = f(v,q) при q = ' + str(round(x0[5], 2)) + 'рад')
plt.xlabel('Угол наклона траектории q, рад')
plt.ylabel('Модуль скорости ЛА v, м/c')
#plt.legend()
plt.grid()
plt.show()

