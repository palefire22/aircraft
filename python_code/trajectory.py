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

def draw_segment(start_point, angle, length):

    x0, y0 = start_point # начальная точка

    length = length * (-angle) #настройка длины

    # конечная точка
    x1 = x0 + length * np.cos(angle)
    y1 = y0 + length * np.sin(angle)

    plt.plot([x0, x1], [y0, y1], marker='o')

    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')

def draw(x, y, q, i):
    plt.plot(x, y, label='y(x) при q0 = ' + str(round(x0[i], 2)))
    point = [x[-1], y[-1]]
    draw_segment(point, q, 5)
    plt.title('Траектория ЛА при q0 =' + str(round(x0[i], 2)))
    plt.xlabel('Координата x, м')
    plt.ylabel('Координата y, м')
    plt.legend()
    plt.grid()
    plt.show()




x0_grad = [2.8, 6.8, 18.1, 23.4, 39.2, 48.6]  # начальное значение угла наклона траектории в градусах
x0 = list(map(lambda x: math.radians(x), x0_grad))  # начальное значение угла наклона траектории в радианах
x_end_grad = [-22, -30.9, -36.3, -44.5, -63.2, -64.6]  # конечное значение угла наклона траектории в градусах
x_end = list(map(lambda x: math.radians(x), x_end_grad))  # конечное значение угла наклона траектории в радианах
x_grand_massive = file_get(x0, 'x') # массив массивов для q
y_grand_massive = file_get(x0, 'y') # массив массивов для v



length = len(x0)
for i in range(0, length):
    draw(x_grand_massive[i], y_grand_massive[i], x_end[i], i)