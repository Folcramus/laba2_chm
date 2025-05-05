import numpy as np
import math
import matplotlib.pyplot as plt

# === Параметры задачи ===
a = -2        # Левая граница интегрирования по x
b = 2         # Правая граница интегрирования по x
c = 2         # Начальное значение параметра t
d = 3         # Конечное значение параметра t
m = 15        # Количество интервалов для t (разбиение отрезка [c, d])
epsilon = 0.001  # Погрешность для метода удвоения

# === Определение подынтегральной функции f(x, t) ===
def f(x, t):
    # Функция e^(sqrt(t) / (1 + x^2))
    return math.exp(math.sqrt(t) / (1 + x ** 2))

# === Основная функция F(t) = ∫[a,b] f(x,t) dx ===
def F(t, method, N=None):
    # Выбор метода численного интегрирования
    if method == 'gauss3':
        return gauss_integration(t, 3)
    elif method == 'gauss4':
        return gauss_integration(t, 4)
    elif method in ['rect', 'trap', 'simp']:
        return doubling_method(t, method)  # Метод с удвоением числа разбиений
    else:
        raise ValueError("Unknown method")  # Ошибка при неизвестном методе

# === Квадратурные формулы Гаусса ===
def gauss_integration(t, n):
    # Узлы и веса для стандартного интервала [-1, 1]
    if n == 3:
        nodes = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
        weights = [5 / 9, 8 / 9, 5 / 9]
    elif n == 4:
        nodes = [-0.861136, -0.339981, 0.339981, 0.861136]
        weights = [0.347855, 0.652145, 0.652145, 0.347855]

    # Перенос узлов на интервал [a, b]
    transformed_nodes = [(b - a) / 2 * xi + (a + b) / 2 for xi in nodes]
    transformed_weights = [(b - a) / 2 * wi for wi in weights]

    # Вычисление приближенного значения интеграла
    integral = 0.0
    for xi, wi in zip(transformed_nodes, transformed_weights):
        integral += wi * f(xi, t)

    return integral

# === Метод удвоения числа шагов ===
def doubling_method(t, method):
    # Начальное количество отрезков (четное для Симпсона)
    N = 2 if method != 'simp' else 4
    prev_integral = compute_integral(t, N, method)

    # Удвоение числа шагов до достижения заданной точности
    while True:
        N *= 2
        current_integral = compute_integral(t, N, method)

        if abs(current_integral - prev_integral) < epsilon:
            break  # Выход из цикла при достижении точности

        prev_integral = current_integral  # Обновление предыдущего значения

    return current_integral, N  # Возврат значения и количества отрезков

# === Вычисление интеграла с фиксированным числом отрезков N ===
def compute_integral(t, N, method):
    h = (b - a) / N  # Шаг интегрирования
    integral = 0.0

    if method == 'rect':  # Метод прямоугольников (средних)
        for i in range(N):
            xi = a + (i + 0.5) * h
            integral += f(xi, t)
        integral *= h

    elif method == 'trap':  # Метод трапеций
        integral = (f(a, t) + f(b, t)) / 2
        for i in range(1, N):
            xi = a + i * h
            integral += f(xi, t)
        integral *= h

    elif method == 'simp':  # Метод Симпсона
        if N % 2 != 0:
            raise ValueError("Для метода Симпсона N должно быть четным")

        integral = f(a, t) + f(b, t)
        for i in range(1, N):
            xi = a + i * h
            if i % 2 == 1:
                integral += 4 * f(xi, t)
            else:
                integral += 2 * f(xi, t)
        integral *= h / 3

    return integral  # Возврат значения интеграла

# === Вычисление значений F(t_j) на равномерной сетке ===
tau = (d - c) / m  # Шаг по t
results = []  # Список для хранения результатов

for j in range(m + 1):
    tj = c + j * tau  # Очередная точка t_j

    # Вычисление F(t_j) по методу Симпсона с удвоением шагов
    F_simp, N_simp = doubling_method(tj, 'simp')

    # Вычисление F(t_j) по формуле Гаусса (3 и 4 узла)
    F_gauss3 = gauss_integration(tj, 3)
    F_gauss4 = gauss_integration(tj, 4)

    # Сохранение результата
    results.append((tj, F_simp, N_simp, F_gauss3, F_gauss4))

# === Вывод таблицы с результатами ===
print("Результаты вычисления интеграла F(t) = ∫[a,b] f(x,t) dx")
print(f"Параметры: a={a}, b={b}, c={c}, d={d}, m={m}, ε={epsilon}")
print()
print("{:>5s} {:>10s} {:>10s} {:>16s} {:>12s}".format(
    "t_j", "F(t_j) Симпсон", "N", "F(t_j) Гаусс-3", "F(t_j) Гаусс-4"))
print("-" * 60)

for tj, F_simp, N, F_g3, F_g4 in results:
    print("{:8.3f} {:12.6f} {:8d} {:12.6f} {:12.6f}".format(
        tj, F_simp, N, F_g3, F_g4))

# === Построение графика зависимости F(t) от t ===

# Извлечение данных по методам
t_values = [row[0] for row in results]        # Значения t_j
F_simp_values = [row[1] for row in results]   # Значения F(t_j) по Симпсону
F_gauss3_values = [row[3] for row in results] # Значения по Гауссу 3 узла
F_gauss4_values = [row[4] for row in results] # Значения по Гауссу 4 узла

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(t_values, F_simp_values, label="Симпсон (удвоение)", marker='o')
plt.plot(t_values, F_gauss3_values, label="Гаусс 3 узла", linestyle='--', marker='s')
plt.plot(t_values, F_gauss4_values, label="Гаусс 4 узла", linestyle='-.', marker='^')

# Оформление графика
plt.title("График F(t) = ∫[a,b] f(x,t) dx")  # Заголовок
plt.xlabel("t")                             # Подпись оси X
plt.ylabel("F(t)")                          # Подпись оси Y
plt.grid(True)                              # Сетка
plt.legend()                                # Легенда
plt.tight_layout()                          # Подгонка элементов
plt.show()                                  # Показ графика
