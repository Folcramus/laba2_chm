import numpy as np
import math

# Параметры задачи
a = -2
b = 2
c = 2
d = 3
m = 15
epsilon = 0.001


# Функция f(x, t)
def f(x, t):
    return math.exp(math.sqrt(t) / (1 + t ** 2))


# Функция F(t) = ∫[a,b] f(x,t) dx
def F(t, method, N=None):
    if method == 'gauss3':
        return gauss_integration(t, 3)
    elif method == 'gauss4':
        return gauss_integration(t, 4)
    elif method in ['rect', 'trap', 'simp']:
        return doubling_method(t, method)
    else:
        raise ValueError("Unknown method")


# Квадратурные формулы Гаусса
def gauss_integration(t, n):
    # Узлы и веса для n=3 и n=4 на интервале [-1, 1]
    if n == 3:
        nodes = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
        weights = [5 / 9, 8 / 9, 5 / 9]
    elif n == 4:
        nodes = [-0.861136, -0.339981, 0.339981, 0.861136]
        weights = [0.347855, 0.652145, 0.652145, 0.347855]

    # Преобразование к интервалу [a, b]
    transformed_nodes = [(b - a) / 2 * xi + (a + b) / 2 for xi in nodes]
    transformed_weights = [(b - a) / 2 * wi for wi in weights]

    # Вычисление интеграла
    integral = 0.0
    for xi, wi in zip(transformed_nodes, transformed_weights):
        integral += wi * f(xi, t)

    return integral


# Метод удвоения числа шагов
def doubling_method(t, method):
    N = 2 if method != 'simp' else 4  # Начальное число шагов (для Симпсона должно быть четным)
    prev_integral = compute_integral(t, N, method)

    while True:
        N *= 2
        current_integral = compute_integral(t, N, method)

        if abs(current_integral - prev_integral) < epsilon:
            break

        prev_integral = current_integral

    return current_integral, N


# Вычисление интеграла по конкретному методу
def compute_integral(t, N, method):
    h = (b - a) / N
    integral = 0.0

    if method == 'rect':  # Метод прямоугольников
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

    return integral


# Вычисление значений F(t_j) в точках t_j = c + j*tau, tau = (d-c)/m
tau = (d - c) / m
results = []

for j in range(m + 1):
    tj = c + j * tau

    # Метод удвоения шагов (используем метод Симпсона как стандартный)
    F_simp, N_simp = doubling_method(tj, 'simp')

    # Квадратуры Гаусса
    F_gauss3 = gauss_integration(tj, 3)
    F_gauss4 = gauss_integration(tj, 4)

    results.append((tj, F_simp, N_simp, F_gauss3, F_gauss4))

# Вывод результатов в виде таблицы
print("Результаты вычисления интеграла F(t) = ∫[a,b] f(x,t) dx")
print(f"Параметры: a={a}, b={b}, c={c}, d={d}, m={m}, ε={epsilon}")
print()
print("{:>5s} {:>10s} {:>10s} {:>16s} {:>12s}".format(
    "t_j", "F(t_j) Симпсон", "N", "F(t_j) Гаусс-3", "F(t_j) Гаусс-4"))
print("-" * 60)

for tj, F_simp, N, F_g3, F_g4 in results:
    print("{:8.3f} {:12.6f} {:8d} {:12.6f} {:12.6f}".format(
        tj, F_simp, N, F_g3, F_g4))
