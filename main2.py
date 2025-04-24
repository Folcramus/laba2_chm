import numpy as np
import matplotlib.pyplot as plt
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


# Квадратурные формулы Гаусса
def gauss_integration(t, n):
    if n == 3:
        nodes = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
        weights = [5 / 9, 8 / 9, 5 / 9]
    elif n == 4:
        nodes = [-0.861136, -0.339981, 0.339981, 0.861136]
        weights = [0.347855, 0.652145, 0.652145, 0.347855]

    transformed_nodes = [(b - a) / 2 * xi + (a + b) / 2 for xi in nodes]
    transformed_weights = [(b - a) / 2 * wi for wi in weights]

    integral = 0.0
    for xi, wi in zip(transformed_nodes, transformed_weights):
        integral += wi * f(xi, t)
    return integral


# Метод удвоения числа шагов
def doubling_method(t, method):
    N = 2 if method != 'simp' else 4
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
    if method == 'rect':
        for i in range(N):
            xi = a + (i + 0.5) * h
            integral += f(xi, t)
        integral *= h
    elif method == 'trap':
        integral = (f(a, t) + f(b, t)) / 2
        for i in range(1, N):
            xi = a + i * h
            integral += f(xi, t)
        integral *= h
    elif method == 'simp':
        integral = f(a, t) + f(b, t)
        for i in range(1, N):
            xi = a + i * h
            if i % 2 == 1:
                integral += 4 * f(xi, t)
            else:
                integral += 2 * f(xi, t)
        integral *= h / 3
    return integral


# Графическое представление результатов
def visualize_results(t_vals, simp_vals, gauss3_vals, gauss4_vals):
    errors_gauss3 = [abs(s - g3) for s, g3 in zip(simp_vals, gauss3_vals)]
    errors_gauss4 = [abs(s - g4) for s, g4 in zip(simp_vals, gauss4_vals)]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(t_vals, simp_vals, label='Симпсон', marker='o')
    axs[0].plot(t_vals, gauss3_vals, label='Гаусс-3', marker='s')
    axs[0].plot(t_vals, gauss4_vals, label='Гаусс-4', marker='^')
    axs[0].set_title('Значения интеграла F(t) при различных методах')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('F(t)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t_vals, errors_gauss3, label='|Симпсон - Гаусс-3|', marker='s', color='orange')
    axs[1].plot(t_vals, errors_gauss4, label='|Симпсон - Гаусс-4|', marker='^', color='green')
    axs[1].set_title('Абсолютные погрешности относительно метода Симпсона')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Ошибка')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# Основной расчет
tau = (d - c) / m
t_vals = []
simp_vals = []
gauss3_vals = []
gauss4_vals = []

for j in range(m + 1):
    tj = c + j * tau
    F_simp, _ = doubling_method(tj, 'simp')
    F_gauss3 = gauss_integration(tj, 3)
    F_gauss4 = gauss_integration(tj, 4)

    t_vals.append(tj)
    simp_vals.append(F_simp)
    gauss3_vals.append(F_gauss3)
    gauss4_vals.append(F_gauss4)

# Визуализация результатов
visualize_results(t_vals, simp_vals, gauss3_vals, gauss4_vals)

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