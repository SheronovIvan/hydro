"""
Решение уравнения Пуассона с граничными условиями Дирихле
Сравнение методов Гаусса-Зейделя и Сопряженных Градиентов
Визуализация результатов и анализ погрешностей
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def boundary_condition(x, y):
    """
    Граничные условия Дирихле (в данном случае нулевые)
    Можно изменить на другие при необходимости
    """
    return 0

def source_function(x, y, A, b):
    """
    Функция источника f(x,y) = A*exp(b*r^2)
    где r^2 = (x-0.5)^2 + (y-0.5)^2
    """
    r_squared = (x - 0.5)**2 + (y - 0.5)**2
    return A * np.exp(b * r_squared)

def gauss_seidel_poisson_solver(n, A, b, max_iter=10000, tol=1e-6):
    """
    Решение уравнения Пуассона методом Гаусса-Зейделя
    Возвращает решение и количество итераций
    """
    h = 1.0 / n  # Шаг сетки
    h_sq = h**2   # Квадрат шага сетки
    u = np.zeros((n+1, n+1))  # Инициализация сетки
    iterations = 0  # Счетчик итераций
   
    # Установка граничных условий
    for i in range(n+1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0)  # Нижняя граница
        u[i, n] = boundary_condition(x, 1)  # Верхняя граница
    for j in range(n+1):
        y = j * h
        u[0, j] = boundary_condition(0, y)  # Левая граница
        u[n, j] = boundary_condition(1, y)  # Правая граница

    # Итерационный процесс
    for _ in range(max_iter):
        max_diff = 0.0  # Максимальное изменение на итерации
        for i in range(1, n):
            for j in range(1, n):
                x = i * h
                y = j * h
                f_ij = source_function(x, y, A, b)
                old_val = u[i, j]
                # Обновление значения по формуле Гаусса-Зейделя для уравнения Пуассона
                u[i, j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + h_sq * f_ij)
                max_diff = max(max_diff, abs(u[i,j] - old_val))
        iterations += 1
        # Проверка критерия сходимости
        if max_diff < tol:
            break

    return u, iterations

def conjugate_gradient_poisson_solver(n, A, b, max_iter=1000, tol=1e-6):
    """
    Решение уравнения Пуассона методом Сопряженных Градиентов
    Возвращает решение и количество итераций
    """
    h = 1.0 / n  # Шаг сетки
    h_sq = h**2  # Квадрат шага сетки
    size = (n-1)*(n-1)  # Количество внутренних узлов
    u = np.zeros((n+1, n+1))  # Инициализация сетки
    iterations = 0  # Счетчик итераций

    # Установка граничных условий
    for i in range(n+1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0)  # Нижняя граница
        u[i, n] = boundary_condition(x, 1)  # Верхняя граница
    for j in range(n+1):
        y = j * h
        u[0, j] = boundary_condition(0, y)  # Левая граница
        u[n, j] = boundary_condition(1, y)  # Правая граница

    # Вспомогательные функции для преобразования сетка <-> вектор
    def grid_to_vec(u):
        return u[1:n, 1:n].flatten()

    def vec_to_grid(vec):
        u_new = u.copy()
        u_new[1:n, 1:n] = vec.reshape((n-1, n-1))
        return u_new

    # Оператор Лапласа в матричной форме
    def laplace_operator(vec):
        u_temp = vec_to_grid(vec)
        result = np.zeros_like(vec)
        for i in range(1, n):
            for j in range(1, n):
                idx = (i-1)*(n-1) + (j-1)
                result[idx] = (4*u_temp[i,j] - u_temp[i-1,j] - u_temp[i+1,j] 
                              - u_temp[i,j-1] - u_temp[i,j+1]) / h_sq
        return result

    # Формирование правой части системы уравнений
    b_vec = np.zeros(size)
    for i in range(1, n):
        for j in range(1, n):
            idx = (i-1)*(n-1) + (j-1)
            x = i * h
            y = j * h
            # ИСПРАВЛЕНИЕ: Добавляем минус перед source_function, так как уравнение Δu = -f
            b_vec[idx] = source_function(x, y, A, b)  # Теперь без минуса, так как оператор Лапласа уже с минусом
            # Учет граничных условий в правой части
            if i == 1: b_vec[idx] += u[0,j] / h_sq
            if i == n-1: b_vec[idx] += u[n,j] / h_sq
            if j == 1: b_vec[idx] += u[i,0] / h_sq
            if j == n-1: b_vec[idx] += u[i,n] / h_sq

    # Алгоритм Сопряженных Градиентов
    x = np.zeros(size)  # Начальное приближение - нулевой вектор
    r = b_vec - laplace_operator(x)  # Начальная невязка
    p = r.copy()
    rsold = np.dot(r, r)

    for _ in range(max_iter):
        Ap = laplace_operator(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        iterations += 1

        if np.sqrt(rsnew) < tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return vec_to_grid(x), iterations

def plot_solutions(u_gs, u_cg, A, b):
    """
    Визуализация решений: 3D поверхности и контурные графики
    """
    x = np.linspace(0, 1, u_gs.shape[0])
    y = np.linspace(0, 1, u_gs.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Решение уравнения Пуассона (A={A}, b={b})', fontsize=16)

    # Список решений для визуализации
    solutions = [
        (u_gs, 'Гаусс-Зейдель'),
        (u_cg, 'Сопр. Градиенты')
    ]

    # Построение 3D графиков
    for i, (data, title) in enumerate(solutions, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        surf = ax.plot_surface(X, Y, data.T, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('U')
        ax.set_title(title)
        ax.view_init(elev=30, azim=135)

    # Построение контурных графиков
    for i, (data, title) in enumerate(solutions, 3):
        ax = fig.add_subplot(2, 2, i)
        contour = ax.contourf(X, Y, data.T, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Контур: {title}')

    plt.tight_layout()
    plt.show()

def print_statistics(iter_gs, iter_cg, diff):
    """
    Вывод таблицы с результатами вычислений
    """
    from tabulate import tabulate
    table = [
        ["Метод", "Итерации", "Разница между методами"],
        ["Гаусс-Зейдель", iter_gs, f"{diff:.2e}"],
        ["Сопр. Градиенты", iter_cg, f"{diff:.2e}"]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

# Основные параметры расчета
n = 50  # Количество узлов сетки
A_values = [5, 10, 20]  # Значения параметра A
b_values = [0.3, 1, 3]  # Значения параметра b

# Проведение расчетов для каждой комбинации A и b
for A in A_values:
    for b in b_values:
        print(f"\n\033[1m=== Решение для A = {A}, b = {b} ===\033[0m")
        
        # Численные решения
        u_gs, iter_gs = gauss_seidel_poisson_solver(n, A, b)
        u_cg, iter_cg = conjugate_gradient_poisson_solver(n, A, b)
        
        # Расчет разницы между методами
        diff = np.max(np.abs(u_gs - u_cg))
        
        # Вывод результатов
        print_statistics(iter_gs, iter_cg, diff)
        plot_solutions(u_gs, u_cg, A, b)