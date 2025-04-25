"""
Решение уравнения Лапласа с граничными условиями Дирихле
Сравнение методов Гаусса-Зейделя и Сопряженных Градиентов
Визуализация результатов и анализ погрешностей
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def boundary_condition(x, y, k):
    """
    Задание граничных условий Дирихле
    u(x,y)|_Γ = cos(πk x)cos(πk y)
    """
    if x == 0:
        return np.cos(np.pi * k * y)                # Левая граница: x=0
    elif x == 1:
        return np.cos(np.pi * k) * np.cos(np.pi * k * y)  # Правая граница: x=1
    elif y == 0:
        return np.cos(np.pi * k * x)                # Нижняя граница: y=0
    elif y == 1:
        return np.cos(np.pi * k * x) * np.cos(np.pi * k)  # Верхняя граница: y=1
    return 0

def gauss_seidel_laplace_solver(n, k, max_iter=10000, tol=1e-6):
    """
    Решение уравнения Лапласа методом Гаусса-Зейделя
    Возвращает решение и количество итераций
    """
    h = 1.0 / n  # Шаг сетки
    u = np.zeros((n+1, n+1))  # Инициализация сетки
    iterations = 0  # Счетчик итераций

    # Установка граничных условий
    for i in range(n+1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0, k)  # Нижняя граница
        u[i, n] = boundary_condition(x, 1, k)  # Верхняя граница
    for j in range(n+1):
        y = j * h
        u[0, j] = boundary_condition(0, y, k)  # Левая граница
        u[n, j] = boundary_condition(1, y, k)  # Правая граница

    # Итерационный процесс
    for _ in range(max_iter):
        max_diff = 0.0  # Максимальное изменение на итерации
        for i in range(1, n):
            for j in range(1, n):
                old_val = u[i, j]
                # Обновление значения по формуле Гаусса-Зейделя
                u[i, j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
                max_diff = max(max_diff, abs(u[i,j] - old_val))
        iterations += 1
        # Проверка критерия сходимости
        if max_diff < tol:
            break

    return u, iterations

def conjugate_gradient_laplace_solver(n, k, max_iter=10000, tol=1e-1):
    h = 1.0 / n
    u = np.zeros((n + 1, n + 1))

    # Установка граничных условий
    for i in range(n + 1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0, k)
        u[i, n] = boundary_condition(x, 1, k)
    for j in range(n + 1):
        y = j * h
        u[0, j] = boundary_condition(0, y, k)
        u[n, j] = boundary_condition(1, y, k)

    # Вспомогательные функции
    def grid_to_vec(u):
        return u[1:n, 1:n].flatten()

    def vec_to_grid(vec):
        u_temp = u.copy()
        u_temp[1:n, 1:n] = vec.reshape((n - 1, n - 1))
        return u_temp

    def apply_laplace(vec):
        u_grid = vec_to_grid(vec)
        laplace = np.zeros_like(vec)
        for i in range(1, n):
            for j in range(1, n):
                idx = (i - 1) * (n - 1) + (j - 1)
                laplace[idx] = (
                    4 * u_grid[i, j]
                    - u_grid[i - 1, j]
                    - u_grid[i + 1, j]
                    - u_grid[i, j - 1]
                    - u_grid[i, j + 1]
                ) / h**2
        return laplace

    # Формируем правую часть b
    b = np.zeros((n - 1) * (n - 1))
    for i in range(1, n):
        for j in range(1, n):
            idx = (i - 1) * (n - 1) + (j - 1)
            val = 0.0
            if i == 1:
                val += u[0, j] / h**2
            if i == n - 1:
                val += u[n, j] / h**2
            if j == 1:
                val += u[i, 0] / h**2
            if j == n - 1:
                val += u[i, n] / h**2
            b[idx] = val

    # Начальное приближение
    x = np.zeros_like(b)
    r = b - apply_laplace(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    iter = 0
    for iteration in range(max_iter):
        Ap = apply_laplace(p)
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        iter += 1

        if np.sqrt(rs_new) < tol * np.linalg.norm(b):
            print(np.sqrt(rs_new))
            return vec_to_grid(x), iteration + 1

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

  
    print("⚠️ Метод СГ не сошелся за max_iter")
    return vec_to_grid(x), iter




def plot_all_solutions(u_gs, u_cg, u_analytical, k):
    """
    Визуализация решений: 3D поверхности и контурные графики
    """
    x = np.linspace(0, 1, u_gs.shape[0])
    y = np.linspace(0, 1, u_gs.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(18, 10))

    # Список решений для визуализации
    solutions = [
        (u_gs, 'Гаусс-Зейдель'),
        (u_cg, 'Сопр. Градиенты'),
        (u_analytical, 'Аналитическое')
    ]

    # Построение 3D графиков
    for i, (data, title) in enumerate(solutions, 1):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        surf = ax.plot_surface(X, Y, data.T, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('U')
        ax.set_title(title)
        ax.view_init(elev=30, azim=135)

    # Построение контурных графиков
    for i, (data, title) in enumerate(solutions, 4):
        ax = fig.add_subplot(2, 3, i)
        contour = ax.contourf(X, Y, data.T, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Контур: {title}')

    plt.tight_layout()
    plt.show()

def print_statistics(iter_gs, iter_cg, error_gs, error_cg):
    """
    Вывод таблицы с результатами вычислений
    """
    from tabulate import tabulate
    table = [
        ["Метод", "Итерации", "Погрешность"],
        ["Гаусс-Зейдель", iter_gs, f"{error_gs:.2e}"],
        ["Сопр. Градиенты", iter_cg, f"{error_cg:.2e}"]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

# Основные параметры расчета
n = 20  # Количество узлов сетки
k_values = [1, 2, 4]  # Исследуемые волновые числа

# Проведение расчетов для каждого k
for k in k_values:
    print(f"\n\033[1m=== Решение для k = {k} ===\033[0m")
    
    # Численные решения
    u_gs, iter_gs = gauss_seidel_laplace_solver(n, k)
    u_cg, iter_cg = conjugate_gradient_laplace_solver(n, k)
    
    # Аналитическое решение
    X, Y = np.meshgrid(np.linspace(0,1,n+1), np.linspace(0,1,n+1))
    u_analytical = np.cos(np.pi*k*X) * np.cos(np.pi*k*Y)
    
    # Расчет погрешностей
    error_gs = np.max(np.abs(u_gs - u_analytical))
    error_cg = np.max(np.abs(u_cg - u_analytical))
    
    # Вывод результатов
    print_statistics(iter_gs, iter_cg, error_gs, error_cg)
    plot_all_solutions(u_gs, u_cg, u_analytical, k)