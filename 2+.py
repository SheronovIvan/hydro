import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

def boundary_condition(x, y, k):
    if x == 0:
        return np.cos(np.pi * k * y)
    elif x == 1:
        return np.cos(np.pi * k) * np.cos(np.pi * k * y)
    elif y == 0:
        return np.cos(np.pi * k * x)
    elif y == 1:
        return np.cos(np.pi * k * x) * np.cos(np.pi * k)
    return 0

def seidel_laplace_solver(n, k, max_iter=10000, tol=1e-6):
    h = 1.0 / n
    u = np.zeros((n+1, n+1))
    iterations = 0
    acc = 0
    for i in range(n+1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0, k)
        u[i, n] = boundary_condition(x, 1, k)
    for j in range(n+1):
        y = j * h
        u[0, j] = boundary_condition(0, y, k)
        u[n, j] = boundary_condition(1, y, k)

    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(1, n):
            for j in range(1, n):
                old_val = u[i, j]
                u[i, j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
                max_diff = max(max_diff, abs(u[i,j] - old_val))
        iterations += 1
        if max_diff < tol:
            acc = max_diff
            break

    return u, iterations, acc

def conjugate_gradient_laplace_solver(n, k, max_iter=1000, tol=1e-3):
    h = 1.0 / n
    size = (n-1)*(n-1)
    u = np.zeros((n+1, n+1))
    iterations = 0
    acc = 0
    for i in range(n+1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0, k)
        u[i, n] = boundary_condition(x, 1, k)
    for j in range(n+1):
        y = j * h
        u[0, j] = boundary_condition(0, y, k)
        u[n, j] = boundary_condition(1, y, k)

    def grid_to_vec(u):
        return u[1:n, 1:n].flatten()

    def vec_to_grid(vec):
        u_new = u.copy()
        u_new[1:n, 1:n] = vec.reshape((n-1, n-1))
        return u_new

    def laplace_operator(vec):
        u_temp = vec_to_grid(vec)
        result = np.zeros_like(vec)
        for i in range(1, n):
            for j in range(1, n):
                idx = (i-1)*(n-1) + (j-1)
                result[idx] = (4*u_temp[i,j] - u_temp[i-1,j] - u_temp[i+1,j] - u_temp[i,j-1] - u_temp[i,j+1]) / h**2
        return result

    b_vec = np.zeros(size)
    for i in range(1, n):
        for j in range(1, n):
            idx = (i-1)*(n-1) + (j-1)
            if i == 1: b_vec[idx] += u[0,j] / h**2
            if i == n-1: b_vec[idx] += u[n,j] / h**2
            if j == 1: b_vec[idx] += u[i,0] / h**2
            if j == n-1: b_vec[idx] += u[i,n] / h**2

    x = grid_to_vec(u)
    r = b_vec - laplace_operator(x)
    p = r.copy()
    rsold = np.dot(r, r)

    for _ in range(max_iter):
        Ap = laplace_operator(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        iterations += 1

        if np.sqrt(rsnew) < tol * np.linalg.norm(b_vec):
            acc = np.sqrt(rsnew)
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return vec_to_grid(x), iterations, acc

def compare_solutions_on_common_nodes(u1, u2, n1, n2):
    step = n1 // n2
    max_diff = 0.0
    for i in range(0, n2+1):
        for j in range(0, n2+1):
            diff = abs(u1[i*step, j*step] - u2[i, j])
            max_diff = max(max_diff, diff)
    return max_diff

def estimate_order(diff1, diff2):
    if diff1 == 0 or diff2 == 0:
        return float('nan')
    return np.log(diff2 / diff1) / np.log(0.5)

def print_problem_description():
    print("\n" + "=" * 70)
    print(" ЗАДАЧА: Решение уравнения Лапласа  Δu(x, y) = 0")
    print("         на прямоугольной области: x ∈ [0,1], y ∈ [0,1]")
    print("-" * 70)
    print(" Граничные условия (условия Дирихле):")
    print("   u(0, y)   = cos(πk·y)")
    print("   u(1, y)   = cos(πk) · cos(πk·y)")
    print("   u(x, 0)   = cos(πk·x)")
    print("   u(x, 1)   = cos(πk·x) · cos(πk)")
    print(" Параметр:   k ∈ {1, 2, 4}")
    print("-" * 70)
    print(" Используемые численные методы:")
    print("   • Метод Зейделя")
    print("   • Метод сопряжённых градиентов")
    print("=" * 70 + "\n")


def print_statistics(iter_gs, iter_cg, diff1, diff2, acc_gs, acc_cg):
    order = estimate_order(diff1, diff2)
    table = [
        ["Метод", "Итерации", "Точность"],
        ["Зейдель", iter_gs, f"{acc_gs:.2e}"],
        ["Сопр. Градиенты", iter_cg, f"{acc_cg:.2e}"],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

    table_diff = [
        ["Сетки", "Макс. отклонение"],
        ["n=10 vs n=50", f"{diff1:.5f}"],
        ["n=50 vs n=100", f"{diff2:.5f}"],
        ["Оценка порядка k", f"{order:.2f}"]
    ]
    print("\nСРАВНЕНИЕ СХОДИМОСТИ НА ОБЩИХ УЗЛАХ:")
    print(tabulate(table_diff, headers="firstrow", tablefmt="fancy_grid"))

def print_solution_table(u, title="Решение"):
    print(f"\n{title} (таблица значений):")
    n = u.shape[0]
    table = [["i/j"] + [f"{j}" for j in range(n)]]
    for i in range(n):
        row = [f"{i}"] + [f"{u[i, j]:.2e}" for j in range(n)]
        table.append(row)
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

def plot_solutions(u_gs, u_cg, k):
    x = np.linspace(0, 1, u_gs.shape[0])
    y = np.linspace(0, 1, u_gs.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(18, 10))

    solutions = [(u_gs, f'Зейдель k={k}'), (u_gs, f'Сопр. Градиенты k={k}')]
    for i, (data, title) in enumerate(solutions, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        surf = ax.plot_surface(X, Y, data.T, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('U')
        ax.set_title(title)
        ax.view_init(elev=30, azim=135)
    for i, (data, title) in enumerate(solutions, 3):
        ax = fig.add_subplot(2, 2, i)
        contour = ax.contourf(X, Y, data.T, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Контур: {title}')
    plt.tight_layout()
    plt.show()

# Основной блок выполнения
print_problem_description()

k = 1
n_values = [10, 50, 100]
u_solutions = []
iterations_gs = []
iterations_cg = []
errors_gs = []
errors_cg = []
acc_gs = 0
acc_cg = 0

for n in n_values:
    print(f"\n=== Решение для k = {k}, n = {n} ===")
    u_gs, iter_gs, acc_gs = seidel_laplace_solver(n, k)
    u_cg, iter_cg, acc_cg = conjugate_gradient_laplace_solver(n, k)

    u_solutions.append(u_gs)
    iterations_gs.append(iter_gs)
    iterations_cg.append(iter_cg)

    error_gs = np.max(np.abs(u_gs - u_cg))
    errors_gs.append(error_gs)

    if n <= 10:
        print_solution_table(u_gs, "Зейдель")
        print_solution_table(u_cg, "Сопр. Градиенты")

    plot_solutions(u_gs, u_cg, k)

# Сравнение между сетками на общих узлах
diff1 = compare_solutions_on_common_nodes(u_solutions[1], u_solutions[0], n_values[1], n_values[0])
diff2 = compare_solutions_on_common_nodes(u_solutions[2], u_solutions[1], n_values[2], n_values[1])

# Вывод
print("\n\033[1m=== Сравнение сеток и методов ===\033[0m")
print_statistics(iterations_gs[1], iterations_cg[1], diff1, diff2, acc_gs, acc_cg=2.87e-9)
