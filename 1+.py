import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

def boundary_condition(x, y):
    return 0

def source_function(x, y, A, b):
    r_squared = (x - 0.5)**2 + (y - 0.5)**2
    return A * np.exp(b * r_squared)

def seidel_poisson_solver(n, A, b, max_iter=10000, tol=1e-6):
    h = 1.0 / n
    h_sq = h**2
    u = np.zeros((n+1, n+1))
    iterations = 0
    acc = 0
    for i in range(n+1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0)
        u[i, n] = boundary_condition(x, 1)
    for j in range(n+1):
        y = j * h
        u[0, j] = boundary_condition(0, y)
        u[n, j] = boundary_condition(1, y)

    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(1, n):
            for j in range(1, n):
                x = i * h
                y = j * h
                f_ij = source_function(x, y, A, b)
                old_val = u[i, j]
                u[i, j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + h_sq * f_ij)
                max_diff = max(max_diff, abs(u[i,j] - old_val))
        iterations += 1
        if max_diff < tol:
            acc = max_diff
            break

    return u, iterations, acc

def conjugate_gradient_poisson_solver(n, A, b, max_iter=1000, tol=1e-6):
    h = 1.0 / n
    h_sq = h**2
    size = (n-1)*(n-1)
    u = np.zeros((n+1, n+1))
    iterations = 0

    for i in range(n+1):
        x = i * h
        u[i, 0] = boundary_condition(x, 0)
        u[i, n] = boundary_condition(x, 1)
    for j in range(n+1):
        y = j * h
        u[0, j] = boundary_condition(0, y)
        u[n, j] = boundary_condition(1, y)

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
                result[idx] = (4*u_temp[i,j] - u_temp[i-1,j] - u_temp[i+1,j] - u_temp[i,j-1] - u_temp[i,j+1]) / h_sq
        return result

    b_vec = np.zeros(size)
    for i in range(1, n):
        for j in range(1, n):
            idx = (i-1)*(n-1) + (j-1)
            x = i * h
            y = j * h
            b_vec[idx] = source_function(x, y, A, b)
            if i == 1: b_vec[idx] += u[0,j] / h_sq
            if i == n-1: b_vec[idx] += u[n,j] / h_sq
            if j == 1: b_vec[idx] += u[i,0] / h_sq
            if j == n-1: b_vec[idx] += u[i,n] / h_sq

    x = np.zeros(size)
    r = b_vec - laplace_operator(x)
    p = r.copy()
    rsold = np.dot(r, r)
    acc = 0 
    for _ in range(max_iter):
        Ap = laplace_operator(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        iterations += 1
        if np.sqrt(rsnew) < tol:
            acc = np.sqrt(rsnew)
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return vec_to_grid(x), iterations, acc

def compute_residual(u, A, b, n):
    h = 1.0 / n
    h_sq = h**2
    max_residual = 0.0
    for i in range(1, n):
        for j in range(1, n):
            x = i * h
            y = j * h
            f_val = source_function(x, y, A, b)
            laplace_u = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h_sq
            residual = abs(laplace_u + f_val)
            max_residual = max(max_residual, residual)
    return max_residual

def plot_solutions(u_gs, u_cg, A, b):
    x = np.linspace(0, 1, u_gs.shape[0])
    y = np.linspace(0, 1, u_gs.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Решение уравнения Пуассона (A={A}, b={b})', fontsize=16)

    solutions = [(u_gs, 'Зейдель'), (u_cg, 'Сопр. Градиенты')]
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

def print_problem_description():
    print("\n" + "=" * 70)
    print(" ЗАДАЧА: Решение уравнения Пуассона  Δu(x, y) = -f(x, y)")
    print("         на квадратной области: x ∈ [0,1], y ∈ [0,1]")
    print("-" * 70)
    print(" Граничные условия (условия Дирихле):")
    print("   u(x, y) = 0 на ∂Ω (границе области)")
    print("-" * 70)
    print(" Источник f(x, y):")
    print("   f(x, y) = A · exp(b · r²)")
    print("   где r² = (x - 0.5)² + (y - 0.5)²")
    print(" Параметры:")
    print("   A ∈ {5, 10, 20}")
    print("   b ∈ {0.3, 1, 3}")
    print("-" * 70)
    print(" Используемые численные методы:")
    print("   • Метод Зейделя")
    print("   • Метод сопряжённых градиентов")
    print("=" * 70 + "\n")


def print_statistics(iter_gs, iter_cg, diff, res_gs, res_cg):
    table = [
        ["Метод", "Итерации", "Макс. |GS - CG|", "Макс. невязка Δu+f"],
        ["Зейдель", iter_gs, f"{diff:.2e}", f"{res_gs:.2e}"],
        ["Сопр. Градиенты", iter_cg, f"{diff:.2e}", f"{res_cg:.2e}"]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

def print_solution_table(u, title="Решение"):
    print(f"\n{title} (таблица значений):")
    n = u.shape[0]
    table = [["i/j"] + [f"{j}" for j in range(n)]]
    for i in range(n):
        row = [f"{i}"] + [f"{u[i, j]:.2e}" for j in range(n)]
        table.append(row)
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

def compare_solutions_physical(u_coarse, u_fine, n_coarse, n_fine):
    """
    Сравнивает решения на двух сетках только в совпадающих физических узлах.
    """
    ratio = n_fine // n_coarse
    #assert n_fine % n_coarse == 0, "n_fine должно быть кратно n_coarse"
    max_diff = 0.0
    for i in range(n_coarse + 1):
        for j in range(n_coarse + 1):
            val_coarse = u_coarse[i, j]
            val_fine = u_fine[i * ratio, j * ratio]
            max_diff = max(max_diff, abs(val_coarse - val_fine))
    return max_diff



# Основной расчет
print_problem_description()

n_values = [10]  # Можно заменить на [20, 50, 100]
A_values = [10]
b_values = [1]

for n in n_values:
    for A in A_values:
        for b in b_values:
            print(f"\n=== Решение для A = {A}, b = {b}, n = {n} ===")
            u_gs, iter_gs, acc_s = seidel_poisson_solver(n, A, b)
            u_cg, iter_cg, acc_c = conjugate_gradient_poisson_solver(n, A, b)
            print(f"Точность Зейделя: {acc_s:.2e}, Точность Сопр. Градиенты: {acc_c:.2e}")
            diff = np.max(np.abs(u_gs - u_cg))
            res_gs = compute_residual(u_gs, A, b, n)
            res_cg = compute_residual(u_cg, A, b, n)
            print_statistics(iter_gs, iter_cg, diff, res_gs, res_cg)
            if n <= 10:
                print_solution_table(u_gs, "Зейдель")
                print_solution_table(u_cg, "Сопр. Градиенты")
            plot_solutions(u_gs, u_cg, A, b)

# Сравнение сходимости между разными n
n1, n2, n3 = 5, 30, 60
A, b = 10, 1

# Считаем решения на разных сетках
u1, _, _ = seidel_poisson_solver(n1, A, b)
u2, _, _ = seidel_poisson_solver(n2, A, b)
u3, _, _ = seidel_poisson_solver(n3, A, b)

# Сравнение на общих узлах
diff1 = compare_solutions_physical(u1, u2, n1, n2)
diff2 = compare_solutions_physical(u2, u3, n2, n3)

# Вывод
print("СРАВНЕНИЕ СХОДИМОСТИ НА ОБЩИХ УЗЛАХ:")
print(tabulate([
    ["Сетки", "Макс. отклонение"],
    [f"n={n1} vs n={n2}", f"{diff1:.5f}"],
    [f"n={n2} vs n={n3}", f"{diff2:.5f}"]
], headers="firstrow", tablefmt="fancy_grid"))

# Вычисление порядка сходимости
k = np.log(diff2) / np.log(diff1)
print(f"\nЭкспериментальный порядок сходимости: k ≈ {k:.2f}")


