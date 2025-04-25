import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tabulate import tabulate

# Параметры области
a, b = 0.0, 1.0  # x ∈ [a, b]
c, d = 0.0, 1.0  # y ∈ [c, d]

def source_function(x, y, A, b_val):
    """Функция источника f(x,y) = A*exp(b*r^2)"""
    r_sq = (x - (a+b)/2)**2 + (y - (c+d)/2)**2
    return A * np.exp(b_val * r_sq)

def boundary_conditions(x, y):
    """Граничные условия"""
    if x == a:
        return 0.0  # u(a,y) = 0 (левая граница)
    elif y == c:
        return 0.0  # u(x,c) = 0 (нижняя граница)
    elif y == d:
        return 0.0  # u(x,d) = 0 (верхняя граница)
    return None

def seidel_solver(nx, ny, A, b_val, max_iter=10000, tol=1e-6):
    """Метод Зейделя с граничными условиями"""
    hx = (b - a) / nx
    hy = (d - c) / ny
    
    # Создание сетки
    x = np.linspace(a, b, nx+1)
    y = np.linspace(c, d, ny+1)
    X, Y = np.meshgrid(x, y)
    F = source_function(X, Y, A, b_val)
    acc = 0
    # Инициализация решения с граничными условиями
    u = np.zeros((ny+1, nx+1))
    for i in range(nx+1):
        for j in range(ny+1):
            bc = boundary_conditions(x[i], y[j])
            if bc is not None:
                u[j, i] = bc
    
    # Итерационный процесс
    for iteration in range(1, max_iter+1):
        max_diff = 0.0
        
        # Обход внутренних точек
        for j in range(1, ny):
            for i in range(1, nx):
                # Особый случай для правой границы (условие Неймана du/dx=0)
                if i == nx-1:
                    new_val = (2*u[j, i-1] + u[j-1, i] + u[j+1, i] + hx**2 * F[j, i]) / 4
                else:
                    new_val = (u[j, i-1] + u[j, i+1] + u[j-1, i] + u[j+1, i] + hx**2 * F[j, i]) / 4
                
                diff = abs(new_val - u[j, i])
                if diff > max_diff:
                    max_diff = diff
                u[j, i] = new_val
        
        # Проверка сходимости
        if max_diff < tol:
            acc = max_diff
            break
    
    return X, Y, u, iteration, acc

def conjugate_gradient_solver(nx, ny, A, b_val, max_iter=10000, tol=1e-6):
    """Метод Сопряжённых Градиентов с граничными условиями"""
    hx = (b - a) / nx
    hy = (d - c) / ny
    
    # Создание сетки
    x = np.linspace(a, b, nx+1)
    y = np.linspace(c, d, ny+1)
    X, Y = np.meshgrid(x, y)
    F = source_function(X, Y, A, b_val)
    acc = 0
    # Инициализация решения с граничными условиями
    u = np.zeros((ny+1, nx+1))
    for i in range(nx+1):
        for j in range(ny+1):
            bc = boundary_conditions(x[i], y[j])
            if bc is not None:
                u[j, i] = bc
    
    # Векторизация внутренних точек (исключаем границы)
    size = (ny-1)*(nx-1)
    u_inner = u[1:-1, 1:-1].flatten()
    
    # Правая часть уравнения (только внутренние точки)
    B = -F[1:-1, 1:-1].flatten()
    
    # Оператор матрицы Лапласа
    def laplace_operator(v):
        v_mat = v.reshape((ny-1, nx-1))
        lap = np.zeros_like(v_mat)
        
        # Внутренние точки (не касаются границ)
        for j in range(1, ny-2):
            for i in range(1, nx-2):
                lap[j, i] = (v_mat[j, i-1] + v_mat[j, i+1] + 
                            v_mat[j-1, i] + v_mat[j+1, i] - 4*v_mat[j, i]) / hx**2
        
        # Граничные условия внутри области
        # Левая граница (i=0)
        for j in range(ny-1):
            # Используем граничное значение u[j+1, 0] вместо v_mat[j, -1]
            lap[j, 0] = (u[j+1, 0] + v_mat[j, 1] + 
                        (v_mat[j-1, 0] if j > 0 else u[j, 0]) + 
                        (v_mat[j+1, 0] if j < ny-2 else u[j+2, 0]) - 
                        4*v_mat[j, 0]) / hx**2
        
        # Правая граница (i=nx-2) - условие Неймана
        for j in range(ny-1):
            # Односторонняя разность для условия Неймана
            lap[j, -1] = (2*v_mat[j, -2] - v_mat[j, -1] + 
                         (v_mat[j-1, -1] if j > 0 else u[j, -1]) + 
                         (v_mat[j+1, -1] if j < ny-2 else u[j+2, -1]) - 
                         4*v_mat[j, -1]) / hx**2
        
        # Верхняя и нижняя границы
        for i in range(nx-1):
            # Нижняя граница (j=0)
            lap[0, i] = ((v_mat[0, i-1] if i > 0 else u[1, i]) + 
                        (v_mat[0, i+1] if i < nx-2 else u[1, i+2]) + 
                        u[0, i+1] + v_mat[1, i] - 4*v_mat[0, i]) / hx**2
            
            # Верхняя граница (j=ny-2)
            lap[-1, i] = ((v_mat[-1, i-1] if i > 0 else u[-2, i]) + 
                         (v_mat[-1, i+1] if i < nx-2 else u[-2, i+2]) + 
                         v_mat[-2, i] + u[-1, i+1] - 4*v_mat[-1, i]) / hx**2
        
        return lap.flatten()
    
    # Реализация метода сопряжённых градиентов
    r = B - laplace_operator(u_inner)
    p = r.copy()
    rsold = np.dot(r, r)
    
    for iteration in range(1, max_iter+1):
        Ap = laplace_operator(p)
        alpha = rsold / np.dot(p, Ap)
        u_inner = u_inner + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        
        if np.sqrt(rsnew) < tol:
            acc = np.sqrt(rsnew)
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    # Восстановление полного решения
    u[1:-1, 1:-1] = u_inner.reshape((ny-1, nx-1))
    
    # Учет условия Неймана на правой границе
    u[:, -1] = u[:, -2]
    
    return X, Y, u, iteration, acc

def plot_solutions(X, Y, u_gs, u_cg, A, b_val, iter_gs, iter_cg):
    """Визуализация решений"""
    fig = plt.figure(figsize=(16, 8))
    
    # График Зейделя
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(X, Y, u_gs, cmap=cm.viridis)
    ax1.set_title(f'Зейдель\nИтераций: {iter_gs}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')
    
    # График Сопряжённых Градиентов
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(X, Y, u_gs, cmap=cm.viridis)
    ax2.set_title(f'Сопр. Градиенты\nИтераций: {iter_cg}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('U')
    
    
    # Контурные графики
    ax4 = fig.add_subplot(223)
    contour = ax4.contourf(X, Y, u_gs, levels=20, cmap=cm.viridis)
    plt.colorbar(contour, ax=ax4)
    ax4.set_title('Зейдель (контур)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    
    ax5 = fig.add_subplot(224)
    contour = ax5.contourf(X, Y, u_gs, levels=20, cmap=cm.viridis)
    plt.colorbar(contour, ax=ax5)
    ax5.set_title('Сопр. Градиенты (контур)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    
    
    plt.suptitle(f'Решение уравнения Пуассона: A={A}, b={b_val}')
    plt.tight_layout()
    plt.show()

# Основные параметры
nx, ny = 50, 50  # Размер сетки
A_values = [10]
b_values = [3]

# Проведение расчетов
for A in A_values:
    for b_val in b_values:
        #print(f"\n=== Решение для A={A}, b={b_val} ===")
        
        # Решение методами
        X, Y, u_gs, iter_gs, acc_s = seidel_solver(nx, ny, A, b_val)
        X, Y, u_cg, iter_cg, acc_c = conjugate_gradient_solver(nx, ny, A, b_val)
        
        # Визуализация
        #plot_solutions(X, Y, u_gs, u_cg, A, b_val, iter_gs, iter_cg)
        
        # Вывод статистики
        #max_diff = np.max(np.abs(u_gs - u_cg))
        #print(f"Метод Зейделя: {iter_gs} итераций")
        #print(f"Метод Сопряжённых Градиентов: {iter_cg} итераций")
        #print(f"Максимальная разность решений: {max_diff:.2e}")

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
    print("\n" + "="*70)
    print("ЗАДАЧА: Решение уравнения Пуассона Δu(x, y) = -f(x, y)")
    print("в прямоугольной области: x ∈ [0, 1], y ∈ [0, 1]")
    print("Краевые условия:")
    print("  u(0, y) = μ₁(y)          (Дирихле по левой границе)")
    print("  u(x, 0) = μ₃(x)          (Дирихле по нижней границе)")
    print("  u(x, 1) = μ₄(x)          (Дирихле по верхней границе)")
    print("  ∂u/∂x (1, y) = 0         (Неймана по правой границе)")
    print("f(x, y) = A * exp(b * ((x - 0.5)^2 + (y - 0.5)^2))")
    print("Методы: Метод Зейделя, Метод сопряжённых градиентов")
    print("="*70 + "\n")

def print_statistics(iter_gs, iter_cg, diff, res_gs, res_cg):
    table = [
        ["Метод", "Итерации", "Точность"],
        ["Зейдель", iter_gs, f"{acc_s:.2e}"],
        ["Сопр. Градиенты", iter_cg, f"{acc_c:.2e}"]
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
            x_gs, y_gs, u_gs, iter_gs, acc_s = seidel_solver(n, n, A, b)
            x_cg, y_cg, u_cg, iter_cg, acc_c = conjugate_gradient_solver(n, n, A, b)
            print(f"Точность Зейделя: {acc_s:.2e}, Точность Сопр. Градиенты: {acc_c:.2e}")
            diff = np.max(np.abs(u_gs - u_cg))
            res_gs = compute_residual(u_gs, A, b, n)
            res_cg = compute_residual(u_cg, A, b, n)
            print_statistics(iter_gs, iter_cg, diff, res_gs, res_cg)
            if n <= 10:
                print_solution_table(u_gs, "Зейдель")
                print_solution_table(u_cg, "Сопр. Градиенты")
            plot_solutions(u_gs, u_gs, A, b)

# Сравнение сходимости между разными n
n1, n2, n3 = 5, 30, 60
A, b = 10, 1

# Считаем решения на разных сетках
_, _, u1, _, _ = seidel_solver(n1, n1, A, b)
_, _, u2, _, _ = seidel_solver(n2, n2, A, b)
_, _, u3, _, _ = seidel_solver(n3, n3, A, b)

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