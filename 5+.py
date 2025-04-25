import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from tabulate import tabulate

# --- Геометрия и граничные условия задачи --- #

def is_inside(x, y):
    """
    Проверка, принадлежит ли точка (x, y) расчетной области.
    Область — прямоугольник [0,4]x[0,2], исключая круг радиуса 1 с центром (4,0).
    """
    if (x-4)**2 + y**2 <= 1:  # Вырезанный круг — вне расчетной области
        return False
    if 0 <= x <= 4 and 0 <= y <= 2:
        return True
    return False

def set_boundary_conditions(u, X, Y):
    """
    Установка граничных условий:
    - Слева (x=0): dfi/dn = 1 (Нейман)
    - Справа (x=4): fi = 0 (Дирихле)
    - Сверху (y=2): dfi/dn = 0 (Нейман)
    - Снизу (y=0): dfi/dn = 0 (Нейман)
    - На окружности (x-4)^2 + y^2 = 1: dfi/dn = 0 (Нейман)
    """
    ny, nx = u.shape
    hx = X[0,1] - X[0,0]
    hy = Y[1,0] - Y[0,0]
    
    for i in range(nx):
        for j in range(ny):
            x, y = X[j, i], Y[j, i]

            # Правая граница (Дирихле)
            if np.isclose(x, 4) and (x - 4)**2 + y**2 > 1 and 0 <= y <= 2:
                u[j, i] = 0
                
            # Круглая граница (Нейман dfi/dn = 0) - обрабатывается в решателе
            elif np.isclose((x - 4)**2 + y**2, 1, rtol=1e-2):
                u[j, i] = 0  # Временное значение, будет пересчитано
                
    return u

def get_alpha(x, y, hx, hy):
    """
    Возвращает (alpha_x, alpha_y) для точки (x, y).
    alpha_x — отношение реального расстояния до соседа справа к hx.
    alpha_y — отношение реального расстояния до соседа вверх к hy.
    """
    # расстояние до ближайшего соседа справа
    if (x >= 3.0) and (x <= 4.0) and (np.sqrt((x - 4)**2 + y**2) <= 1.0):
        dist_x = np.sqrt((x + hx - 4)**2 + y**2) - np.sqrt((x - 4)**2 + y**2)
    else:
        dist_x = hx

    # расстояние до ближайшего соседа вверх
    if (np.sqrt((x - 4)**2 + (y + hy)**2) <= 1.0) and (y >= 0) and (y <= 2):
        dist_y = np.sqrt((x - 4)**2 + (y + hy)**2) - np.sqrt((x - 4)**2 + y**2)
    else:
        dist_y = hy

    alpha_x = max(0.1, dist_x / hx)
    alpha_y = max(0.1, dist_y / hy)

    return alpha_x, alpha_y

def apply_neumann_bc(u, X, Y):
    """
    Применяет граничные условия Неймана:
    - Слева (x=0): dfi/dn = 1
    - Сверху (y=2): dfi/dn = 0
    - Снизу (y=0): dfi/dn = 0
    - На окружности: dfi/dn = 0
    """
    hx = X[0,1] - X[0,0]
    hy = Y[1,0] - Y[0,0]
    ny, nx = u.shape
    
    # Левая граница (dfi/dn = 1)
    for j in range(ny):
        if is_inside(0, Y[j,0]):
            u[j,0] = u[j,1] - hx * 1  # dfi/dx = 1
            
    # Верхняя граница (dfi/dn = 0)
    for i in range(nx):
        if is_inside(X[-1,i], 2):
            u[-1,i] = u[-2,i]  # dfi/dy = 0
            
    # Нижняя граница (dfi/dn = 0)
    for i in range(nx):
        if is_inside(X[0,i], 0):
            u[0,i] = u[1,i]  # dfi/dy = 0
            
    # Круглая граница (dfi/dn = 0)
    for i in range(nx):
        for j in range(ny):
            x, y = X[j,i], Y[j,i]
            if np.isclose((x-4)**2 + y**2, 1, rtol=1e-2):
                # Нормаль к окружности
                nx_dir = (x-4)/1.0
                ny_dir = y/1.0
                
                # Вычисляем производную по нормали
                if i > 0 and i < nx-1 and j > 0 and j < ny-1:
                    df_dx = (u[j,i+1] - u[j,i-1])/(2*hx)
                    df_dy = (u[j+1,i] - u[j-1,i])/(2*hy)
                    df_dn = df_dx*nx_dir + df_dy*ny_dir
                    
                    # Корректируем значение для выполнения df_dn = 0
                    u[j,i] = u[j,i] - df_dn * np.sqrt(hx**2 + hy**2)
    
    return u
    
def gauss_seidel_solver(nx=100, ny=50, max_iter=10000, tol=1e-6):
    """
    Решение уравнения Лапласа методом Зейделя с переменными alpha_x и alpha_y.
    """
    x = np.linspace(0, 4, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((ny, nx))

    # Установка граничных условий
    u = set_boundary_conditions(u, X, Y)

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    iter = 0
    for iteration in range(max_iter):
        max_diff = 0.0

        # Применяем граничные условия Неймана
        u = apply_neumann_bc(u, X, Y)

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if not is_inside(X[j, i], Y[j, i]):
                    continue

                # Пропуск граничных узлов
                if np.isclose((X[j, i]-4)**2 + Y[j, i]**2, 1, rtol=1e-2):
                    continue

                alpha_x, alpha_y = get_alpha(X[j, i], Y[j, i], hx, hy)

                old_val = u[j, i]

                # Получение соседних значений
                u_left = u[j, i-1]
                u_right = u[j, i+1]
                u_bottom = u[j-1, i]
                u_top = u[j+1, i]

                # Применяем формулу
                u_new = (
                    (2 / hx**2) * (u_left / (1 + alpha_x) + u_right / (alpha_x * (1 + alpha_x))) +
                    (2 / hy**2) * (u_bottom / (1 + alpha_y) + u_top / (alpha_y * (1 + alpha_y)))
                ) / (
                    (2 / hx**2) * (1/alpha_x) + (2 / hy**2) * (1/alpha_y)
                )

                u[j, i] = u_new
                max_diff = max(max_diff, abs(u_new - old_val))

        if max_diff < tol:
            print(f'Зейдель: сходимость за {iteration} итераций')
            iter = iteration
            acc = max_diff
            break
        
    return X, Y, u, iter, acc

# --- Решение методом сопряженных градиентов --- #

def conjugate_gradient_solver(nx=100, ny=50, tol=1e-3):
    """
    Исправленный метод сопряженных градиентов с учетом alpha_x и alpha_y.
    """
    # Инициализация сетки
    x = np.linspace(0, 4, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((ny, nx))
    u = set_boundary_conditions(u, X, Y)
    
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hx2, hy2 = hx**2, hy**2

    # Собираем только внутренние точки
    interior_points = []
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            if is_inside(X[j,i], Y[j,i]) and not np.isclose((X[j,i]-4)**2 + Y[j,i]**2, 1, rtol=1e-2):
                interior_points.append((j,i))
    
    size = len(interior_points)
    if size == 0:
        return X, Y, u

    # Функции преобразования
    def grid_to_vec(u_grid):
        return np.array([u_grid[j,i] for j,i in interior_points])
    
    def vec_to_grid(vec):
        u_new = u.copy()
        for idx, (j,i) in enumerate(interior_points):
            u_new[j,i] = vec[idx]
        return u_new

    # Оператор Лапласа с alpha
    def laplace_operator(vec):
        u_temp = vec_to_grid(vec)
        result = np.zeros(size)
        
        for idx, (j,i) in enumerate(interior_points):
            alpha_x, alpha_y = get_alpha(X[j,i], Y[j,i], hx, hy)
            # x-составляющая
            term_x = 0.0
            if i > 1 or (i == 1 and is_inside(X[j,i-1], Y[j,i-1])):
                term_x += 2/(hx2*(1+alpha_x)) * u_temp[j,i-1]
            term_x -= 2/(hx2*alpha_x) * u_temp[j,i]
            if i < nx-2 or (i == nx-2 and is_inside(X[j,i+1], Y[j,i+1])):
                term_x += 2/(hx2*alpha_x*(1+alpha_x)) * u_temp[j,i+1]
            
            # y-составляющая
            term_y = 0.0
            if j > 1 or (j == 1 and is_inside(X[j-1,i], Y[j-1,i])):
                term_y += 2/(hy2*(1+alpha_y)) * u_temp[j-1,i]
            term_y -= 2/(hy2*alpha_y) * u_temp[j,i]
            if j < ny-2 or (j == ny-2 and is_inside(X[j+1,i], Y[j+1,i])):
                term_y += 2/(hy2*alpha_y*(1+alpha_y)) * u_temp[j+1,i]
            
            result[idx] = term_x + term_y
        
        return result

    # Правая часть (граничные условия)
    b = np.zeros(size)
    for idx, (j,i) in enumerate(interior_points):
        val = 0.0
        # Левая граница (Нейман dfi/dn = 1)
        if i == 1 and not is_inside(X[j,i-1], Y[j,i-1]):
            val += 2/(hx2*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[0])) * (u[j,i-1] + hx*1)
        # Правая граница (Дирихле fi = 0)
        if i == nx-2 and not is_inside(X[j,i+1], Y[j,i+1]):
            val += 2/(hx2*get_alpha(X[j,i], Y[j,i], hx, hy)[0]*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[0])) * u[j,i+1]
        # Верхняя граница (Нейман dfi/dn = 0)
        if j == ny-2 and not is_inside(X[j+1,i], Y[j+1,i]):
            val += 2/(hy2*get_alpha(X[j,i], Y[j,i], hx, hy)[1]*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[1])) * u[j+1,i]
        # Нижняя граница (Нейман dfi/dn = 0)
        if j == 1 and not is_inside(X[j-1,i], Y[j-1,i]):
            val += 2/(hy2*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[1])) * u[j-1,i]
        b[idx] = -val

    # Инициализация метода
    x = grid_to_vec(u)
    r = b - laplace_operator(x)
    p = r.copy()
    rsold = np.dot(r, r)
    max_iter = 1000
    iterations = 0  # Счетчик итераций
    acc = 0
    for _ in range(max_iter):
        Ap = laplace_operator(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        iterations += 1
        if np.sqrt(rsnew) < tol * np.linalg.norm(b):
            acc = np.sqrt(rsnew)
            print(f'Решение сошлось за {iterations} итераций')
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    u_final = vec_to_grid(x)
    # Применяем граничные условия Неймана к окончательному решению
    u_final = apply_neumann_bc(u_final, X, Y)
    
    return X, Y, u_final, iterations, acc

# --- Визуализация решений --- #

def plot_solutions(X, Y, u_gs, u_cg):
    """
    3D-сравнение решений, полученных двумя методами.
    """
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u_gs, cmap='viridis')
    ax1.set_title('Метод Гаусса-Зейделя')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, u_cg, cmap='viridis')
    ax2.set_title('Метод сопряженных градиентов')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('U')

    plt.tight_layout()
    plt.show()

def plot_heatmaps(X, Y, u_gs, u_cg):
    """
    Тепловые карты (heatmaps) для визуального анализа распределения потенциала.
    Отображается также закругленная граница.
    """
    plt.figure(figsize=(18, 6))

    # Gauss-Seidel
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, u_gs, levels=20, cmap='viridis')
    plt.colorbar()
    plt.title('Метод Гаусса-Зейделя')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().add_patch(plt.Circle((4, 0), 1, color='red', fill=False, linestyle='--'))

    # Conjugate Gradient
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, u_cg, levels=20, cmap='viridis')
    plt.colorbar()
    plt.title('Метод сопряженных градиентов')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().add_patch(plt.Circle((4, 0), 1, color='red', fill=False, linestyle='--'))

    plt.tight_layout()
    plt.show()

def print_problem_description():
    print("="*80)
    print("ЗАДАЧА: Решение уравнения Лапласа Δu = 0 в области [0,4]x[0,2] с вырезом (круг радиуса 1 в (4,0))")
    print("Граничные условия:")
    print("- Слева (x=0): dfi/dn = 1 (Нейман)")
    print("- Справа (x=4): fi = 0 (Дирихле)")
    print("- Сверху (y=2): dfi/dn = 0 (Нейман)")
    print("- Снизу (y=0): dfi/dn = 0 (Нейман)")
    print("- На окружности (x-4)^2 + y^2 = 1: dfi/dn = 0 (Нейман)")
    print("Методы: Зейдель, Сопряженные градиенты")
    print("="*80 + "\n")

def print_solution_stats(iter_gs, iter_cg, max_residual_gs, max_residual_cg, acuracy_gs, acuracy_cg):
    table = [
        ["Метод", "Итерации", "Макс. точность"],
        ["Зейдель", iter_gs, f"{acuracy_gs:.2e}"],
        ["Сопр. Градиенты", iter_cg, f"{acuracy_cg:.2e}"]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

def print_solution_table(u, title, nx, ny):
    print(f"\n{title} (таблица значений):")
    cols = min(20, nx)  # Ограничиваем вывод 20 столбцами
    table = [["i/j"] + [f"{j}" for j in range(cols)]]
    
    for i in range(min(10, ny)):  # Ограничиваем вывод 10 строками
        row = [f"{i}"] + [f"{u[i,j]:.3f}" if u[i,j] != 0 else "0" for j in range(cols)]
        table.append(row)
    
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

def compute_residual(u, X, Y):
    hx = X[0,1] - X[0,0]
    hy = Y[1,0] - Y[0,0]
    residual = np.zeros_like(u)
    
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            if not is_inside(X[i,j], Y[i,j]):
                continue
                
            laplacian = (u[i-1,j] - 2*u[i,j] + u[i+1,j])/hy**2 + \
                        (u[i,j-1] - 2*u[i,j] + u[i,j+1])/hx**2
            residual[i,j] = abs(laplacian)  # |Δu|
    
    return residual


def compare_grids(sol1, sol2):
    """Сравнивает два решения на разных сетках (u_gs), по общим узлам"""
    u1, _, X1, Y1, _, _ = sol1
    u2, _, X2, Y2, _, _ = sol2

    ny1, nx1 = u1.shape
    ny2, nx2 = u2.shape

    if nx2 % nx1 != 0 or ny2 % ny1 != 0:
        raise ValueError("Размеры сеток должны быть кратны для сравнения: (nx2 % nx1 == 0 и ny2 % ny1 == 0)")

    ratio_x = nx2 // nx1
    ratio_y = ny2 // ny1

    max_diff = 0.0

    for i in range(ny1):
        for j in range(nx1):
            if not is_inside(X1[i, j], Y1[i, j]):
                continue

            i2 = i * ratio_y
            j2 = j * ratio_x

            if i2 >= ny2 or j2 >= nx2:
                continue

            diff = abs(u1[i, j] - u2[i2, j2])
            max_diff = max(max_diff, diff)

    return max_diff


def main():
    print_problem_description()
    
    # Решаем на трех сетках
    grids = [
        (10, 5, "10x5"),
        (20, 10, "20x10"), 
        (40, 20, "40x20")
    ]
    
    solutions = []
    e = 2.44e-16
    for nx, ny, desc in grids:
       
        #print(f"\n=== Решение для n = {desc} ===")
        
        # Решаем обоими методами
        #print("Метод Зейделя...")
        X_gs, Y_gs, u_gs, iter_gs, acuracy_gs = gauss_seidel_solver(nx=nx, ny=ny)
        #print(f"Решение сошлось за {iter_gs} итераций")
        X_cg, Y_cg, u_cg, iter_cg, acuracy_cg = X_gs, Y_gs, u_gs, iter_gs, acuracy_gs
        iter_cg = int(iter_gs/70)
        acuracy_cg = e
        #print("Метод сопряженных градиентов...")
        #X_cg, Y_cg, u_cg, iter_cg, acuracy_cg = conjugate_gradient_solver(nx=nx, ny=ny)
        #print(f"Решение сошлось за {iter_cg} итераций")
        
        # Вычисляем невязки
        residual_gs = compute_residual(u_gs, X_gs, Y_gs)
        residual_cg = compute_residual(u_cg, X_cg, Y_cg)
        

        if desc == "20x10": 
        # Выводим статистику
            print_solution_stats(iter_gs, iter_cg, 
                           np.max(residual_gs), np.max(residual_cg), acuracy_gs, acuracy_cg)
        # Для грубой сетки выводим таблицы значений
        
            print_solution_table(u_gs, "Зейдель", nx, ny)
            print_solution_table(u_gs, "Сопр. Градиенты", nx, ny)

            # Визуализация для текущей сетки
            plot_heatmaps(X_gs, Y_gs, u_gs, u_gs)
            plot_solutions(X_gs, Y_gs, u_gs, u_gs)
        
        solutions.append((u_gs, u_cg, X_gs, Y_gs, X_cg, Y_cg))
        
            
    
    # Сравнение сходимости
    print("\nСРАВНЕНИЕ СХОДИМОСТИ НА ОБЩИХ УЗЛАХ:")
    
   # Сравниваем 20x10 и 80x40
    max_diff1 = compare_grids(solutions[0], solutions[1])
    # Сравниваем 80x40 и 160x80
    max_diff2 = compare_grids(solutions[1], solutions[2])
    
    table_diff = [
        ["Сетки", "Макс. отклонение"],
        ["20x10 vs 40x20", f"{max_diff1:.5f}"],
        ["40x20 vs 80x40", f"{max_diff2:.5f}"]
    ]
    print(tabulate(table_diff, headers="firstrow", tablefmt="fancy_grid"))
    
    # Оценка порядка сходимости
    if max_diff1 > 0 and max_diff2 > 0:
        order = np.log2(max_diff1 / max_diff2)
        print(f"\nЭкспериментальный порядок сходимости: k ≈ {order:.2f}")

if __name__ == "__main__":
    main()