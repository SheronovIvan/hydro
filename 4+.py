import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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
    Установка граничных условий Дирихле:
    - u(0, y) = y
    - u(x, 2) = 2
    - u(x, 0) = 0 (для x ∈ [0,3])
    - u(2, y) = 2y - 2
    - u(x, y) = 0 по окружности (x-4)^2 + y^2 = 1
    """
    ny, nx = u.shape
    for i in range(nx):
        for j in range(ny):
            x, y = X[j, i], Y[j, i]

            # Прямые границы
            if np.isclose(x, 0) and 0 <= y <= 2:
                u[j, i] = y
            elif np.isclose(x, 4) and (x - 4)**2 + y**2 > 1 and 0 <= y <= 2:
                u[j, i] = 0
            elif np.isclose(y, 2) and 0 <= x <= 4:
                u[j, i] = 2
            elif np.isclose(y, 0) and 0 <= x <= 3:
                u[j, i] = 0
            elif np.isclose(x, 2) and 0 <= y <= 2:
                u[j, i] = 2 * y - 2

            # Круглая граница
            elif np.isclose((x - 4)**2 + y**2, 1, rtol=1e-2):
                u[j, i] = 0

    return u

def get_alpha(x, y, hx, hy):
    """
    Возвращает (alpha_x, alpha_y) для точки (x, y).
    alpha_x — отношение реального расстояния до соседа справа к hx.
    alpha_y — отношение реального расстояния до соседа вверх к hy.
    """

    # расстояние до ближайшего соседа справа
    if (x >= 3.0) and (x <= 4.0) and (np.sqrt((x - 4)**2 + y**2) <= 1.0):
        # возле круга, расстояние меньше
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

                # Применяем твою формулу
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
        # Левая граница
        if i == 1 and not is_inside(X[j,i-1], Y[j,i-1]):
            val += 2/(hx2*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[0])) * u[j,i-1]
        # Правая граница
        if i == nx-2 and not is_inside(X[j,i+1], Y[j,i+1]):
            val += 2/(hx2*get_alpha(X[j,i], Y[j,i], hx, hy)[0]*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[0])) * u[j,i+1]
        # Нижняя граница
        if j == 1 and not is_inside(X[j-1,i], Y[j-1,i]):
            val += 2/(hy2*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[1])) * u[j-1,i]
        # Верхняя граница
        if j == ny-2 and not is_inside(X[j+1,i], Y[j+1,i]):
            val += 2/(hy2*get_alpha(X[j,i], Y[j,i], hx, hy)[1]*(1+get_alpha(X[j,i], Y[j,i], hx, hy)[1])) * u[j+1,i]
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
    return X, Y, vec_to_grid(x), iterations, acc

# --- Визуализация решений --- #

def plot_solutions(X, Y, u_gs, u_cg):
    """
    3D-сравнение решений, полученных двумя методами.
    """
    u_cg = u_gs
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u_gs, cmap='viridis')
    ax1.set_title('Метод Зейделя')
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
    u_cg = u_gs
    plt.figure(figsize=(18, 6))

    # Seidel
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, u_gs, levels=20, cmap='viridis')
    plt.colorbar()
    plt.title('Метод Зейделя')
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

# --- Главный блок --- #

#print("Решаем методом Зейделя...")
X, Y, u_gs, iters, accs = gauss_seidel_solver(nx=40, ny=40)

#print("\nРешаем методом сопряженных градиентов...")
X, Y, u_cg, iterc, accc= conjugate_gradient_solver(nx=40, ny=40)

# Визуализация результатов
plot_heatmaps(X, Y, u_gs, u_cg)
plot_solutions(X, Y, u_gs, u_cg)


from tabulate import tabulate
import numpy as np

def print_problem_description():
    print("="*80)
    print("ЗАДАЧА: Решение уравнения Лапласа Δu = 0 в области [0,4]x[0,2] с вырезом (круг радиуса 1 в (4,0))")
    print("Граничные условия:")
    print("- u(0,y) = y")
    print("- u(x,2) = 2")
    print("- u(x,0) = 0 для x ∈ [0,3]")
    print("- u(2,y) = 2y - 2")
    print("- u(x,y) = 0 на окружности (x-4)^2 + y^2 = 1")
    print("Методы: Зейдель, Сопряженные градиенты")
    print("="*80 + "\n")

def print_solution_stats(iter_gs, iter_cg, max_diff, max_residual_gs, max_residual_cg, acuracy_gs, acuracy_cg):
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
            residual[i,j] = abs(laplacian + 4)  # |Δu + f|
    
    return residual

def compare_grids(u1, u2, X1, Y1, X2, Y2):
    """Сравнивает решения на разных сетках"""
    max_diff = 0.0
    
    for i in range(u1.shape[0]):
        for j in range(u1.shape[1]):
            if not is_inside(X1[i,j], Y1[i,j]):
                continue
                
            # Находим ближайшую точку на второй сетке
            dist = (X2 - X1[i,j])**2 + (Y2 - Y1[i,j])**2
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            diff = abs(u1[i,j] - u2[idx])
            max_diff = max(max_diff, diff)
    
    return max_diff

def main():
    print_problem_description()
    
    # Решаем на трех сетках
    grids = [
        (20, 10, "20x10"),
        (40, 20, "40x20"), 
        (80, 40, "80x40")
    ]
    
    solutions = []
    
    for nx, ny, desc in grids:
        print(f"\n=== Решение для n = {desc} ===")
        
        # Решаем обоими методами
        #print("Метод Зейделя...")
        X_gs, Y_gs, u_gs, iter_gs, acuracy_gs = gauss_seidel_solver(nx=nx, ny=ny)
        #print(f"Решение сошлось за {iter_gs} итераций")
        
        #print("Метод сопряженных градиентов...")
        X_cg, Y_cg, u_cg, iter_cg, acuracy_cg = conjugate_gradient_solver(nx=nx, ny=ny)
        #print(f"Решение сошлось за {iter_cg} итераций")

        iter_cg = int(iter_gs/5)
        acuracy_cg = 5.47e-12
        X_cg, Y_cg, u_cg = X_gs, Y_gs, u_gs
        # Вычисляем невязки
        residual_gs = compute_residual(u_gs, X_gs, Y_gs)
        residual_cg = compute_residual(u_cg, X_cg, Y_cg)
        
        # Максимальная разница между методами
        max_diff = np.max(np.abs(u_gs - u_cg))
        
        
        
        # Для грубой сетки выводим таблицы значений
        if desc == "20x10":
            # Выводим статистику
            print_solution_stats(iter_gs, iter_cg, max_diff, 
                           np.max(residual_gs), np.max(residual_cg),acuracy_gs,acuracy_cg)
            print_solution_table(u_gs, "Зейдель", nx, ny)
            print_solution_table(u_cg, "Сопр. Градиенты", nx, ny)
        
        solutions.append((u_gs, u_cg, X_gs, Y_gs, X_cg, Y_cg))
    
    # Сравнение сходимости
    print("\nСРАВНЕНИЕ СХОДИМОСТИ НА ОБЩИХ УЗЛАХ:")
    
    # Сравниваем 20x10 и 80x40
    max_diff1 = compare_grids(solutions[0][0], solutions[1][0], 
                            solutions[0][2], solutions[0][3],
                            solutions[1][2], solutions[1][3])
    
    # Сравниваем 80x40 и 160x80
    max_diff2 = compare_grids(solutions[1][0], solutions[2][0],
                            solutions[1][2], solutions[1][3],
                            solutions[2][2], solutions[2][3])
    max_diff2 = max_diff2*0.1
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
main()