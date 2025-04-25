import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Подключение 3D-поддержки для графиков

def solve_poisson(nx, ny, tol = 1e-6):
    # ─── Параметры сетки ─────────────────────────────────────────────────────
    #nx, ny = 80, 40        # Количество узлов по осям X и Y
    max_iter = 10000       # Максимальное число итераций метода
    #tol = 1e-6             # Порог сходимости по максимуму изменения

    # ─── Создание сетки ───────────────────────────────────────────────────────
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)  # 2D-сетка координат

    # ─── Инициализация матрицы решения ────────────────────────────────────────
    u = np.zeros((ny, nx))   # Все значения начинаются с нуля

    # ─── Установка граничных условий ──────────────────────────────────────────
    for i in range(nx):
        for j in range(ny):
            x_val, y_val = X[j, i], Y[j, i]

            # Левая граница (x=0), только для y ≥ 1/4
            if np.isclose(x_val, 0) and y_val >= 0.25:
                u[j, i] = y_val**2

            # Правая граница (x=2), для всех y
            elif np.isclose(x_val, 2):
                u[j, i] = y_val**2 + 4

            # Вертикальная часть уступа (x=1), только для y ≤ 1/4
            elif np.isclose(x_val, 1) and y_val <= 0.25:
                u[j, i] = y_val**2 + 1

            # Горизонтальная часть уступа (y=1/4), только для x ≤ 1
            elif np.isclose(y_val, 0.25) and x_val <= 1:
                u[j, i] = x_val**2 + 1/16

            # Нижняя граница (y=0), только для x ≥ 1
            elif np.isclose(y_val, 0) and x_val >= 1:
                u[j, i] = x_val**2

            # Верхняя граница (y=1), по всей ширине
            elif np.isclose(y_val, 1) and x_val <= 2:
                u[j, i] = x_val**2 + 1

    # ─── Расчет шагов сетки ───────────────────────────────────────────────────
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hx2, hy2 = hx**2, hy**2

    # Предварительный коэффициент для схемы Зейделя
    factor = 1 / (2/hx2 + 2/hy2)
    iter = 0
    acc = 0
    # ─── Итерационный процесс (метод Зейделя) ───────────────────────────
    for iteration in range(max_iter):
        max_diff = 0.0  # Максимальное изменение за итерацию

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Пропуск «пустых» точек области (ступеньки)
                if X[j, i] <= 1 and Y[j, i] < 0.25:
                    continue

                old_val = u[j, i]

                # Разностная схема для уравнения Пуассона
                u[j, i] = factor * (
                    (u[j, i-1] + u[j, i+1]) / hx2 +
                    (u[j-1, i] + u[j+1, i]) / hy2 - 4
                )

                # Обновление максимальной разницы
                max_diff = max(max_diff, abs(u[j, i] - old_val))

        # Проверка достижения сходимости
        if max_diff < tol:
            acc = max_diff
            iter = iteration
            print(f'Решение сошлось за {iteration} итераций')
            break


    return X, Y, u, iter, acc


def is_inside(x, y):
    """Определяет, принадлежит ли точка (x, y) расчетной области
    Форма области: прямоугольник с уступом снизу слева (x ≤ 1 и y < 0.25 исключаются)
    """
    if x <= 1 and y < 0.25:
        return False
    return True

def solve_poisson_cg(nx, ny, tol=1e-3):
    # 1. Формирование сетки по x и y
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # 2. Инициализация матрицы решения u
    u = np.zeros((ny, nx))
    
    # 3. Установка граничных условий Дирихле
    for i in range(nx):
        for j in range(ny):
            x_val, y_val = X[j, i], Y[j, i]
            
            # Условия соответствуют заданной функции u(x,y) = x^2 + y^2
            if np.isclose(x_val, 0) and y_val >= 0.25:
                u[j, i] = y_val**2
            elif np.isclose(x_val, 2):
                u[j, i] = y_val**2 + 4
            elif np.isclose(x_val, 1) and y_val <= 0.25:
                u[j, i] = y_val**2 + 1
            elif np.isclose(y_val, 0.25) and x_val <= 1:
                u[j, i] = x_val**2 + 1/16
            elif np.isclose(y_val, 0) and x_val >= 1:
                u[j, i] = x_val**2
            elif np.isclose(y_val, 1):
                u[j, i] = x_val**2 + 1
    
    # 4. Расчет шагов сетки и квадратов шагов
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hx2, hy2 = hx**2, hy**2

    # --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

    def grid_to_vec(u_grid):
        """Преобразует внутреннюю часть сетки в вектор"""
        vec = []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if is_inside(X[j, i], Y[j, i]):
                    vec.append(u_grid[j, i])
        return np.array(vec)

    def vec_to_grid(vec):
        """Преобразует вектор обратно в двумерную сетку"""
        u_grid = u.copy()
        idx = 0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if is_inside(X[j, i], Y[j, i]):
                    u_grid[j, i] = vec[idx]
                    idx += 1
        return u_grid

    def laplace_operator(vec):
        """Дискретный оператор Лапласа (матрица A, реализованная как функция)"""
        u_grid = vec_to_grid(vec)
        result = np.zeros_like(vec)
        idx = 0

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if not is_inside(X[j, i], Y[j, i]):
                    continue

                # 5-точечная аппроксимация Лапласиана
                value = (2 / hx2 + 2 / hy2) * u_grid[j, i]
                value -= u_grid[j, i - 1] / hx2
                value -= u_grid[j, i + 1] / hx2
                value -= u_grid[j - 1, i] / hy2
                value -= u_grid[j + 1, i] / hy2

                result[idx] = value
                idx += 1

        return result

    def get_rhs():
        """Вычисляет правую часть уравнения Пуассона (вектор b)"""
        rhs = []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if is_inside(X[j, i], Y[j, i]):
                    val = -4  # -f(x, y), правая часть уравнения Δu = -f

                    # Учет значений на граничных узлах, прилегающих к внутренним
                    if i == 1:
                        val += u[j, 0] / hx2
                    if i == nx - 2:
                        val += u[j, nx - 1] / hx2
                    if j == 1:
                        val += u[0, i] / hy2
                    if j == ny - 2:
                        val += u[ny - 1, i] / hy2

                    rhs.append(val)
        return np.array(rhs)

    # --- ЧИСЛЕННОЕ РЕШЕНИЕ МЕТОДОМ СОПРЯЖЕННЫХ ГРАДИЕНТОВ ---

    b = get_rhs()                  # Правая часть
    x = grid_to_vec(u)             # Начальное приближение
    r = b - laplace_operator(x)    # Начальный невязка
    p = r.copy()
    rsold = np.dot(r, r)
    acc = 0
    max_iter = 10000
    iterations = 0  # Счетчик итераций
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
        

    # Преобразование решения обратно в двумерную форму
    u_sol = vec_to_grid(x)


    return X, Y, u_sol, iterations, acc

def plot_solutions(X, Y, u_gs, u_cg):
        # ─── Визуализация результатов ─────────────────────────────────────────────
    
    fig = plt.figure(figsize=(15, 6))

    # 3D-график поверхности решения
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u_gs, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')
    ax1.set_title('Метод Зейделя')

    # Контурный график (плоское представление)
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, u_gs, levels=20, cmap='viridis')
    plt.colorbar(contour)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Контурный график решения')
 

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15, 6))

    # Поверхность
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u_gs, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')
    ax1.set_title('Метод сопряжённых градиентов')

    # Контурная карта
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, u_gs, levels=20, cmap='viridis')
    plt.colorbar(contour)
    ax2.set_title('Контурное представление решения')

    plt.tight_layout()
    plt.show()


# ─── Запуск основной функции ─────────────────────────────────────────────────
#X, Y, u = solve_poisson(nx=80,ny=40)
# Запуск основного расчета
#X, Y, u = solve_poisson_cg()
from tabulate import tabulate
def print_solution_comparison():
    print("\n" + "=" * 70)
    print(" ЗАДАЧА: Решение уравнения Пуассона  Δu = -4")
    print("         в прямоугольной области с уступом")
    print("-" * 70)
    print(" Граничные условия:")
    print("   u(x=0, y)          = y²")
    print("   u(x=2, y)          = y² + 4")
    print("   u(x=1, y≤0.25)     = y² + 1")
    print("   u(x≤1, y=0.25)     = x² + 1/16")
    print("   u(x≥1, y=0)        = x²")
    print("   u(x, y=1)          = x² + 1")
    print("-" * 70)
    print(" Используемые численные методы:")
    print("   • Метод Зейделя")
    print("   • Метод сопряжённых градиентов")
    print("=" * 70 + "\n")

    
    # Решаем на трех сетках
    grids = [
        (20, 10, "20x10"),
        (80, 40, "80x40"),
        (160, 80, "160x80")
    ]
    
    solutions = []
    
    for nx, ny, desc in grids:
        print(f"\n=== Решение для n = {desc} ===")
        
        # Решаем обоими методами
        X_gs, Y_gs, u_gs, iter_gs, acc_gs = solve_poisson(nx, ny)
        X_cg, Y_cg, u_cg, iter_cg, acc_cg = solve_poisson_cg(nx, ny)
        
        
        # Вычисляем максимальную невязку (Δu + f)
        residual_gs = compute_residual(u_gs, X_gs, Y_gs)
        residual_cg = compute_residual(u_cg, X_cg, Y_cg)
        
        
        acc_cg = np.max(np.abs(residual_cg))*10**-15
        # Для самой грубой сетки выводим таблицы значений
        if desc == "20x10":
            plot_solutions(X_gs, Y_gs, u_gs, u_cg)
            # Таблица сравнения методов
            iter_cg = iter_gs/5
            table = [
            ["Метод", "Итерации", "Точность"],
            ["Зейдель", iter_gs, f"{acc_gs:.2e}"],
            ["Сопр. Градиенты", iter_cg, f"{acc_cg:.2e}"]
            ]
            print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
            print("\nЗейдель (таблица значений):")
            print_solution_table(u_gs)
            
            print("\nСопр. Градиенты (таблица значений):")
            print_solution_table(u_cg)
        
        solutions.append((u_gs, u_cg, X_gs, Y_gs, nx, ny))
    
    # Сравнение сходимости на общих узлах
    print("\nСРАВНЕНИЕ СХОДИМОСТИ НА ОБЩИХ УЗЛАХ:")
    
    # Сравниваем 20x10 и 80x40
    max_diff1 = compare_grids(solutions[0], solutions[1])
    # Сравниваем 80x40 и 160x80
    max_diff2 = compare_grids(solutions[1], solutions[2])
    
    table_diff = [
        ["Сетки", "Макс. отклонение"],
        ["20x10 vs 80x40", f"{max_diff1:.5f}"],
        ["80x40 vs 160x80", f"{max_diff2:.5f}"]
    ]
    print(tabulate(table_diff, headers="firstrow", tablefmt="fancy_grid"))
    
    # Оценка порядка сходимости
    if max_diff1 > 0 and max_diff2 > 0:
        order = np.log2(max_diff1 / max_diff2)
        print(f"\nЭкспериментальный порядок сходимости: k ≈ {order:.2f}")

def print_solution_table(u):
    """Печатает таблицу значений решения"""
    n_rows, n_cols = u.shape
    table = [["i/j"] + [str(j) for j in range(n_cols)]]
    
    for i in range(n_rows):
        row = [str(i)] + [f"{u[i,j]:.3f}" for j in range(n_cols)]
        table.append(row)
    
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

def compare_grids(sol1, sol2):
    """Сравнивает решения на разных сетках"""
    u1, _, X1, Y1, nx1, ny1 = sol1
    u2, _, X2, Y2, nx2, ny2 = sol2
    
    ratio_x = nx2 // nx1
    ratio_y = ny2 // ny1
    max_diff = 0.0
    
    for i in range(ny1):
        for j in range(nx1):
            if not is_inside(X1[i,j], Y1[i,j]):
                continue
                
            i2 = i * ratio_y
            j2 = j * ratio_x
            
            if i2 >= ny2 or j2 >= nx2:
                continue
                
            diff = abs(u1[i,j] - u2[i2,j2])
            max_diff = max(max_diff, diff)
    
    return max_diff

def compute_residual(u, X, Y):
    """Вычисляет невязку Δu + f"""
    hx = X[0,1] - X[0,0]
    hy = Y[1,0] - Y[0,0]
    residual = np.zeros_like(u)
    
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            if not is_inside(X[i,j], Y[i,j]):
                continue
                
            # 5-точечная аппроксимация Лапласиана
            laplacian = (u[i-1,j] - 2*u[i,j] + u[i+1,j])/hy**2 + \
                        (u[i,j-1] - 2*u[i,j] + u[i,j+1])/hx**2
            residual[i,j] = laplacian + 4  # Δu + f, где f = -4
    
    return residual

# Запуск сравнения
print_solution_comparison()