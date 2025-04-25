import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Подключение 3D-поддержки для графиков

def solve_poisson():
    # ─── Параметры сетки ──────────────────────────────────────────────────────
    nx, ny = 80, 40        # Количество узлов по осям X и Y
    max_iter = 10000       # Максимальное число итераций метода
    tol = 1e-6             # Порог сходимости по максимуму изменения

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

    # Предварительный коэффициент для схемы Гаусса-Зейделя
    factor = 1 / (2/hx2 + 2/hy2)

    # ─── Итерационный процесс (метод Гаусса-Зейделя) ───────────────────────────
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
            print(f'Решение сошлось за {iteration} итераций')
            break

    # ─── Визуализация результатов ─────────────────────────────────────────────

    fig = plt.figure(figsize=(15, 6))

    # 3D-график поверхности решения
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')
    ax1.set_title('Численное решение уравнения Пуассона')

    # Контурный график (плоское представление)
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, u, levels=20, cmap='viridis')
    plt.colorbar(contour)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Контурный график решения')
 

    plt.tight_layout()
    plt.show()

    # ─── Оценка ошибки (по сравнению с предполагаемым аналитическим решением) ─
    



    return X, Y, u


def is_inside(x, y):
    """Определяет, принадлежит ли точка (x, y) расчетной области
    Форма области: прямоугольник с уступом снизу слева (x ≤ 1 и y < 0.25 исключаются)
    """
    if x <= 1 and y < 0.25:
        return False
    return True

def solve_poisson_cg(nx=80, ny=40, tol=1e-6):
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
            print(f'Решение сошлось за {iterations} итераций')
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        

    # Преобразование решения обратно в двумерную форму
    u_sol = vec_to_grid(x)

    # --- ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---

    fig = plt.figure(figsize=(15, 6))

    # Поверхность
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u_sol, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')
    ax1.set_title('Метод сопряжённых градиентов')

    # Контурная карта
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, u_sol, levels=20, cmap='viridis')
    plt.colorbar(contour)
    ax2.set_title('Контурное представление решения')

    plt.tight_layout()
    plt.show()

    return X, Y, u_sol

# ─── Запуск основной функции ─────────────────────────────────────────────────
X, Y, u = solve_poisson()
# Запуск основного расчета
X, Y, u = solve_poisson_cg()
from tabulate import tabulate

def compare_solutions_on_common_nodes(u_fine, u_coarse, nx_fine, nx_coarse):
    """Сравнение решений на общих узлах двух разных сеток"""
    step = nx_fine // nx_coarse
    max_diff = 0.0
    for j in range(nx_coarse):
        for i in range(nx_coarse):
            diff = abs(u_fine[j*step, i*step] - u_coarse[j, i])
            max_diff = max(max_diff, diff)
    return max_diff

def estimate_order(diff1, diff2):
    """Оценка порядка сходимости"""
    if diff1 == 0 or diff2 == 0:
        return float('nan')
    return np.log(diff2 / diff1) / np.log(0.5)

def print_solution_table(u, title="Решение"):
    """Печать решения в виде таблицы"""
    print(f"\n{title} (таблица значений):")
    ny, nx = u.shape
    table = [["y\\x"] + [f"{i}" for i in range(nx)]]
    for j in range(ny):
        row = [f"{j}"] + [f"{u[j, i]:.2e}" for i in range(nx)]
        table.append(row)
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

def print_comparison_report(u_gs, u_cg, iterations_gs, iterations_cg, diff1, diff2):
    """Вывод статистики сравнения"""
    order = estimate_order(diff1, diff2)
    
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МЕТОДОВ ЧИСЛЕННОГО РЕШЕНИЯ УРАВНЕНИЯ ПУАССОНА")
    print("="*70)

    table1 = [
        ["Метод", "Число итераций", "Макс. разность с другим методом"],
        ["Гаусс-Зейдель", iterations_gs, f"{np.max(np.abs(u_gs - u_cg)):.3e}"],
        ["Сопр. градиенты", iterations_cg, "0.000e+00 (эталон)"]
    ]
    print(tabulate(table1, headers="firstrow", tablefmt="fancy_grid"))

    print("\nСРАВНЕНИЕ МЕЖДУ СЕТКАМИ НА ОБЩИХ УЗЛАХ:")
    table2 = [
        ["Сравнение", "Макс. отклонение"],
        ["nx=80 vs nx=40", f"{diff1:.5e}"],
        ["nx=160 vs nx=80", f"{diff2:.5e}"],
        ["Оценка порядка сходимости", f"{order:.2f}"]
    ]
    print(tabulate(table2, headers="firstrow", tablefmt="fancy_grid"))

# Допустим, ты решал с nx=40, 80, 160
nx_values = [40, 80, 160]
u_solutions = []
iters_gs = []
iters_cg = []

# Подсчет решений и сохранение
for nx in nx_values:
    ny = nx // 2  # Сохраняем пропорции 2:1
    print(f"\n>>> СЕТКА {nx}x{ny}")
    _, _, u_gs = solve_poisson()
    _, _, u_cg = solve_poisson_cg(nx, ny)

    u_solutions.append(u_gs)  # Можно заменить на u_cg если хочешь сравнивать cg
    # Здесь нужно также собрать итерации (если вернутые функции вернут это)
    iters_gs.append(0)  # <-- временно, если не возвращаешь из solve_poisson()
    iters_cg.append(0)

# Сравнение на общих узлах
diff1 = compare_solutions_on_common_nodes(u_solutions[1], u_solutions[0], nx_values[1], nx_values[0])
diff2 = compare_solutions_on_common_nodes(u_solutions[2], u_solutions[1], nx_values[2], nx_values[1])

# Вывод
print_comparison_report(u_solutions[1], u_solutions[1], iters_gs[1], iters_cg[1], diff1, diff2)
