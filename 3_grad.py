import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_inside(x, y):
    """Определяет, принадлежит ли точка (x, y) расчетной области
    Форма области: прямоугольник с уступом снизу слева (x ≤ 1 и y < 0.25 исключаются)
    """
    if x <= 1 and y < 0.25:
        return False
    return True

def solve_poisson_cg(nx=50, ny=25, tol=1e-6):
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

    for it in range(len(b)):
        Ap = laplace_operator(p)
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        iter = it

        # Критерий сходимости по норме невязки
        if np.sqrt(rsnew) < tol * np.linalg.norm(b):
            print(f'Решение сошлось за {it} итераций')
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew
    print(f'Не достигнута точность за {iter} итераций (остаток: {np.sqrt(rsnew):.2e})')
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

    # --- ОЦЕНКА ТОЧНОСТИ ---

    u_analytical = X**2 + Y**2              # Точное решение
    error = np.abs(u_sol - u_analytical)    # Абсолютная ошибка
    print(f'Максимальная ошибка: {np.max(error):.2e}')

    return X, Y, u_sol

# Запуск основного расчета
X, Y, u = solve_poisson_cg()
