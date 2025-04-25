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
    Решение уравнения Лапласа методом Гаусса-Зейделя с переменными alpha_x и alpha_y.
    """
    x = np.linspace(0, 4, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((ny, nx))

    # Установка граничных условий
    u = set_boundary_conditions(u, X, Y)

    hx = x[1] - x[0]
    hy = y[1] - y[0]

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
            print(f'Гаусса-Зейдель: сходимость за {iteration} итераций')
            break

    return X, Y, u

# --- Решение методом сопряженных градиентов --- #

def conjugate_gradient_solver(nx=100, ny=50, tol=1e-6):
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

    # Главный цикл
    for it in range(size):
        Ap = laplace_operator(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)

        if np.sqrt(rsnew) < tol:
            print(f'Сопряженные градиенты: сходимость за {it+1} итераций')
            break

        p = r + (rsnew/rsold)*p
        rsold = rsnew

    return X, Y, vec_to_grid(x)

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

# --- Главный блок --- #

print("Решаем методом Гаусса-Зейделя...")
X, Y, u_gs = gauss_seidel_solver(nx=40, ny=40)

print("\nРешаем методом сопряженных градиентов...")
X, Y, u_cg = conjugate_gradient_solver(nx=40, ny=40)

# Визуализация результатов
plot_heatmaps(X, Y, u_gs, u_cg)
plot_solutions(X, Y, u_gs, u_cg)

# Сравнение точности
max_diff = np.max(np.abs(u_gs - u_cg))
print(f"\nМаксимальная разница между методами: {max_diff:.2e}")