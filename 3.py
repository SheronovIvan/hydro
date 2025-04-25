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

        def u_analytical(X, Y):
            u = np.zeros_like(X)

            for i in range(X.shape[1]):
                for j in range(X.shape[0]):
                    x_val, y_val = X[j, i], Y[j, i]

                    # Условие в "допустимой" области — где решение должно быть определено
                    if x_val > 1 or y_val >= 0.25:
                        # Теоретическое решение (просто X^2 + Y^2 как пример)
                        u[j, i] = x_val**2 + y_val**2
                    else:
                        # Пустая часть (ступенька) — остаётся ноль
                        continue

            return u
    # ─── Визуализация результатов ─────────────────────────────────────────────

    fig = plt.figure(figsize=(15, 6))

    # 3D-график поверхности решения
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')
    ax1.set_title('Численное решение уравнения Пуассона')

    ''''
    # Контурный график (плоское представление)
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, u, levels=20, cmap='viridis')
    plt.colorbar(contour)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Контурный график решения')
    '''
    u_real = u_analytical(X,Y)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, u_real, cmap='viridis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Контурный график решения')

    plt.tight_layout()
    plt.show()

    # ─── Оценка ошибки (по сравнению с предполагаемым аналитическим решением) ─
    



    return X, Y, u

# ─── Запуск основной функции ─────────────────────────────────────────────────
X, Y, u = solve_poisson()
def u_analytical(X, Y):
    u = np.zeros_like(X)

    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            x_val, y_val = X[j, i], Y[j, i]

            # Ступенька: пропуск "пустой" области
            if x_val <= 1 and y_val < 0.25:
                continue

            # Левая граница (x=0), только для y ≥ 1/4
            if np.isclose(x_val, 0) and y_val >= 0.25:
                u[j, i] = y_val**2

            # Правая граница (x=2), для всех y
            elif np.isclose(x_val, 2):
                u[j, i] = y_val**2 + 4

            # Вертикальная часть уступа (x=1), только для y ≤ 1/4
            elif np.isclose(x_val, 1) and y_val <= 0.25:
                u[j, i] = y_val**2 + 1

            # Горизонтальная часть уступа (y=0.25), только для x ≤ 1
            elif np.isclose(y_val, 0.25) and x_val <= 1:
                u[j, i] = x_val**2 + 1/16

            # Нижняя граница (y=0), только для x ≥ 1
            elif np.isclose(y_val, 0) and x_val >= 1:
                u[j, i] = x_val**2

            # Верхняя граница (y=1), по всей ширине
            elif np.isclose(y_val, 1) and x_val <= 2:
                u[j, i] = x_val**2 + 1

            # Внутренняя допустимая область
            else:
                u[j, i] = x_val**2 + y_val**2

    return u

error = np.abs(u - u_analytical(X,Y))
print(f'Максимальная ошибка: {np.max(error):.2e}')
# Найдём индексы максимальной ошибки
max_error_index = np.unravel_index(np.argmax(error), error.shape)

# Получим координаты X, Y и значения численного и аналитического решения в этой точке
j_max, i_max = max_error_index
x_max = X[j_max, i_max]
y_max = Y[j_max, i_max]
u_num = u[j_max, i_max]
u_exact = u_analytical(X, Y)[j_max, i_max]
err_val = error[j_max, i_max]

# Вывод информации
print(f"Максимальная ошибка: {err_val:.2e}")
print(f"Координаты точки: x = {x_max:.4f}, y = {y_max:.4f}")
print(f"Численное значение: u = {u_num:.6f}")
print(f"Аналитическое значение: u = {u_exact:.6f}")