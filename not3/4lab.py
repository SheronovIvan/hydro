from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def create_grid(length, time_slice, n, m, u0, mu):
    grid = []
    h = length / n
    t = time_slice / m
    for j in range(m):
        grid.append([0] * n)
    for i in range(n):
        grid[0][i] = u0(i * h)
    for j in range(m):
        grid[j][0] = mu(j * t)
    return grid, h, t

def u0_simple(x):
    return sin(pi * (x - 1)) if 1 <= x <= 2 else 0  # Гладкий импульс на [1, 2]

def mu(t):
    return 0  # Граничное условие u(0, t) = 0

# Явная схема 1-го порядка (backward difference)
def solve1(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t])):
            grid[t][x] = grid[t-1][x] - a * tau * (grid[t-1][x] - grid[t-1][x-1]) / h
    return grid

# Явная схема 2-го порядка (central difference)
def solve2(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t]) - 1):
            grid[t][x] = grid[t-1][x] - a * tau * (grid[t-1][x+1] - grid[t-1][x-1]) / (2 * h)
    return grid

# Неявная схема через прогонку
def solve_single(t, grid, a, h, tau):
    prev = grid[t - 1]
    curr = grid[t]
    ti = 1 / tau
    mu1 = curr[0]
    mu2 = 0
    kappa1 = 0
    kappa2 = 0
    ai = [kappa1]
    bi = [mu1]
    C = -ti
    A = -a / (4 * h)
    B = a / (4 * h)
    for x in range(1, len(curr) - 1):
        phi = prev[x] * ti - a / (4 * h) * (prev[x + 1] - prev[x - 1])
        alpha = B / (C - A * ai[x - 1])
        beta = (-phi + A * bi[x - 1]) / (C - A * ai[x - 1])
        ai.append(alpha)
        bi.append(beta)
    curr[-1] = (kappa2 * bi[-1] + mu2) / (1 - kappa2 * ai[-1])
    for x in range(len(curr) - 2, -1, -1):
        curr[x] = ai[x] * curr[x + 1] + bi[x]

def solve3(grid, a, h, tau):
    for t in range(1, len(grid)):
        solve_single(t, grid, a, h, tau)
    return grid

# Точное решение сдвига импульса
def generate_precise(u0, a, i, h, n, tau):
    return [u0(x * h - i * tau * a) for x in range(n)]

# Параметры
a = 0.5   # Скорость переноса
length = 3
time_slice = 5
n = 30
m = 100

# CFL условие
if __name__ == '__main__':
    target = u0_simple
    grid1, hx, tau = create_grid(length, time_slice, n, m, target, mu)
    grid2, hx, tau = create_grid(length, time_slice, n, m, target, mu)
    grid3, hx, tau = create_grid(length, time_slice, n, m, target, mu)

    c = abs(a) * tau / hx
    if c > 1:
        print('Не выполняется условие устойчивости: c =', c)
        exit(1)

    grid1 = solve1(grid1, a, hx, tau)
    grid2 = solve2(grid2, a, hx, tau)
    grid3 = solve3(grid3, a, hx, tau)

    # Визуализация
    fig, ax = plt.subplots()
    x = np.linspace(0, length, n)
    ax.set_xlim(0, length)
    ax.set_ylim(-1.1, 1.1)
    ax.grid()

    def update(frame):
        ax.clear()
        ax.plot(x, grid1[frame], 'r-', label='Явная 1-го порядка', linewidth=1.5)
        ax.plot(x, grid2[frame], 'g-', label='Явная 2-го порядка', linewidth=1.5)
        ax.plot(x, grid3[frame], 'b-', label='Неявная', linewidth=1.5)
        ax.plot(x, generate_precise(target, a, frame, hx, n, tau), 'k--', label='Точное', linewidth=2)
        ax.set_title(f"Время t = {frame * tau:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_xlim(0, length)
        ax.set_ylim(-1.1, 1.1)
        ax.grid()
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=m, interval=50)
    plt.show()
