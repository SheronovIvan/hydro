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

def solve_single(t, grid, a, h, tau):
    prev = grid[t - 1]
    curr = grid[t]
    hs = 1 / h**2
    ti = 1 / tau
    ah = a / 2
    mu1 = curr[0]
    mu2 = 0
    kappa1 = 0
    kappa2 = 0
    ai = [kappa1]
    bi = [mu1]
    C = -(ti + a * hs)
    for x in range(1, len(curr) - 1):
        A = -0.25 / h * prev[x - 1] - ah * hs
        B = 0.25 / h * prev[x + 1] - ah * hs
        phi = prev[x] * ti + ah * hs * (prev[x + 1] - 2 * prev[x] + prev[x - 1])
        alpha = B / (C - A * ai[x - 1])
        beta = (-phi + A * bi[x - 1]) / (C - A * ai[x - 1])
        ai.append(alpha)
        bi.append(beta)
    curr[-1] = (kappa2 * bi[-1] + mu2) / (1 - kappa2 * ai[-1])
    for x in range(len(curr) - 2, -1, -1):
        curr[x] = ai[x] * curr[x + 1] + bi[x]

def solve(grid, a, h, tau):
    for t in range(1, len(grid)):
        solve_single(t, grid, a, h, tau)

# Параметры
a = 0.05  # Коэффициент диссипации
length = 3  # область по x
time_slice = 5  # Время
n = 30  # Узлов по x (3 / 0.1)
m = 40  # Узлов по t (2 / 0.05)

if __name__ == '__main__':
    grid, hx, tau = create_grid(length, time_slice, n, m, u0_simple, mu)
    solve(grid, a, hx, tau)

    # Визуализация
    fig, ax = plt.subplots()
    x = np.linspace(0, length, n)
    ax.set_xlim(0, 3)
    ax.set_ylim(-1.1, 1.1)
    ax.grid()

    def update(frame):
        ax.clear()
        ax.plot(x, grid[frame], 'b-', linewidth=2)
        ax.set_title(f"Время t = {frame * tau:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")

    ani = animation.FuncAnimation(fig, update, frames=m, interval=50)
    plt.show()