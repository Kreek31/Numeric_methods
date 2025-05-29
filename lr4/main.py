import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import math

def eiler(step: float):
    # y''=(y-(x+1)*y')/x^2
    # x лежит в [1; 2]
    x, y, z = sp.symbols('x y z', real=True)
    double_step = 2 * step
    p = 1
    x_start = 1
    x_end = 2
    y_start = 2 + sp.exp(1)
    z_start = 1
    y_accurate = x + 1 + x*sp.exp(1/x)
    y_accurate_lamb = sp.lambdify(x, y_accurate)

    x__ = []
    y__ = []
    y_2h = []
    x_accurate_vals = np.linspace(x_start, x_end, 400)
    y_accurate_vals = y_accurate_lamb(x_accurate_vals)
    for i in np.linspace(x_start, x_end, int((x_end - x_start)/step)+1):
        x__.append(i)

    y_diff_func = z
    z_diff_func = (y-(x+1)*z)/(x**2)
    y_diff_lamb = sp.lambdify([x, y, z], y_diff_func)
    z_diff_lamb = sp.lambdify([x, y, z], z_diff_func)

    x_prev = x_start
    y_prev = y_start
    z_prev = z_start
    y__.append(y_start)
    for i in range(int((x_end - x_start) / step)):
        y_next = y_prev + step * y_diff_lamb(x_prev, y_prev, z_prev)
        z_next = z_prev + step * z_diff_lamb(x_prev, y_prev, z_prev)
        x_next = x_prev + step

        x_prev = x_next
        y_prev = y_next
        z_prev = z_next
        y__.append(y_next)
        print("y:", y_next, "\tz:", z_next)

    x_prev = x_start
    y_prev = y_start
    z_prev = z_start
    y_2h.append(y_start)
    for i in range(int((x_end - x_start) / double_step)):
        y_next = y_prev + double_step * y_diff_lamb(x_prev, y_prev, z_prev)
        z_next = z_prev + double_step * z_diff_lamb(x_prev, y_prev, z_prev)
        x_next = x_prev + double_step

        x_prev = x_next
        y_prev = y_next
        z_prev = z_next
        y_2h.append(y_next)

    for i in range(len(y_2h)):
        R = (y__[2*i] - y_2h[i]) / (2**p - 1)
        print("Член погрешности на шаге", 2*i, "равен:", R.evalf(), "\tразность с точным решением:", (y_accurate_lamb(x__[2*i]) - y__[2*i]).evalf())
    
    plt.figure()
    plt.plot(x_accurate_vals, y_accurate_vals, c='green') # точные значения
    plt.scatter(x__, y__, c='red') # приближения
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def runge_kutt(step: float):
    # y''=(y-(x+1)*y')/x^2
    # x лежит в [1; 2]
    x, y, z = sp.symbols('x y z', real=True)
    double_step = step*2
    p = 4
    x_start = 1
    x_end = 2
    y_start = 2 + sp.exp(1)
    z_start = 1
    y_accurate = x + 1 + x*sp.exp(1/x)
    y_accurate_lamb = sp.lambdify(x, y_accurate)

    x__ = []
    y__ = []
    y_2h = []
    x_accurate_vals = np.linspace(x_start, x_end, 400)
    y_accurate_vals = y_accurate_lamb(x_accurate_vals)
    for i in np.linspace(x_start, x_end, int((x_end - x_start)/step)+1):
        x__.append(i)

    y_diff_func = z
    z_diff_func = (y-(x+1)*z)/(x**2)
    y_diff_lamb = sp.lambdify([x, y, z], y_diff_func)
    z_diff_lamb = sp.lambdify([x, y, z], z_diff_func)

    x_prev = x_start
    y_prev = y_start
    z_prev = z_start
    y__.append(y_start)
    for i in range(int((x_end - x_start) / step)):
        k1_y = step * y_diff_lamb(x_prev, y_prev, z_prev)
        k2_y = step * y_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k1_y, z_prev + (1/2)*k1_y)
        k3_y = step * y_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k2_y, z_prev + (1/2)*k2_y)
        k4_y = step * y_diff_lamb(x_prev + step, y_prev + k3_y, z_prev + k3_y)

        k1_z = step * z_diff_lamb(x_prev, y_prev, z_prev)
        k2_z = step * z_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k1_z, z_prev + (1/2)*k1_z)
        k3_z = step * z_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k2_z, z_prev + (1/2)*k2_z)
        k4_z = step * z_diff_lamb(x_prev + step, y_prev + k3_z, z_prev + k3_z)


        y_next = y_prev + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
        z_next = z_prev + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z)
        x_next = x_prev + step

        x_prev = x_next
        y_prev = y_next
        z_prev = z_next
        y__.append(y_next)
        print("y:", y_next, "\tz:", z_next)

    x_prev = x_start
    y_prev = y_start
    z_prev = z_start
    y_2h.append(y_start)
    for i in range(int((x_end - x_start) / double_step)):
        k1_y = double_step * y_diff_lamb(x_prev, y_prev, z_prev)
        k2_y = double_step * y_diff_lamb(x_prev + (1/2)*double_step, y_prev + (1/2)*k1_y, z_prev + (1/2)*k1_y)
        k3_y = double_step * y_diff_lamb(x_prev + (1/2)*double_step, y_prev + (1/2)*k2_y, z_prev + (1/2)*k2_y)
        k4_y = double_step * y_diff_lamb(x_prev + double_step, y_prev + k3_y, z_prev + k3_y)

        k1_z = double_step * z_diff_lamb(x_prev, y_prev, z_prev)
        k2_z = double_step * z_diff_lamb(x_prev + (1/2)*double_step, y_prev + (1/2)*k1_z, z_prev + (1/2)*k1_z)
        k3_z = double_step * z_diff_lamb(x_prev + (1/2)*double_step, y_prev + (1/2)*k2_z, z_prev + (1/2)*k2_z)
        k4_z = double_step * z_diff_lamb(x_prev + double_step, y_prev + k3_z, z_prev + k3_z)


        y_next = y_prev + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
        z_next = z_prev + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z)
        x_next = x_prev + double_step

        x_prev = x_next
        y_prev = y_next
        z_prev = z_next
        y_2h.append(y_next)
    
    for i in range(len(y_2h)):
        R = (y__[2*i] - y_2h[i]) / (2**p - 1)
        print("Член погрешности на шаге", 2*i, "равен:", R.evalf(), "\tразность с точным решением:", (y_accurate_lamb(x__[2*i]) - y__[2*i]).evalf())
    
    plt.figure()
    plt.plot(x_accurate_vals, y_accurate_vals, c='green') # точные значения
    plt.scatter(x__, y__, c='red') # приближения
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def adams(step: float):
    # y''=(y-(x+1)*y')/x^2
    # x лежит в [1; 2]
    x, y, z = sp.symbols('x y z', real=True)
    double_step = 2*step
    p = 4
    x_start = 1
    x_end = 2
    y_start = 2 + sp.exp(1)
    z_start = 1
    y_accurate = x + 1 + x*sp.exp(1/x)
    y_accurate_lamb = sp.lambdify(x, y_accurate)

    x__ = []
    y__ = []
    y_2h = []
    x_accurate_vals = np.linspace(x_start, x_end, 400)
    y_accurate_vals = y_accurate_lamb(x_accurate_vals)
    for i in np.linspace(x_start, x_end, int((x_end - x_start)/step)+1):
        x__.append(i)

    y_diff_func = z
    z_diff_func = (y-(x+1)*z)/(x**2)
    y_diff_lamb = sp.lambdify([x, y, z], y_diff_func)
    z_diff_lamb = sp.lambdify([x, y, z], z_diff_func)

    last_4_x = [x_start]
    last_4_y = [y_start]
    last_4_z = [z_start]
    y__.append(y_start)
    while len(last_4_x) < 4:
        k1_y = step * y_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])
        k2_y = step * y_diff_lamb(last_4_x[-1] + (1/2)*step, last_4_y[-1] + (1/2)*k1_y, last_4_z[-1] + (1/2)*k1_y)
        k3_y = step * y_diff_lamb(last_4_x[-1] + (1/2)*step, last_4_y[-1] + (1/2)*k2_y, last_4_z[-1] + (1/2)*k2_y)
        k4_y = step * y_diff_lamb(last_4_x[-1] + step, last_4_y[-1] + k3_y, last_4_z[-1] + k3_y)

        k1_z = step * z_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])
        k2_z = step * z_diff_lamb(last_4_x[-1] + (1/2)*step, last_4_y[-1] + (1/2)*k1_z, last_4_z[-1] + (1/2)*k1_z)
        k3_z = step * z_diff_lamb(last_4_x[-1] + (1/2)*step, last_4_y[-1] + (1/2)*k2_z, last_4_z[-1] + (1/2)*k2_z)
        k4_z = step * z_diff_lamb(last_4_x[-1] + step, last_4_y[-1] + k3_z, last_4_z[-1] + k3_z)


        y__.append(last_4_y[-1] + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y))
        last_4_y.append(last_4_y[-1] + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y))
        last_4_z.append(last_4_z[-1] + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z))
        last_4_x.append(last_4_x[-1] + step)
        print("y:", last_4_y[-1] + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y), "\tz:", last_4_z[-1] + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z))
    
    while(np.abs(last_4_x[-1] - x_end) > 0.000001 and last_4_x[-1] < x_end):
        y_next = last_4_y[-1] + (step/24)*(55*y_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])  -  59*y_diff_lamb(last_4_x[-2], last_4_y[-2], last_4_z[-2])  +  37*y_diff_lamb(last_4_x[-3], last_4_y[-3], last_4_z[-3])  -  9*y_diff_lamb(last_4_x[-4], last_4_y[-4], last_4_z[-4]))
        z_next = last_4_z[-1] + (step/24)*(55*z_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])  -  59*z_diff_lamb(last_4_x[-2], last_4_y[-2], last_4_z[-2])  +  37*z_diff_lamb(last_4_x[-3], last_4_y[-3], last_4_z[-3])  -  9*z_diff_lamb(last_4_x[-4], last_4_y[-4], last_4_z[-4]))
        x_next = last_4_x[-1] + step

        last_4_y.append(y_next)
        last_4_z.append(z_next)
        last_4_x.append(x_next)
        last_4_y.pop(0)
        last_4_z.pop(0)
        last_4_x.pop(0)
        y__.append(y_next)
        print("y:", y_next, "\tz:", z_next, "\tx:", last_4_x[-1])

    last_4_x = [x_start]
    last_4_y = [y_start]
    last_4_z = [z_start]
    y_2h.append(y_start)
    while len(last_4_x) < 4:
        k1_y = double_step * y_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])
        k2_y = double_step * y_diff_lamb(last_4_x[-1] + (1/2)*double_step, last_4_y[-1] + (1/2)*k1_y, last_4_z[-1] + (1/2)*k1_y)
        k3_y = double_step * y_diff_lamb(last_4_x[-1] + (1/2)*double_step, last_4_y[-1] + (1/2)*k2_y, last_4_z[-1] + (1/2)*k2_y)
        k4_y = double_step * y_diff_lamb(last_4_x[-1] + double_step, last_4_y[-1] + k3_y, last_4_z[-1] + k3_y)

        k1_z = double_step * z_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])
        k2_z = double_step * z_diff_lamb(last_4_x[-1] + (1/2)*double_step, last_4_y[-1] + (1/2)*k1_z, last_4_z[-1] + (1/2)*k1_z)
        k3_z = double_step * z_diff_lamb(last_4_x[-1] + (1/2)*double_step, last_4_y[-1] + (1/2)*k2_z, last_4_z[-1] + (1/2)*k2_z)
        k4_z = double_step * z_diff_lamb(last_4_x[-1] + double_step, last_4_y[-1] + k3_z, last_4_z[-1] + k3_z)


        y_2h.append(last_4_y[-1] + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y))
        last_4_y.append(last_4_y[-1] + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y))
        last_4_z.append(last_4_z[-1] + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z))
        last_4_x.append(last_4_x[-1] + double_step)
        print("y:", last_4_y[-1] + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y), "\tz:", last_4_z[-1] + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z))
    
    while(np.abs(last_4_x[-1] - x_end) > 0.000001 and last_4_x[-1] < x_end):
        y_next = last_4_y[-1] + (double_step/24)*(55*y_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])  -  59*y_diff_lamb(last_4_x[-2], last_4_y[-2], last_4_z[-2])  +  37*y_diff_lamb(last_4_x[-3], last_4_y[-3], last_4_z[-3])  -  9*y_diff_lamb(last_4_x[-4], last_4_y[-4], last_4_z[-4]))
        z_next = last_4_z[-1] + (double_step/24)*(55*z_diff_lamb(last_4_x[-1], last_4_y[-1], last_4_z[-1])  -  59*z_diff_lamb(last_4_x[-2], last_4_y[-2], last_4_z[-2])  +  37*z_diff_lamb(last_4_x[-3], last_4_y[-3], last_4_z[-3])  -  9*z_diff_lamb(last_4_x[-4], last_4_y[-4], last_4_z[-4]))
        x_next = last_4_x[-1] + double_step

        last_4_y.append(y_next)
        last_4_z.append(z_next)
        last_4_x.append(x_next)
        last_4_y.pop(0)
        last_4_z.pop(0)
        last_4_x.pop(0)
        y_2h.append(y_next)

    for i in range(len(y_2h)):
        R = (y__[2*i] - y_2h[i]) / (2**p - 1)
        print("Член погрешности на шаге", 2*i, "равен:", R.evalf(), "\tразность с точным решением:", (y_accurate_lamb(x__[2*i]) - y__[2*i]).evalf())
    
    # print(len(x__), len(y__))
    plt.figure()
    plt.plot(x_accurate_vals, y_accurate_vals, c='green') # точные значения
    plt.scatter(x__, y__, c='red') # приближения
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def shooting():
    # y'' = 2y/(x**2+1)
    # y'(0) = 2
    # y(1) = 3+(П/2)
    # x в [0; 1]
    x, y, z= sp.symbols('x y z', real=True)
    accuracy = 0.00001

    def f(t):
        return t**2 + t + 1 + (t**2 + 1) * np.arctan(t)

    def solving(step):
        x_start = 0
        x_end = 1
        y_end = 3 + np.pi/2
        z_start = 2
        n_mas = [0., 0.2] # y(a) <-------- подбираемые параметры для решения задачи
        solutions = []

        y__ = []
        x__ = np.linspace(x_start, x_end, int((x_end - x_start)/np.abs(step))+1)

        y_diff_func = z
        z_diff_func = 2*y/(1+x**2)
        y_diff_lamb = sp.lambdify([x, y, z], y_diff_func)
        z_diff_lamb = sp.lambdify([x, y, z], z_diff_func)

        for j in range(2):
            y__.clear()
            x_prev = x_start
            y_prev = n_mas[j]
            z_prev = z_start
            for i in range(int((x_end - x_start) / np.abs(step)) + 1):
                k1_y = step * y_diff_lamb(x_prev, y_prev, z_prev)
                k2_y = step * y_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k1_y, z_prev + (1/2)*k1_y)
                k3_y = step * y_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k2_y, z_prev + (1/2)*k2_y)
                k4_y = step * y_diff_lamb(x_prev + step, y_prev + k3_y, z_prev + k3_y)

                k1_z = step * z_diff_lamb(x_prev, y_prev, z_prev)
                k2_z = step * z_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k1_z, z_prev + (1/2)*k1_z)
                k3_z = step * z_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k2_z, z_prev + (1/2)*k2_z)
                k4_z = step * z_diff_lamb(x_prev + step, y_prev + k3_z, z_prev + k3_z)

                y_next = y_prev + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
                z_next = z_prev + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z)
                x_next = x_prev + step

                x_prev = x_next
                y_prev = y_next
                z_prev = z_next
                # y__.insert(0, y_next)
                y__.append(y_next)
            
            solutions.append(y__[-1])
        
        while np.abs(solutions[-1] - y_end) > accuracy:
            n_mas.append(n_mas[-1] - ((n_mas[-1] - n_mas[-2]) / (solutions[-1] - solutions[-2])) * (solutions[-1] - y_end))
            y__.clear()
            x_prev = x_start
            y_prev = n_mas[-1]
            z_prev = z_start
            # y__.append(y_end)
            for i in range(int((x_end - x_start) / np.abs(step)) + 1):
                k1_y = step * y_diff_lamb(x_prev, y_prev, z_prev)
                k2_y = step * y_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k1_y, z_prev + (1/2)*k1_y)
                k3_y = step * y_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k2_y, z_prev + (1/2)*k2_y)
                k4_y = step * y_diff_lamb(x_prev + step, y_prev + k3_y, z_prev + k3_y)

                k1_z = step * z_diff_lamb(x_prev, y_prev, z_prev)
                k2_z = step * z_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k1_z, z_prev + (1/2)*k1_z)
                k3_z = step * z_diff_lamb(x_prev + (1/2)*step, y_prev + (1/2)*k2_z, z_prev + (1/2)*k2_z)
                k4_z = step * z_diff_lamb(x_prev + step, y_prev + k3_z, z_prev + k3_z)

                y_next = y_prev + (1/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
                z_next = z_prev + (1/6)*(k1_z + 2*k2_z + 2*k3_z + k4_z)
                x_next = x_prev + step

                x_prev = x_next
                y_prev = y_next
                z_prev = z_next
                # y__.insert(0, y_next)
                y__.append(y_next)
                # print("y:", y_next, "\tz:", z_next)
            
            solutions.append(y__[-1])
        # print("Все подбираемые y(a):", n_mas)
        # print("Все вычисленные значения y(b):", solutions)
        # print("Точное значение y(b):", y_end)
        return x__, y__

    x_h, y_h = solving(0.1)
    x_2h, y_2h = solving(0.2)

    # Вычисление погрешности методом Рунге-Ромберга (порядок 2)
    for i in range(len(y_2h)):
        print("x=", x_2h[i], "Остаточный член погрешности (Рунге-Ромберг):", np.abs(y_h[2*i] - y_2h[i]) / 3, "\t\tРазность с точным решением", np.abs(f(x_2h[i]) - y_h[2*i]))

    plt.scatter(x_h, y_h, label="Численное решение")
    plt.plot(x_h, f(x_h), c="green", label="Точное решение")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Метод стрельбы")
    plt.grid(True)
    plt.legend()
    plt.show()


def end_diffirence():
    # y'' = 2y/(x**2+1)
    # y'(0) = 2
    # y(1) = 3+(П/2)
    # x в [0; 1]

    def p(x):
        return 0

    def q(x):
        return 2 / (1 + x**2)

    def r(x):
        return 0

    # Точное решение
    def f(x):
        return x**2 + x + 1 + (x**2 + 1) * np.arctan(x)

    def solving(h):
        a, b = 0.0, 1.0
        N = int((b - a) / h) # число разбиений
        x = np.linspace(a, b, N + 1)
        dy_a = 2.0  # y'(0) = 2
        y_b = 3 + np.pi / 2  # y(1) = 3 + π/2

        A = np.zeros(N)
        B = np.zeros(N)
        C = np.zeros(N)
        D = np.zeros(N)
        B[0] = -1/h
        C[0] = 1/h
        D[0] = dy_a
        for i in range(1, N - 1):
            A[i] = 1 / h**2 - p(x[i]) / (2 * h)
            B[i] = -2 / h**2 - q(x[i])
            C[i] = 1 / h**2 + p(x[i]) / (2 * h)
            D[i] = r(x[i])

        A[N - 1] = 1 / h**2 - p(x[N - 1]) / (2 * h)
        B[N - 1] = -2 / h**2 + q(x[N - 1])
        C[N - 1] = 0  # поскольку y_N известно
        D[N - 1] = r(x[N - 1]) - (1 / h**2 + p(x[N - 1]) / (2 * h)) * y_b

        # Решение СЛАУ методом прогонки
        for i in range(1, N):
            m = A[i] / B[i - 1]
            B[i] = B[i] - m * C[i - 1]
            D[i] = D[i] - m * D[i - 1]

        # Обратный ход
        y_2h = np.zeros((N + 1) // 2)
        y = np.zeros(N + 1)
        y[N - 1] = D[N - 1] / B[N - 1]  # Начинаем с предпоследней точки
        for i in range(N - 2, -1, -1):
            y[i] = (D[i] - C[i] * y[i + 1]) / B[i]

        # y_N = y(1)
        y[N] = y_b
        return x, y
    
    x_h, y_h = solving(0.1)
    x_2h, y_2h = solving(0.2)

    # Вычисление погрешности методом Рунге-Ромберга (порядок 2)
    for i in range(len(y_2h)):
        print("x=", x_2h[i], "Остаточный член погрешности (Рунге-Ромберг):", np.abs(y_h[2*i] - y_2h[i]) / 3, "\t\tРазность с точным решением", np.abs(f(x_2h[i]) - y_h[2*i]))

    plt.scatter(x_h, y_h, label="Численное решение")
    plt.plot(x_h, f(x_h), c="green", label="Точное решение")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Конечно-разностное решение")
    plt.grid(True)
    plt.legend()
    plt.show()


# eiler(0.1)
# runge_kutt(0.1)
# adams(0.1)
# shooting()
end_diffirence()