import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def lagranj_interpolation(x_list: float):
    x = sp.symbols('x', real=True)
    y = 1/(x**2)
    y_lamb = sp.lambdify(x, y, 'numpy')
    x_error = 0.8
    print("Исходная функция: y=", y)

    # получаем значения функции y(x) в каждой точке x
    y_list= list()
    for i in x_list:
        y_list.append(y_lamb(i))

    # составляем интерполяционный многочлен Лагранжа
    L = 0
    for i in range(len(y_list)):
        li = 1
        for j in range(len(y_list)):
            if j == i:
                continue
            li *= (x - x_list[j])/(x_list[i] - x_list[j])
        L += y_list[i] * li
        L = sp.simplify(L)
    L_lamb = sp.lambdify(x, L, 'numpy')
    print("Интерполяционный многочлен Лагранжа имеет вид: L=", L)
    print("Проверим, что полученный многочлен и начальная функция имеют одинаковые значения в определенных точках:")
    for i in x_list:
        print("При x=", i, ":\ty(x)=", y_lamb(i), ",\tL(x)=", L_lamb(i))
    print("Погрешность в точке", x_error, ": ", np.abs(y_lamb(x_error)-L_lamb(x_error)))


def divided_difference(function, start_idex, end_index, x_list):
    ''' Как считаются разделенные разности:
    # разделенные разности 0-го порядка
    f0 = y_lamb(x_list[0])
    f1 = y_lamb(x_list[1])
    f2 = y_lamb(x_list[2])
    f3 = y_lamb(x_list[3])
    # разделенные разности 1-го порядка
    f01 = (f0 - f1)/(x_list[0] - x_list[1])
    f12 = (f1 - f2)/(x_list[1] - x_list[2])
    f23 = (f2 - f3)/(x_list[2] - x_list[3])
    # разделенные разности 2-го порядка
    f02 = (f01 - f12)/(x_list[0] - x_list[2])
    f13 = (f12 - f23)/(x_list[1] - x_list[3])
    # разделенные разности 3-го порядка
    f03 = (f02 - f13)/(x_list[0] - x_list[3])
    '''
    if start_idex < end_index:
        return (divided_difference(function, start_idex, end_index-1, x_list) - divided_difference(function, start_idex+1, end_index, x_list)) / (x_list[start_idex] - x_list[end_index])
    elif start_idex == end_index:
        return function(x_list[start_idex])


def newton_interpolation(x_list: float):
    x = sp.symbols('x', real=True)
    y = 1/(x**2)
    y_lamb = sp.lambdify(x, y, 'numpy')
    x_error = 0.8
    print("Исходная функция: y=", y)

    # получаем значения функции y(x) в каждой точке x
    y_list= list()
    for i in x_list:
        y_list.append(y_lamb(i))

    # Пример итогового интерполяционного многочлена Ньютона:
    # P = f0+(x-x_list[0])*f01+(x-x_list[0])*(x-x_list[1])*f02+(x-x_list[0])*(x-x_list[1])*(x-x_list[2])*f03

    # Вычислим интерполяционный многочлен Ньютона
    P = divided_difference(y_lamb, 0, 0, x_list)
    for i in range(1, 4):
        coef = 1
        for j in range(i):
            coef *= x-x_list[j]
        P += coef*divided_difference(y_lamb, 0, i, x_list)
    P = sp.simplify(P)
    P_lamb = sp.lambdify(x, P, 'numpy')
    print("Интерполяционный многочлен Ньютона имеет вид: P=", P)
    print("Проверим, что полученный многочлен и начальная функция имеют одинаковые значения в определенных точках:")
    for i in x_list:
        print("При x=", i, ":\ty(x)=", y_lamb(i), ",\tP(x)=", P_lamb(i))
    print("Погрешность в точке", x_error, ": ", np.abs(y_lamb(x_error)-P_lamb(x_error)))
    

def cube_spline():
    xi = [0.1, 0.5, 0.9, 1.3, 1.7]
    fi = [100., 4., 1.2346, 0.59172, 0.34602]
    x_calculate = 0.8
    n = len(xi)
    ai = np.zeros(n-1)
    bi = np.zeros(n-1)
    ci = np.zeros(n-1)
    di = np.zeros(n-1)
    Matrix = np.zeros((n, n))
    free_numbers = np.zeros(n)

    h = lambda i: xi[i+1]-xi[i] # возможно тут должен быть abs
    delta = lambda i: (fi[i+1]-fi[i])/h(i)

    free_numbers[0] = 0
    free_numbers[n-1] = 0
    Matrix[0,0] = 1
    Matrix[n-1,n-1] = 1
    for i in range(1, n-1):
        Matrix[i, i-1] = h(i-1)
        Matrix[i, i] = 2*(h(i-1)+h(i))
        Matrix[i, i+1] = h(i)
        free_numbers[i] = 3*(delta(i)-delta(i-1))
    ci = np.linalg.solve(Matrix, free_numbers)

    for i in range(n-1):
        ai[i] = fi[i]
        bi[i] = delta(i)-(h(i)/3)*(2*ci[i]+ci[i+1])
        di[i] = (ci[i+1]-ci[i])/(3*h(i))
    print("a = ", ai, "\nb = ", bi, "\nc = ", ci, "\nd = ", di)

    # считаем значение сплайна в заданной точке
    for i in range(n-1):
        if x_calculate < xi[i+1]:
            x_cur = x_calculate - xi[i]
            print("Значение сплайна в точке x=", x_calculate, ":\n", ai[i] + bi[i]*x_cur + ci[i]*x_cur**2 + di[i]*x_cur**3)
            break

    # строим плотную сетку и считаем значения сплайна во всех точках
    x_dense = np.linspace(xi[0], xi[-1], 400)
    y_dense = np.empty_like(x_dense)
    for i in range(len(ai)):
        mask = (x_dense >= xi[i]) & (x_dense <= xi[i+1])
        x_cur = x_dense[mask] - xi[i]
        y_dense[mask] = ai[i] + bi[i]*x_cur + ci[i]*x_cur**2 + di[i]*x_cur**3

    # рисуем график
    plt.figure()
    plt.plot(x_dense, y_dense)
    plt.scatter(xi, fi)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Natural Cubic Spline Interpolation')
    plt.show()


def mnk(): # для бОльшей точности можно домножить все степени при x на -1. Тогда получим многочлен F(x)=a_0+a_1*x^-1+a_2*x^-2+...+a_n*x^-n
    xi = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    fi = [100., 4., 1.2346, 0.59172, 0.34602, 0.22676]
    x = sp.symbols('x', real=True)
    n = 2 # здесь задаем степень многочлена
    system_matrix = np.zeros((n+1, n+1))
    free_numbers = np.zeros(n+1)

    for i in range(n+1):
        free_numbers[i] = sum(fi[k]*xi[k]**(i) for k in range(len(xi))) # можно домножить степень x на -1
        for j in range(n+1):
            system_matrix[i, j] = sum(elem**(i+j) for elem in xi) # можно домножить степень elem на -1, т. е. elem**(-i-j)
    
    ai = np.linalg.solve(system_matrix, free_numbers)
    F = sum(ai[i]*x**(i) for i in range(len(ai))) # можно домножить степень x на -1
    F_lambd = sp.lambdify(x, F, 'numpy')
    print("Приближающий многочлен степени", n, ":\nF(x)=", F)
    error = sum((F_lambd(xi[i]) - fi[i])**2 for i in range(len(fi)))
    print("Сумма квадратов ошибок:\n", error)

    x_vals = np.linspace(xi[0], xi[-1], 400)
    y_vals = F_lambd(x_vals)

    # рисуем график
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.scatter(xi, fi)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('МНК')
    plt.show()


def differentiation():
    xi = [1., 1.2, 1.4, 1.6, 1.8]
    fi = [2., 2.1344, 2.4702, 2.9506, 3.5486]
    x = sp.symbols('x', real=True)
    x_calculate = 1.4
    l = 0

    for i in range(1, len(xi)):
        if xi[i] > x_calculate:
            l = i - 1
            break
    
    print("Границы отрезка: [", xi[l], ";", xi[l+1], "]")
    phi = fi[l] + (fi[l+1]-fi[l])/(xi[l+1]-xi[l])*(x-xi[l]) + ((fi[l+2]-fi[l+1])/(xi[l+2]-xi[l+1])-(fi[l+1]-fi[l])/(xi[l+1]-xi[l]))/(xi[l+2]-xi[l])*(x-xi[l])*(x-xi[l+1])
    phi_diff_1 = sp.diff(phi, x)
    phi_diff_2 = sp.diff(phi_diff_1, x)
    print("Первая производная в точке x=", x_calculate, ":\n", sp.lambdify(x, phi_diff_1)(x_calculate))
    print("Вторая производная в точке x=", x_calculate, ":\n", phi_diff_2)


def rectangle_integrate():
    x = sp.symbols('x', real=True)
    f = sp.sqrt(16-x**2)
    f_lambd = sp.lambdify(x, f, 'numpy')
    x_0 = -2
    x_k = 2
    h_1 = 1.
    h_2 = 0.5
    p = 2 # порядок аппроксимации
    N_1 = np.abs(x_k - x_0) // h_1
    N_2 = np.abs(x_k - x_0) // h_2
    xi_1 = [x_0+h_1*i for i in range(0, int(N_1)+1)]
    xi_2 = [x_0+h_2*i for i in range(0, int(N_2)+1)]
    print("Концы отрезков с шагом", h_1, ":\n", xi_1)
    print("Концы отрезков с шагом", h_2, ":\n", xi_2)
    result_1 = sum(h_1*f_lambd((xi_1[i-1]+xi_1[i])/2) for i in range(1, len(xi_1)))
    result_2 = sum(h_2*f_lambd((xi_2[i-1]+xi_2[i])/2) for i in range(1, len(xi_2)))
    print("Результат интегрирования формулой прямоугольников с шагом", h_1, ":\n", result_1)
    print("Результат интегрирования формулой прямоугольников с шагом", h_2, ":\n", result_2)
    runge_romberg = result_2 + (result_2 - result_1) / (2**p - 1)
    print("Результат интегрирования после уточнения методом Рунге-Ромберга:\n", runge_romberg)
    print("Разница между вычислениями:\n", np.abs(result_2 - runge_romberg))


def trapeze_integrate():
    x = sp.symbols('x', real=True)
    f = sp.sqrt(16-x**2)
    f_lambd = sp.lambdify(x, f, 'numpy')
    x_0 = -2
    x_k = 2
    h_1 = 1.
    h_2 = 0.5
    p = 2 # порядок аппроксимации
    N_1 = np.abs(x_k - x_0) // h_1
    N_2 = np.abs(x_k - x_0) // h_2
    xi_1 = [x_0+h_1*i for i in range(0, int(N_1)+1)]
    xi_2 = [x_0+h_2*i for i in range(0, int(N_2)+1)]
    print("Концы отрезков с шагом", h_1, ":\n", xi_1)
    print("Концы отрезков с шагом", h_2, ":\n", xi_2)
    result_1 = 0.5*sum(h_1*(f_lambd(xi_1[i]) + f_lambd(xi_1[i-1])) for i in range(1, len(xi_1)))
    result_2 = 0.5*sum(h_2*(f_lambd(xi_2[i]) + f_lambd(xi_2[i-1])) for i in range(1, len(xi_2)))
    print("Результат интегрирования формулой трапеции с шагом", h_1, ":\n", result_1)
    print("Результат интегрирования формулой трапеции с шагом", h_2, ":\n", result_2)
    runge_romberg = result_2 + (result_2 - result_1) / (2**p - 1)
    print("Результат интегрирования после уточнения методом Рунге-Ромберга:\n", runge_romberg)
    print("Разница между вычислениями:\n", np.abs(result_2 - runge_romberg))


def simpson_integrate():
    x = sp.symbols('x', real=True)
    f = sp.sqrt(16-x**2)
    f_lambd = sp.lambdify(x, f, 'numpy')
    x_0 = -2
    x_k = 2
    h_1 = 1.
    h_2 = 0.5
    p = 4 # порядок аппроксимации
    N_1 = np.abs(x_k - x_0) // h_1
    N_2 = np.abs(x_k - x_0) // h_2
    xi_1 = [x_0+h_1*i for i in range(0, int(N_1)+1)]
    xi_2 = [x_0+h_2*i for i in range(0, int(N_2)+1)]
    print("Концы отрезков с шагом", h_1, ":\n", xi_1)
    print("Концы отрезков с шагом", h_2, ":\n", xi_2)
    result_1 = (h_1/3)*sum(f_lambd(xi_1[i-1])+4*f_lambd(xi_1[i])+f_lambd(xi_1[i+1]) for i in range(1, len(xi_1)-1, 2))
    result_2 = (h_2/3)*sum(f_lambd(xi_2[i-1])+4*f_lambd(xi_2[i])+f_lambd(xi_2[i+1]) for i in range(1, len(xi_2)-1, 2))
    print("Результат интегрирования формулой Симпсона с шагом", h_1, ":\n", result_1)
    print("Результат интегрирования формулой Симпсона с шагом", h_2, ":\n", result_2)
    runge_romberg = result_2 + (result_2 - result_1) / (2**p - 1)
    print("Результат интегрирования после уточнения методом Рунге-Ромберга:\n", runge_romberg)
    print("Разница между вычислениями:\n", np.abs(result_2 - runge_romberg))

# lagranj_interpolation([0.1, 0.5, 0.9, 1.3])
# lagranj_interpolation([0.1, 0.5, 1.1, 1.3])
# newton_interpolation([0.1, 0.5, 0.9, 1.3])
# newton_interpolation([0.1, 0.5, 1.1, 1.3])
# cube_spline()
# mnk()
# differentiation()
rectangle_integrate()
trapeze_integrate()
simpson_integrate()