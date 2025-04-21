import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def newton(accuracy: float):
    #задаем функции
    x = sp.symbols('x', real=True)
    expression = x**6-5*x-2
    first_diff = sp.diff(expression, x)
    second_diff = sp.diff(first_diff, x)
    expression_labdified = sp.lambdify(x, expression, 'numpy')
    first_diff_labdified = sp.lambdify(x, first_diff, 'numpy')
    second_diff_labdified = sp.lambdify(x, second_diff, 'numpy')
    print("Заданная функция:", expression)
    print("Первая производная:", first_diff)
    print("Вторая производная:", second_diff)

    #рисуем график функции и ее производных
    x_vals = np.linspace(-2, 2, 200)
    y_expression_vals = expression_labdified(x_vals)
    y_first_diff_vals = first_diff_labdified(x_vals)
    y_second_diff_vals = second_diff_labdified(x_vals)

    plt.figure(figsize=(10, 9))
    plt.plot(x_vals, y_expression_vals, label='f(x)')
    plt.plot(x_vals, y_first_diff_vals, label="первая производная")
    plt.plot(x_vals, y_second_diff_vals, label="вторая производная")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("График функции и её производной")
    plt.legend()
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.show()

    #по графику видно, что обе производные не меняют знаки на отрезке [1, 2]
    a = 1
    b = 2
    print("Взяли отрезок [a; b]: [", a, ";", b, "]. Значение выражения f(a)*f(b) должно быть меньше нуля:", expression_labdified(a)*expression_labdified(b))
    if (expression_labdified(a)*expression_labdified(b) >= 0):
        print("Отрезок [a; b] выбран неправильно")
        exit(0)

    x_0 = 2
    print("Взяли начальную точку x_0:", x_0, ". Выражение f(x_0)*f''(x_0) должно быть больше нуля:", expression_labdified(x_0)*second_diff_labdified(x_0))
    if ((expression_labdified(x_0)*second_diff_labdified(x_0) <= 0) or (x_0 < a) or (x_0 > b)):
        print("Начальная точка x_0 выбрана неправильно или не входит в отрезок [a, b]")
        exit(0)

    print("Начинаем вычислять приближения к корню\n")

    x = x_0 - (expression_labdified(x_0) / first_diff_labdified(x_0))
    iteration = 1
    while np.abs(x - x_0) > accuracy:
        print("Итерация", iteration, "; Значение x:", x, "; Точность вычислений: ", np.abs(x - x_0), "; значение функции в точке: ", expression_labdified(x))
        x_0 = x
        x = x_0 - (expression_labdified(x_0) / first_diff_labdified(x_0))
        iteration += 1
    print("Финальная итерация", iteration, "; Значение x:", x, "; Точность вычислений: ", np.abs(x - x_0), "; значение функции в точке: ", expression_labdified(x))


def simple_iteration(accuracy: float):
    #задаем функции
    x = sp.symbols('x', real=True)
    expression = x**6-5*x-2
    equivalent_expression = np.pow((5*x+2), 1/6)
    equivalent_diff = sp.diff(equivalent_expression, x)
    expression_lambdified = sp.lambdify(x, expression, 'numpy')
    equivalent_lambdified = sp.lambdify(x, equivalent_expression, 'numpy')
    equivalent_diff_lambdified = sp.lambdify(x, equivalent_diff, 'numpy')

    #строим графики (т к один из графиков имеет степень 1/6, то могут выводится предупреждения о наличии отрицательных значений под корнем)
    x_vals = np.linspace(0, 2, 300)
    y_expression_vals = expression_lambdified(x_vals)
    y_equivalent_vals = equivalent_lambdified(x_vals)
    y_equivalent_diff_vals = equivalent_diff_lambdified(x_vals)

    plt.figure(figsize=(10, 9))
    plt.plot(x_vals, y_expression_vals, label='f(x)')
    plt.plot(x_vals, y_equivalent_vals, label="phi(x)")
    plt.plot(x_vals, y_equivalent_diff_vals, label="phi'(x)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("График функции и её эквивалента")
    plt.legend()
    plt.ylim(-5, 5)
    plt.xticks(np.arange(-7, 7, 1))
    plt.yticks(np.arange(-7, 7, 1))
    plt.grid(True)
    plt.show()

    #задаем a, b и решаем задачу
    a = 0
    b = 2
    print("Выбран отрезок [a; b]: [", a, ";", b,  "]. Проверим условие |phi'(x)| < 1 для любых x, лежащих в [a; b]:")
    for x in np.linspace(a, b, 300):
        if (np.abs(equivalent_diff_lambdified(x)) >= 1):
            print("|phi'(x)| > 1 в точке x =", x, "; Значение |phi'(x)| =", np.abs(equivalent_diff_lambdified(x)))
            exit(0)
    print("Условие выполняется. Решаем задачу")
    x_0 = a
    x = equivalent_lambdified(x_0)
    iteration = 1
    while np.abs(x - x_0) > accuracy:
        print("Итерация", iteration, "; Значение x:", x, "; Точность вычислений: ", np.abs(x - x_0), "; значение функции в точке: ", expression_lambdified(x))
        x_0 = x
        x = equivalent_lambdified(x_0)
        iteration += 1
    print("Финальная итерация", iteration, "; Значение x:", x, "; Точность вычислений: ", np.abs(x - x_0), "; значение функции в точке: ", expression_lambdified(x))


def LU_for_newton_system(J_lamb: np.matrix, f_lamb:np.array, x:np.array):
    # Используем метод LU разложения для решения системы f(x_k)+J*(delta_x)=0, в которой ищем вектор приращений delta_x для решения другой системы методом Ньютона
    # Вычисляем значения J при подставлении x
    J = np.copy(J_lamb)
    f = np.copy(f_lamb)
    J[0, 0] = J[0, 0](x[0], x[1])
    J[0, 1] = J[0, 1](x[0], x[1])
    J[1, 0] = J[1, 0](x[0], x[1])
    J[1, 1] = J[1, 1](x[0], x[1])

    # Вычисляем f при подставлении x и умножаем на -1, т к переносим f из правой части в левую
    f[0] = -1*f[0](x[0], x[1])
    f[1] = -1*f[1](x[0], x[1])
    U = J.copy()
    f = f.transpose()
    L = np.eye(2, 2)
    p = 0 #кол-во перестановок
    
    # поиск  LU разложения
    for i in range (0, 2):
        if U[i, i] == 0:
            for j in range(i, 2):
                if U[j, j] != 0:
                    buffer = np.copy(U[i])
                    U[i] = np.copy(U[j])
                    U[j] = np.copy(buffer)
                    p += 1

                    buffer = np.copy(L[i, :i])
                    L[i, :i] = np.copy(L[j, :i])
                    L[j, :i] = np.copy(buffer)

                    buffer = np.copy(f[0, i])
                    f[0, i] = np.copy(f[0, j])
                    f[0, j] = np.copy(buffer)
        for j in range(i+1, 2):
            mu_j = (U[j, i]/U[i, i])
            L[j, i] = mu_j
            U[j] = U[j]-U[i]*mu_j
    # print("Верхняя матрица:\n", U, "\nНижняя матрица:\n", L)

    z = np.zeros(2)
    delta_x = np.zeros(2)

    # решение Lz=f
    for i in range(0, 2):
        summ = 0
        for j in range(0, i):
            summ += L[i, j]*z[j]
        z[i] = f[i]-summ

    # Решение Ux=z
    for i in range(1, -1, -1):
        summ = 0
        for j in range(i+1, 2):
            summ += U[i, j]*delta_x[j]
        delta_x[i] = (1 / U[i, i])*(z[i]-summ)
    print("Решения СЛАУ в векторе delta_x: ", delta_x)

    det = np.pow(-1, p)
    for i in range(0, 2):
        det *= U[i, i]
    # print(, "Определитель: ", det)
    print("Определитель матрицы Якоби: ", det, "\nПроверка определителя: ", np.linalg.det(np.array(J, dtype=np.float64)))
    if (np.abs(det - np.linalg.det(np.array(J, dtype=np.float64))) > 0.0000001):
        print("Ошибка решения")
        exit(0)

    #проверка правильности решения с помощью вычисления вектора b
    print("Проверка правильности решения (Свободные члены, вычесленные из вектора delta_x и свободные члены вектора f):")
    for i in range (0, 2):
        answer = 0.
        for j in range(0, 2):
            answer += J[i, j]*delta_x[j]
        print(answer, "\t\t", f[i])
        if (np.abs(answer - f[i]) > 0.0000001):
            print("Ошибка решения")
            exit(0)
    return delta_x


def newton_system(accuracy: float):
    x = np.array([sp.symbols('x1', real=True), sp.symbols('x2', real=True)])
    f = np.array([3*x[0]**2-x[0]+x[1]**2-1, x[1]-sp.tan(x[0])])
    J = np.matrix([[sp.diff(f[0], x[0]), sp.diff(f[0], x[1])], 
                   [sp.diff(f[1], x[0]), sp.diff(f[1], x[1])]])
    
    f_lamb = np.array([sp.lambdify(x, f[0], 'numpy'), sp.lambdify(x, f[1], 'numpy')])
    J_lamb = np.matrix([[sp.lambdify(x, J[0, 0], 'numpy'), sp.lambdify(x, J[0, 1], 'numpy')],
                        [sp.lambdify(x, J[1, 0], 'numpy'), sp.lambdify(x, J[1, 1], 'numpy')]])
    det = (J[0, 0]*J[1, 1])-(J[0, 1]*J[1, 0])
    det_lamb = sp.lambdify(x, det, 'numpy')
    print("Система уравнений:\n", f[0], "\n", f[1])
    print("Матрица Якоби (1-е производные):\n[", J[0, 0], ",\t", J[0, 1], "]\n[", J[1, 0], ",\t", J[1, 1], "]")
    print("Определитель матрицы Якоби считается по формуле:\n", det)

    #рисуем график функции и ее производных
    x1_vals = np.linspace(-5, 5, 100)
    x2_vals = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    plt.figure(figsize=(9, 6))
    plt.contour(x1_vals, x2_vals, f_lamb[0](X1, X2), levels=[0], colors='r')
    plt.contour(x1_vals, x2_vals, f_lamb[1](X1, X2), levels=[0], colors='b')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Графики первого и второго уравнений")
    plt.xticks(np.arange(-5, 5, 1))
    plt.yticks(np.arange(-5, 5, 1))
    plt.grid(True)
    plt.show()

    x_current = np.array([0.5, 0.5])

    print("\nВычисляем delta_x для 1 итерации метода Ньютона")
    delta_x = LU_for_newton_system(J_lamb, f_lamb, x_current)
    x_current = x_current + delta_x
    iteration = 1
    while(np.linalg.norm(delta_x) >= accuracy and iteration < 30):
        print("Текущее значение x:", x_current, "\nТочность вычитслений:", np.linalg.norm(delta_x))
        iteration += 1
        print("\nВычисляем delta_x для", iteration,  "итерации метода Ньютона")
        delta_x = LU_for_newton_system(J_lamb, f_lamb, x_current)
        x_current = x_current + delta_x
    print("Итоговое решение x:", x_current, "\nТочность вычислений:", np.linalg.norm(delta_x))


def simple_iterations_system(accuracy: float):
    x = np.array([sp.symbols('x1', real=True), sp.symbols('x2', real=True)])
    f = np.array([3*x[0]**2-x[0]+x[1]**2-1, x[1]-sp.tan(x[0])])
    phi = np.array([(1+np.pow(13-12*x[1]**2, 1/2))/6, sp.tan(x[0])])
    # phi = np.array([sp.atan(x[1]), np.pow(1-3*x[0]**2+x[0], 1/2)])
    phi_diff = np.matrix([[sp.diff(phi[0], x[0]), sp.diff(phi[0], x[1])],
                          [sp.diff(phi[1], x[0]), sp.diff(phi[1], x[1])]])
    diff_norm = np.pow(np.abs(phi_diff[0, 1] * phi_diff[1, 0]), 1/2)

    
    f_lamb = np.array([sp.lambdify(x, f[0], 'numpy'), sp.lambdify(x, f[1], 'numpy')])
    phi_lamb = np.array([sp.lambdify(x[1], phi[0], 'numpy'), sp.lambdify(x[0], phi[1], 'numpy')])
    phi_diff_lamb = np.matrix([[sp.lambdify(x[1], phi_diff[0, 0], 'numpy'), sp.lambdify(x[1], phi_diff[0, 1], 'numpy')],
                          [sp.lambdify(x[0], phi_diff[1, 0], 'numpy'), sp.lambdify(x[0], phi_diff[1, 1], 'numpy')]])
    diff_norm_lamb = sp.lambdify(x, diff_norm, 'numpy')
    


    print("Система уравнений:\n", f[0], "\n", f[1])
    print("Эквивалентные функции:\n", phi[0], "\n", phi[1])
    print("Матрица производных эквивалентных функций:\n[", phi_diff[0, 0], ",\t", phi_diff[0, 1], "]\n[", phi_diff[1, 0], ",\t", phi_diff[1, 1], "]")
    print("Норма матрицы производных:\n", diff_norm)

    #рисуем график функции и ее производных (могут вылетать предупреждения о наличии отрицательных чисел под корнем)
    x1_vals = np.linspace(-2, 2, 300)
    x2_vals = np.linspace(-2, 2, 300)
    zeros = np.zeros(300)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    plt.figure(figsize=(9, 6))
    plt.title("Графики первого и второго уравнений")
    plt.contour(x1_vals, x2_vals, f_lamb[0](X1, X2), levels=[0], colors='r')
    plt.contour(x1_vals, x2_vals, f_lamb[1](X1, X2), levels=[0], colors='b')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xticks(np.arange(-2, 2, 0.5))
    plt.yticks(np.arange(-2, 2, 0.5))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.title("Графики эквивалентных функций и их производных")
    plt.plot(phi_lamb[0](x2_vals), x2_vals, color='r', linestyle='-', label='x1=phi1(x2)')
    plt.plot(x1_vals, phi_lamb[1](x1_vals), color='b', label='x2=phi2(x1)')
    plt.plot(zeros, x2_vals, linestyle='--', color='r', label="x1=phi1(x2)'x1")
    plt.plot(phi_diff_lamb[0, 1](x2_vals), x2_vals, color='r', linestyle='-.', label="x1=phi1(x2)'x2")
    plt.plot(x1_vals, phi_diff_lamb[1, 0](x1_vals), linestyle='--', color='b', label="x2=phi2(x1)'x1")
    plt.plot(x1_vals, zeros, color='b', linestyle='-.', label="x2=phi2(x1)'x2")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.xticks(np.arange(-2, 2, 0.5))
    plt.yticks(np.arange(-2, 2, 0.5))
    plt.legend()
    plt.grid(True)
    plt.show()

    Z = diff_norm_lamb(X1, X2)
    plt.figure(figsize=(9, 6))
    plt.title("График области значений нормы матрицы производных")
    contour = plt.contourf(X1, X2, Z, levels=[0, 1], colors='lightblue', alpha=0.7)
    contour_line = plt.contour(X1, X2, Z, levels=[1], colors='red', linewidths=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.xticks(np.arange(-2, 2, 0.2))
    plt.yticks(np.arange(-2, 2, 0.2))
    plt.grid(True)
    plt.show()

    print("Решаем задачу")
    x_0 = [0.5, 0.5]
    x[0] = phi_lamb[0](x_0[1])
    x[1] = phi_lamb[1](x_0[0])
    iteration = 1
    while np.linalg.norm(x - x_0) > accuracy and iteration < 30:
        print("Итерация", iteration, "; Значение x:", x, "; Точность вычислений: ", np.linalg.norm(x - x_0))
        x_0 = np.copy(x)
        x[0] = phi_lamb[0](x_0[1])
        x[1] = phi_lamb[1](x_0[0])
        iteration += 1
    print("Финальная итерация", iteration, "; Значение x:", x, "; Точность вычислений: ", np.linalg.norm(x - x_0))



simple_iterations_system(0.1)