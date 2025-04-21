import numpy as np

def LU():
    A = np.matrix([[-7., -2., -1., -4., -12.],
         [-4., 6., 0., -4., 22.],
         [-8., 2., -9., -3., 51.],
         [0., 0., -7., 1., 49.]])
    U = A[:, :-1].copy()
    b = A[:, -1:].copy()# вектор свободных членов
    b = b.transpose()
    L = np.eye(4, 4)
    E = np.eye(4, 4) # вспомогательная матрица для поиска обратной
    p = 0 #кол-во перестановок
    
    # поиск  LU разложения
    for i in range (0, 3):
        print("Итерация ", i+1, "\nВыберите номер ведущего элемента столбца", i+1, ":")
        for j in range(i, 4):
            print(j+1, ": ", U[j, i], end="\n")
        while (True):
            elem = int(input())-1
            if elem > 4 or elem < i:
                print("Неправильный номер элемента")
            elif (U[elem, i] == 0):
                print("Нельзя выбрать нулевой элемент")
            else:
                print("Выбран элемент ", U[elem, i], end="\n")
                if elem == i:
                    break
                else:
                    buffer = np.copy(U[i])
                    U[i] = np.copy(U[elem])
                    U[elem] = np.copy(buffer)
                    p += 1

                    buffer = np.copy(L[i, :i])
                    L[i, :i] = np.copy(L[elem, :i])
                    L[elem, :i] = np.copy(buffer)

                    buffer = np.copy(b[0, i])
                    b[0, i] = np.copy(b[0, elem])
                    b[0, elem] = np.copy(buffer)

                    buffer = np.copy(E[i])
                    E[i] = np.copy(E[elem])
                    E[elem] = np.copy(buffer)

                    break
        for j in range(i+1, 4):
            mu_j = (U[j, i]/U[i, i])
            #print(mu_j)
            L[j, i] = mu_j
            #print(string, end="\n\n")
            U[j] = U[j]-U[i]*mu_j
            E[j] = E[j]-E[i]*mu_j
        print("Верхняя матрица:\n", U, "\nНижняя матрица:\n", L, end="\n\n")

    z = np.zeros([1, 4])
    x = np.zeros([1, 4])

    # решение Lz=b
    for i in range(0, 4):
        summ = 0
        for j in range(0, i):
            summ += L[i, j]*z[0, j]
        z[0, i] = b[0, i]-summ

    # Решение Ux=z
    for i in range(3, -1, -1):
        summ = 0
        for j in range(i+1, 4):
            summ += U[i, j]*x[0, j]
        x[0, i] = (1 / U[i, i])*(z[0, i]-summ)
    print("Решения СЛАУ в векторе x: ", x)

    det = np.pow(-1, p)
    for i in range(0, 4):
        det *= U[i, i]
    # print(, "Определитель: ", det)
    print("Определитель матрицы A: ", det, "\nПроверка определителя: ", np.linalg.det(A[:, :-1]))

    # поиск обратной матрицы
    A_reverse = np.zeros([4, 4])
    for i in range(0, 4):
        A_reverse[3, i] = E[3, i] / U[3, 3]
        for j in range(2, -1, -1):
            summ = 0
            for k in range(3, j, -1):
                summ += A_reverse[k, i] * U[j, k]
            A_reverse[j, i] = (E[j, i] - summ) / U[j, j]
    print("Обратная матрица A^(-1):\n", A_reverse, "\nПроверка обратной матрицы:\n", np.linalg.inv(A[:, :-1]))

    #проверка правильности решения с помощью вычисления вектора b
    print("Проверка правильности решения (Свободные члены, вычесленные из вектора x и свободные члены вектора b):")
    for i in range (0, 4):
        answer = 0.
        for j in range(0, 4):
            answer += A[i, j]*x[0, j]
        print(answer, "\t\t", A[i, -1])


def running_method(n: int):
    #a = np.array([0., -9., 5., -6., 2.])
    #b = np.array([-11., 17., 20., -20., 8.])
    #c = np.array([9., 6., 8., 7., 0.])
    #d = np.array([-117., -97., -6., 59., -86.])
    a = np.empty(n)
    b = np.empty(n)
    c = np.empty(n)
    d = np.empty(n)
    print("Вводите последовательно на каждой строке элементы a, b, c и свободный член (на первой строке a = 0, на последней c = 0), разделяя элементы пробелом")
    for i in range(0, n):
        a[i], b[i], c[i], d[i] = [float(x) for x in input().split()]

    a[0] = 0
    c[-1] = 0
    print("Вектор a:", a, "\nВектор b:", b, "\nВектор c:", c, "\nВектор d (свободные члены):", d, end="\n\n")
    P = np.zeros([1, n])
    Q = np.zeros([1, n])
    x = np.zeros([1, n])

    P = P[0]
    Q = Q[0]
    x = x[0]
    
    P[0] = (-1*c[0]) / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        P[i] = (-1*c[i]) / (b[i]+(a[i]*P[i-1]))
        Q[i] = (d[i]-(a[i]*Q[i-1])) / (b[i]+(a[i]*P[i-1]))

    x[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        x[i] = P[i]*x[i+1]+Q[i]
    #print(P)
    #print(Q)
    print("Вектор решений (x):", x, end="\n")


def simple_iterations(n, accuracy: float):
    '''
    A = np.matrix([[-25., 4., -4., 9.],
         [-9., 21., 5., -6.],
         [9., 2., 19., -7.],
         [-7., 4., -7., 25.]])
    b = np.array([86., 29., 28., 68.])
    
    A = np.matrix([[10., 1., 1.],
         [2., 10., 1.],
         [2., 2., 10.]])
    b = np.array([12., 13., 14.])
    '''
    A = np.zeros([n, n])
    b = np.zeros(n)

    print("Вводите последовательно на каждой строке элементы элементы матрицы A и свободный член, разделяя элементы пробелом")
    for i in range(0, n):
        print("Элементы уравнения на", i+1, "строке:")
        inpt = input().split()
        for j in range(0, n):
            A[i, j] = float(inpt[j])
        b[i] = inpt[-1]

    for i in range(0, n):
        if A[i, i] == 0:
            for j in range(i+1, n):
                if A[j, i] != 0:
                    buffer = np.copy(A[i, i])
                    A[i, i] = np.copy(A[j, i])
                    A[j, i] = np.copy(buffer)

                    buffer = np.copy(b[i])
                    b[i] = np.copy(b[j])
                    b[j] = np.copy(buffer)
                    break
        b[i] = b[i] / A[i, i]
        for j in range(0, n):
            if j != i:
                A[i, j] = -1*A[i, j] / A[i, i]
        A[i, i] = 0
    
    x = np.copy(b)
    x = x.reshape(-1, 1) # транспонирование вектора (делаем его вертикальным)
    b = b.reshape(-1, 1) # транспонирование вектора (делаем его вертикальным)
    x_prev = np.copy(x)
    x = b + np.dot(A, x)
    # критерий окончания: норма разности x с итераций i и i-1 не должна превышать accuracy
    while np.linalg.norm(x - x_prev) > accuracy: # linalg.norm по-умолчанию использует Норму Фробениуса или 2-норма (корень суммы квадратов всех элементов матрицы/вектора)
        x_prev = np.copy(x)
        x = b + np.dot(A, x)
    
    print("Норма разности x с последней итерации:", np.linalg.norm(x - x_prev), "\nВектор решений x:\n", x, end="\n")


def zeydel(n, accuracy: float):
    '''
    A = np.matrix([[-25., 4., -4., 9.],
         [-9., 21., 5., -6.],
         [9., 2., 19., -7.],
         [-7., 4., -7., 25.]])
    b = np.array([86., 29., 28., 68.])
    
    A = np.matrix([[10., 1., 1.],
         [2., 10., 1.],
         [2., 2., 10.]])
    b = np.array([12., 13., 14.])
    '''

    size = n
    B = np.zeros([size, size]) # нижняя треугольная с нулевой диагональю
    C = np.zeros([size, size]) # верхняя треугольная с ненулевой диагональю
    E = np.eye(size)
    A = np.zeros([n, n])
    b = np.zeros(n)
    print("Вводите последовательно на каждой строке элементы элементы матрицы A и свободный член, разделяя элементы пробелом")
    for i in range(0, n):
        print("Элементы уравнения на", i+1, "строке:")
        inpt = input().split()
        for j in range(0, n):
            A[i, j] = float(inpt[j])
        b[i] = inpt[-1]

    for i in range(0, size):
        if A[i, i] == 0:
            for j in range(i+1, size):
                if A[j, i] != 0:
                    buffer = np.copy(A[i, i])
                    A[i, i] = np.copy(A[j, i])
                    A[j, i] = np.copy(buffer)

                    buffer = np.copy(b[i])
                    b[i] = np.copy(b[j])
                    b[j] = np.copy(buffer)
                    break
        b[i] = b[i] / A[i, i]
        for j in range(0, size):
            if j != i:
                A[i, j] = -1*A[i, j] / A[i, i]
            if j < i:
                B[i, j] = np.copy(A[i, j])
            else:
                C[i, j] = np.copy(A[i, j])

        A[i, i] = 0
        C[i, i] = 0
    E_B_inv = np.linalg.inv(E-B)
    x = np.copy(b)
    x = x.reshape(-1, 1) # транспонирование вектора (делаем его вертикальным)
    b = b.reshape(-1, 1) # транспонирование вектора (делаем его вертикальным)
    x_prev = np.copy(x)
    x = np.dot(E_B_inv, np.dot(C, x_prev)) + np.dot(E_B_inv, b)

    # критерий окончания: норма разности x с итераций i и i-1 не должна превышать accuracy
    while np.linalg.norm(x - x_prev) > accuracy: # linalg.norm по-умолчанию использует Норму Фробениуса или 2-норма (корень суммы квадратов всех элементов матрицы/вектора)
        x_prev = np.copy(x)
        x = np.dot(E_B_inv, np.dot(C, x_prev)) + np.dot(E_B_inv, b)
    print("Норма разности x с последней итерации:", np.linalg.norm(x - x_prev), "\nВектор решений x:\n", x, end="\n")


def rotation_method(n, accuracy: float):
    '''
    A = np.matrix([[-8., -4., 8.],
         [-4., -3., 9.],
         [8., 9., -5.]])
    
    A = np.matrix([[4., 2., 1.],
         [2., 5., 3.],
         [1., 3., 6.]])
    '''
    A = np.zeros([n, n])
    print("Вводите последовательно на каждой строке элементы элементы матрицы A, разделяя их пробелом")
    for i in range(0, n):
        print("Элементы матрицы на", i+1, "строке:")
        inpt = input().split()
        for j in range(0, n):
            A[i, j] = float(inpt[j])

    A_starting = np.copy(A)
    own_vectors = np.eye(3)
    iterations = 0
    while(True):
        iterations += 1
        U = np.eye(3)
        criterion = 0
        ii = 0
        jj = 1
        phi = 0
        for i in range(0, 3):
            for j in range(i+1, 3):
                if np.abs(A[i, j]) > np.abs(A[ii, jj]):
                    ii = i
                    jj = j
        # print(A[ii, jj], ii, jj)
        if np.abs(A[ii, ii] - A[jj, jj]) < 0.001:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2*A[ii, jj] / (A[ii, ii] - A[jj, jj]))
        # print(phi, np.sin(phi), np.cos(phi))
        U[ii, ii] = np.cos(phi)
        U[ii, jj] = -1*np.sin(phi)
        U[jj, ii] = np.sin(phi)
        U[jj, jj] = np.cos(phi)
        A = np.dot(np.dot(np.transpose(U), A), U)
        own_vectors = np.dot(own_vectors, U)
        for i in range(0, 3):
            for j in range(i+1, 3):
                criterion += np.pow(A[i, j], 2)
                # print(criterion, A[i, j])
        criterion = np.pow(criterion, 0.5)
        # print("Критерий:", criterion, "\nA:\n", A, "\nU:\n", U, end="\n\n\n")
        if (criterion <= accuracy):
            print("Собственные векторы (транспонированные):") 
            for i in range(0, 3):
                print("СВ", i, " = ", own_vectors[:, i])
            print("Собственные значения:")
            for i in range(0, 3): 
                print("СЗ", i, " = ", A[i, i])
            print("Проверка на ортогональность собственных векторов:")
            for i in range(0, 3-1):
                for j in range(i+1, 3):
                    print("(x", i, ", x", j, ") = ", np.dot(own_vectors[:, i], np.transpose(own_vectors[:, j])))
            print("Проверка на выполнение равенста A*СВ=СЗ*СВ:")
            for i in range(0, 3):
                print(np.dot(A_starting, np.transpose(own_vectors[:, i])), " = ", A[i, i] * np.transpose(own_vectors[:, i]))
            print("Кол-во итераций:", iterations, "\nПогрешность:", criterion)
            break

    
def QR(n, accuracy: float):
    '''
    A = np.matrix([[-3., 1., -1.],
         [6., 9., -4.],
         [5., -4., -8.]])
    
    A = np.matrix([[1., 3., 1.],
         [1., 1., 4.],
         [4., 3., 1.]])
    '''
    A = np.zeros([n, n])
    print("Вводите последовательно на каждой строке элементы элементы матрицы A, разделяя их пробелом")
    for i in range(0, n):
        print("Элементы матрицы на", i+1, "строке:")
        inpt = input().split()
        for j in range(0, n):
            A[i, j] = float(inpt[j])

    complex_list = list()
    A_original = np.copy(A)
    E = np.eye(3)
    first_iteration = True
    k = 0

    while(True):
        A_previous = np.copy(A)
        Q = np.eye(3)
        v = np.zeros([3, 1])
        for i in range(0, 3-1):
            v_i = np.copy(v)
            for j in range(0, i):
                v_i[j, 0] = 0
            v_i[i, 0] = A[i, i]+np.sign(A[i, i])*np.pow(sum([np.pow(A[j, i], 2) for j in range(i, 3)]), 0.5)
            for j in range(i+1, 3):
                v_i[j, 0] = A[j, i]
            
            H = E - 2*(np.dot(v_i, np.transpose(v_i)) / np.dot(np.transpose(v_i), v_i))
            A = np.dot(H, A)
            Q = np.dot(Q, H)
            # print(A, end="\n\n")
        # print(Q, "\n\n", np.linalg.inv(Q), "\n\n", np.transpose(Q))
        # print(A, "\n\n", A_original, "\n\n", np.dot(Q, A))
        R = np.copy(A)
        print("Итерация", k, ":")
        A = np.dot(R, Q)
        print(A, end="\n\n")

        float_criterion = 0
        complex_criterion = 0
        for i in range(0, 3):
            #проверка на комплексные СЗ
            if i+1 < 3:
                if (not first_iteration and np.abs(A[i+1, i]) > np.abs(A_previous[i+1, i])*2 and [i+1, i] not in complex_list):
                    complex_list.append([i+1, i])
                    print("Комплексное СЗ по элементу", i+1, i)
                elif [i+1, i] not in complex_list:
                    float_criterion += np.pow(A[i+1, i], 2)
            
            for j in range(i+2, 3):
                float_criterion += np.pow(A[j, i], 2)
        float_criterion = np.pow(float_criterion, 0.5)

        #Вычисление разности вещественных частей комплексных СЗ с разных итераций для критерия окончания. В каждом элементе complex_list хранится индекс [i+1, i]
        for i in complex_list:
            complex_criterion += np.abs((A_previous[i[0]-1, i[1]] + A_previous[i[0], i[1]+1]) - (A[i[0]-1, i[1]] + A[i[0], i[1]+1])) / 2

        if float_criterion <= accuracy and complex_criterion <= accuracy:
            print("Вещественные собственные значения матрицы:")
            for i in range(0, 3):
                if ([i+1, i] not in complex_list) and ([i, i-1] not in complex_list):
                    print(A[i, i])
            print("Комплексные собственные значения матрицы:")
            #В каждом элементе complex_list хранится индекс [i+1, i]
            for i in complex_list:
                # решаем уравнение вида ax^2+bx+c=0 где x - пара комплексных СЗ; коэффициенты a, b, c вычисляются ниже
                # a = 1
                # b = -1(A[ii]+A[jj])
                # c = (A[ii]*A[jj] - A[ij]*A[ji])
                # где j = i+1
                b = -1*(A[i[0]-1, i[1]] + A[i[0], i[1]+1]) 
                c = A[i[0]-1, i[1]]*A[i[0], i[1]+1] - A[i[0]-1, i[1]+1]*A[i[0], i[1]]
                D = np.pow(b, 2) - 4 * c
                x1 = (-1*b + np.lib.scimath.sqrt(D)) / 2
                x2 = (-1*b - np.lib.scimath.sqrt(D)) / 2
                print(x1)
                print(x2)
            break
        k += 1
        first_iteration = False

QR(3, 0.01)