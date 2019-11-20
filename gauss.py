# TO BE COMPILED WITH PYTHON >= 3.0

from fraction import *
import matplotlib.pyplot as plt
import math
import time
import random as rnd
from decimal import *
from typing import TypeVar, Generic

T = TypeVar('T')


class Matrix:
    def __init__(self, T, n):
        self.T = T
        self.n = n
        self.A = [[T(rnd.randint(1, 1000)) for i in range(n)] for j in range(n)]
        self.X = [T(rnd.randint(1, 1000)) for i in range(n)]
        self.B = [self.T(0) for i in range(n)]
        self.C = [[T(0) for i in range(self.n + 1)] for j in range(self.n + 1)]

        self.find_B(True)

    def find_B(self, new_mults):
        n = self.n
        for i in range(n):
            for j in range(n):
                self.B[i] += self.A[i][j] * self.X[j]

        self.create_C_matrix(new_mults)

    def convert_types(self, T):
        n = self.n
        self.C = [[T(0) for i in range(self.n + 1)] for j in range(self.n + 1)]
        m = len(self.C[0])
        newX = []
        newB = []

        if isinstance(self.A[0][0], Fraction):
            for i in range(n):
                for j in range(n):
                    self.A[i][j] = self.A[i][j].decimal
                newX.append(self.X[i].decimal)
                newB.append(self.B[i].decimal)
        else:
            for i in range(n):
                for j in range(n):
                    self.A[i][j] = T(self.A[i][j])
                newX.append(T(self.X[i]))
                newB.append(T(self.B[i]))
        self.X = newX
        self.B = newB
        self.find_B(False)

    def create_C_matrix(self, new_mults):
        if new_mults:
            r1 = self.generate_mult()
            r2 = self.generate_mult()
        else:
            r1 = 1
            r2 = 1
        for i in range(self.n):
            for j in range(self.n):
                self.C[i][j] = self.A[i][j] * r1
        for i in range(self.n):
            self.C[i][self.n] = self.B[i] * r2

    def print_A(self):
        s = "MATRIX A\n"
        for i in range(self.n):
            for j in range(self.n):
                s += str(self.A[i][j]) + "   "
            s += "\n"
        print(s)

    def print_B(self):
        s = "MATRIX B\n"
        for i in range(self.n):
            s += str(self.B[i]) + "\n"
        print(s)

    def print_C(self):
        s = "MATRIX C\n"
        for i in range(self.n):
            for j in range(self.n + 1):
                s += str(self.C[i][j]) + "   "
            s += "\n"
        print(s)

    def print_X(self):
        s = "MATRIX X\n"
        for i in range(self.n):
            s += str(self.X[i]) + "   "
        s += "\n"
        print(s)

    def print_All(self):
        self.print_A()
        self.print_B()
        self.print_C()
        self.print_X()

    def generate_mult(self):
        return self.T(rnd.randint(-65536, 65535) / 65536)

    def swap_rows(self, C, current_i, current_j):
        n = len(C)
        m = len(C[0])
        T = self.T

        max_index = 0
        max1 = T(0)
        found = False
        for i in range(current_i, n):
            if T(C[i][current_j]) > max1:
                max1 = C[i][current_j]
                max_index = i
                found = True
        if found:
            for j in range(m):
                temp = C[current_i][j]
                C[current_i][j] = C[max_index][j]
                C[max_index][j] = temp

        self.print_C()
        return C

    def swap_all(self, C, current_i, current_j):
        n = len(C)
        m = len(C[0])
        T = self.T

        max1 = 0
        max_row = 0
        max_col = 0
        found = False

        for i in range(current_i, n):
            for j in range(current_j, m):
                if C[i][j] > max1:
                    max1 = C[i][j]
                    max_row = i
                    max_col = j
                    found = True
        if found:
            for j in range(m):
                temp = C[current_i][j]
                C[current_i][j] = C[max_row][j]
                C[max_row][j] = temp
            for i in range(n):
                temp = C[i][current_j]
                C[i][current_j] = C[i][max_col]
                C[i][max_col] = temp

        self.print_C()
        return C

    def gauss(self):
        n = self.n
        ret = [self.T(0) for i in range(n)]
        C = self.C
        m = len(C)

        print("C BEFORE:\n")
        self.print_C()

        for i in range(n):
            if C[i][i] == 0:
                return
            first = C[i][i]
            for k in range(m):
                C[i][k] = C[i][k] / first

            for j in range(i + 1, n + 1):
                for k in range(m):
                    C[j][k] -= C[i][k] * C[j][i]

        for i in range(n - 1, -1, -1):
            ret[i] = C[i][n] / C[i][i]
            for k in range(i - 1, -1, -1):
                C[k][n] -= C[k][i] * ret[i]
        print("C AFTER:\n")
        self.print_C()
        self.print_X()
        return ret

    def gauss_part(self):
        n = self.n
        ret = [self.T(0) for i in range(n)]
        C = self.C
        m = len(C)

        print("C BEFORE:\n")
        self.print_C()

        for i in range(n):

            C = self.swap_rows(C, i, i) ## dodane

            if C[i][i] == 0:
                C = self.swap_rows(C, i, i)
            first = C[i][i]
            for k in range(m):
                C[i][k] /= first

            for j in range(i + 1, n + 1):
                for k in range(m):
                    C[j][k] -= C[i][k] * C[j][i]
            self.print_C()

        for i in range(n - 1, -1, -1):
            ret[i] = C[i][n] / C[i][i]
            for k in range(i - 1, -1, -1):
                C[k][n] -= C[k][i] * ret[i]

        print("C AFTER:\n")
        self.print_C()
        self.print_X()
        return ret

    def gauss_full(self):
        n = self.n
        ret = [self.T(0) for i in range(n)]
        C = self.C
        m = len(C)

        print("C BEFORE:\n")
        self.print_C()

        for i in range(n):

            C = self.swap_all(C, i, i) ## dodane

            if C[i][i] == 0:
                C = self.swap_all(C, i, i)

            first = C[i][i]
            for k in range(m):
                C[i][k] /= first

            for j in range(i + 1, n + 1):
                for k in range(m):
                    C[j][k] -= C[i][k] * C[j][i]
            self.print_C()

        for i in range(n - 1, -1, -1):
            ret[i] = C[i][n] / C[i][i]
            for k in range(i - 1, -1, -1):
                C[k][n] -= C[k][i] * ret[i]

        print("C AFTER:\n")
        self.print_C()
        self.print_X()
        return ret


class Tests:
    global sizes_array
    global results_Q1_method1, results_Q1_method2
    global results_Q2_tf, results_Q2_td, results_Q2_tc

    def __init__(self):
        self.tf_result = 0
        self.tc_result = 0
        self.td_result = 0


    @staticmethod
    def count_error(X, Xprim, size):  # counting an error with fraction matrix result usage
        error_summary = 0
        if isinstance(X[0], Fraction):
            for i in range(size):
                error_summary = (X[i] - Xprim[i]).abs()
                size_fract = Fraction(size)
            return (error_summary / size_fract)
        else:
            for i in range(size):
                error_summary = abs(X[i] - Xprim[i])
            return math.log(error_summary / size)

    # zbadanie czasu i bledu dla n = size_large dla macierzy kazdego typu z uzyciem kazdej metody

    def test_1(self):  # dla odpowiedzenia na H1, H2, H3
        # tc_error_sum_method1 = 0
        # tc_error_sum_method2 = 0
        # tc_error_sum_method3 = 0

        tf_error_sum_method1 = 0
        tf_error_sum_method2 = 0
        tf_error_sum_method3 = 0

        td_error_sum_method1 = 0
        td_error_sum_method2 = 0
        td_error_sum_method3 = 0

        tf_time_sum_method1 = 0
        tf_time_sum_method2 = 0
        tf_time_sum_method3 = 0
        #
        # tc_time_sum_method1 = 0
        # tc_time_sum_method2 = 0
        # tc_time_sum_method3 = 0

        td_time_sum_method1 = 0
        td_time_sum_method2 = 0
        td_time_sum_method3 = 0

        size = 50
        repeats = 5
        for i in range(repeats):
            getcontext().prec = 7
            matrix = Matrix(Decimal, size)  # Tutaj podmienic rozmiar macierzy (zaleznie od sizes_array, czy jak wolisz)

            ####################################################3 stare
            # METODA I gauss
            # typ ulamkowy

            # start_time = time.time()
            # self.tc_result = matrix.gauss()
            # elapsed_time = time.time() - start_time
            # tc_time_method1 = elapsed_time
            #
            # tc_error_method1 = self.count_error(self.tc_result, matrix.X, size)

            # typ pojedynczej prezycji
            getcontext().prec = 7
            matrix.convert_types(Decimal)

            start_time = time.time()
            self.tf_result = matrix.gauss()
            elapsed_time = time.time() - start_time
            tf_time_method1 = elapsed_time

            tf_error_method1 = self.count_error(self.tf_result, matrix.X, size)

            # typ podwojnej prezycji
            getcontext().prec = 16
            matrix.convert_types(Decimal)

            start_time = time.time()
            self.td_result = matrix.gauss()
            elapsed_time = time.time() - start_time
            td_time_method1 = elapsed_time

            td_error_method1 = self.count_error(self.td_result, matrix.X, size)

            # METODA II gauss_part
            # typ ulamkowy

            # matrix.convert_types(Fraction)
            #
            # start_time = time.time()
            # self.tc_result = matrix.gauss_part()
            # elapsed_time = time.time() - start_time
            # tc_time_method2 = elapsed_time
            #
            # tc_error_method2 = self.count_error(self.tc_result, matrix.X, size)

            # typ pojedynczej precyzji
            getcontext().prec = 7
            matrix.convert_types(Decimal)

            start_time = time.time()
            self.tf_result = matrix.gauss_part()
            elapsed_time = time.time() - start_time
            tf_time_method2 = elapsed_time

            tf_error_method2 = self.count_error(self.tf_result, matrix.X, size)

            # typ podwojnej precyzji
            getcontext().prec = 16
            matrix.convert_types(Decimal)

            start_time = time.time()
            self.td_result = matrix.gauss_part()
            elapsed_time = time.time() - start_time
            td_time_method2 = elapsed_time

            td_error_method2 = self.count_error(self.td_result, matrix.X, size)

            # metoda III gauss_full
            # typ ulamkowy
            # matrix.convert_types(Fraction)
            #
            # start_time = time.time()
            # self.tc_result = matrix.gauss_full()
            # elapsed_time = time.time() - start_time
            # tc_time_method3 = elapsed_time
            #
            # tc_error_method3 = self.count_error(self.tc_result, matrix.X, size)

            # typ pojedynczej precyzji
            getcontext().prec = 7
            matrix.convert_types(Decimal)

            start_time = time.time()
            self.tf_result = matrix.gauss_full()
            elapsed_time = time.time() - start_time
            tf_time_method3 = elapsed_time

            tf_error_method3 = self.count_error(self.tf_result, matrix.X, size)

            # typ podwojnej precyzji
            getcontext().prec = 16
            matrix.convert_types(Decimal)

            start_time = time.time()
            self.td_result = matrix.gauss_full()
            elapsed_time = time.time() - start_time
            td_time_method3 = elapsed_time

            td_error_method3 = self.count_error(self.td_result, matrix.X, size)

            # dla policzenia srednich bledow

            # tc_error_sum_method1 += tc_error_method1
            # tc_error_sum_method2 += tc_error_method2
            # tc_error_sum_method3 += tc_error_method3

            tf_error_sum_method1 += tf_error_method1
            tf_error_sum_method2 += tf_error_method2
            tf_error_sum_method3 += tf_error_method3

            td_error_sum_method1 += td_error_method1
            td_error_sum_method2 += td_error_method2
            td_error_sum_method3 += td_error_method3

            # dla policzenia srednich czasow

            tf_time_sum_method1 += tf_time_method1
            tf_time_sum_method2 += tf_time_method2
            tf_time_sum_method3 += tf_time_method3

            # tc_time_sum_method1 += tc_time_method1
            # tc_time_sum_method2 += tc_time_method2
            # tc_time_sum_method3 += tc_time_method3

            td_time_sum_method1 += td_time_method1
            td_time_sum_method2 += td_time_method2
            td_time_sum_method3 += td_time_method3

        # tc_error_sum_method1 /= repeats
        # tc_error_sum_method2 /= repeats
        # tc_error_sum_method3 /= repeats

        tf_error_sum_method1 /= repeats
        tf_error_sum_method2 /= repeats
        tf_error_sum_method3 /= repeats

        td_error_sum_method1 /= repeats
        td_error_sum_method2 /= repeats
        td_error_sum_method3 /= repeats

        tf_time_sum_method1 /= repeats
        tf_time_sum_method2 /= repeats
        tf_time_sum_method3 /= repeats

        # tc_time_sum_method1 /= repeats
        # tc_time_sum_method2 /= repeats
        # tc_time_sum_method3 /= repeats

        td_time_sum_method1 /= repeats
        td_time_sum_method2 /= repeats
        td_time_sum_method3 /= repeats


        plt.scatter([1, 2, 3], [tf_error_sum_method1, tf_error_sum_method2, tf_error_sum_method3],
                    label="dla dokladnosci pojedynczej precyzji", marker=".")
        plt.scatter([1, 2, 3], [td_error_sum_method1, td_error_sum_method2, td_error_sum_method3],
                    label="dla dokladnosci podwojnej precyzji", marker=".")
        # plt.scatter([tc_error_sum_method1, tc_error_sum_method2, tc_error_sum_method3], [1, 2, 3],
        #             label="dla typu fraction", marker=".")
        plt.xlabel('Numery metod 1-G, 2-PG, 3-FG')
        plt.ylabel('Srednia logarytmu z bledu dokladnosci')
        plt.title('Wykres pokazujacy log z dokladnosci wynikow\n w zaleznosci od metody i rodzaju precyzji danych')
        plt.legend()
        plt.show()

        plt.scatter([1, 2, 3], [tf_time_sum_method1, tf_time_sum_method2, tf_time_sum_method3],
                    label="dla dokladnosci pojedynczej precyzji", marker=".")
        plt.scatter([1, 2, 3], [td_time_sum_method1, td_time_sum_method2, td_time_sum_method3],
                    label="dla dokladnosci podwojnej precyzji", marker=".")
        # plt.scatter([tc_time_sum_method1, tc_time_sum_method2, tc_time_sum_method3], [1, 2, 3],
        #             label="dla dokladnosci własnego typu",
        #             marker=".")
        plt.xlabel('Numery metod 1-G, 2-PG, 3-FG')
        plt.ylabel('Czas potrzbeny na wyliczenie metod')
        plt.title('Wykres pokazujacy czas obliczania\n w zaleznosci od metody i rodzaju dokladnosci danych')
        plt.legend()
        plt.show()

    def test_2(self):  # dla odpowiedzi na Q1
        getcontext().prec = 16
        repeats = 5

        for size in sizes_array:
            results_Q1_method1_sum = 0
            results_Q1_method2_sum = 0
            results_Q2_td_sum = 0
            for i in range(repeats):
                matrix = Matrix(Decimal, size)
                # metoda gauss
                start_time = time.time()
                self.td_result = matrix.gauss()
                elapsed_time = time.time() - start_time
                results_Q2_td_sum += elapsed_time

                results_Q1_method1_sum += (self.count_error(self.td_result, matrix.X, size))

                # metoda gauss_part
                self.td_result = matrix.gauss_part()

                results_Q1_method2_sum += (self.count_error(self.td_result, matrix.X, size))

            results_Q1_method1.append(results_Q1_method1_sum/repeats)
            results_Q1_method2.append(results_Q1_method2_sum/repeats)
            results_Q2_td.append(results_Q2_td_sum/repeats) # liczymy juz czas ktory bedzie wykorzystywany w test_3

        plt.scatter(sizes_array, results_Q1_method1, label="dla metody G", marker=".")
        plt.scatter(sizes_array, results_Q1_method2, label="dla metody GP", marker=".")
        plt.xlabel("Rozmiary macierzy")
        plt.ylabel('Logarytm ze sredniego bledu dokladnosci')
        plt.title(
            'Wykres pokazujacy dokladnosc wynikow\n w zaleznosci od ilosci skladnikow dla typu podwojnej precyzji')
        plt.legend()
        plt.show()

    def test_3(self):  # dla odpowiedzi na Q2
        getcontext().prec = 7
        repeats = 5

        for size in sizes_array:  # dla metody G-gauss
            results_Q2_tf_sum = 0
            results_Q2_tc_sum = 0
            for i in range(repeats):
                getcontext().prec = 7
                matrix = Matrix(Decimal, size)

                start_time = time.time()
                self.tf_result = matrix.gauss()
                elapsed_time = time.time() - start_time
                results_Q2_tf_sum += elapsed_time

                # matrix.convert_types(Fraction)
                # start_time = time.time()
                # self.tc_result = matrix.gauss()
                # elapsed_time = time.time() - start_time
                # results_Q2_tf_sum += elapsed_time

            results_Q2_tf.append(results_Q2_tf_sum/repeats)
            results_Q2_tc.append(results_Q2_tc_sum/repeats)

        # czasy dla tej metody dla typow frac i double policzylismy w test_2

        plt.scatter(sizes_array, results_Q2_tf, label="dla typu pojedynczej precyzji", marker=".")
        plt.scatter(sizes_array, results_Q2_td, label="dla typu podwojnen precyzji", marker=".") # policzone w test_2
        # plt.scatter(results_Q2_tc, sizes_array, label="dla typu fraction", marker=".")
        plt.xlabel("Rozmiary macierzy")
        plt.ylabel('Czas trwania obliczania metody G-gauss')
        plt.title('Wykres pokazujacy czas obliczania w zaleznosci od roznych typow macierzy')
        plt.legend()
        plt.show()

    def test_4(self):  # dla odpowiedzenia na H1, H2, H3

        tf_time_sum_method1 = 0
        tf_time_sum_method2 = 0
        tf_time_sum_method3 = 0

        # tc_time_sum_method1 = 0
        # tc_time_sum_method2 = 0
        # tc_time_sum_method3 = 0

        td_time_sum_method1 = 0
        td_time_sum_method2 = 0
        td_time_sum_method3 = 0

        size = 500
        getcontext().prec = 7
        matrix = Matrix(Decimal, size)  # Tutaj podmienic rozmiar macierzy (zaleznie od sizes_array, czy jak wolisz)

        ####################################################3 stare
        # METODA I gauss
        # typ ulamkowy

        # start_time = time.time()
        # self.tc_result = matrix.gauss()
        # elapsed_time = time.time() - start_time
        # tc_time_method1 = elapsed_time

        # typ pojedynczej prezycji
        getcontext().prec = 7
        matrix.convert_types(Decimal)

        start_time = time.time()
        self.tf_result = matrix.gauss()
        elapsed_time = time.time() - start_time
        tf_time_method1 = elapsed_time

        # typ podwojnej prezycji
        getcontext().prec = 16
        matrix.convert_types(Decimal)

        start_time = time.time()
        self.td_result = matrix.gauss()
        elapsed_time = time.time() - start_time
        td_time_method1 = elapsed_time

        # METODA II gauss_part
        # typ ulamkowy

        # matrix.convert_types(Fraction)
        #
        # start_time = time.time()
        # self.tc_result = matrix.gauss_part()
        # elapsed_time = time.time() - start_time
        # tc_time_method2 = elapsed_time

        # typ pojedynczej precyzji
        getcontext().prec = 7
        matrix.convert_types(Decimal)

        start_time = time.time()
        self.tf_result = matrix.gauss_part()
        elapsed_time = time.time() - start_time
        tf_time_method2 = elapsed_time

        # typ podwojnej precyzji
        getcontext().prec = 16
        matrix.convert_types(Decimal)

        start_time = time.time()
        self.td_result = matrix.gauss_part()
        elapsed_time = time.time() - start_time
        td_time_method2 = elapsed_time

        # metoda III gauss_full
        # typ ulamkowy
        # matrix.convert_types(Fraction)
        #
        # start_time = time.time()
        # self.tc_result = matrix.gauss_full()
        # elapsed_time = time.time() - start_time
        # tc_time_method3 = elapsed_time

        # typ pojedynczej precyzji
        getcontext().prec = 7
        matrix.convert_types(Decimal)

        start_time = time.time()
        self.tf_result = matrix.gauss_full()
        elapsed_time = time.time() - start_time
        tf_time_method3 = elapsed_time

        # typ podwojnej precyzji
        getcontext().prec = 16
        matrix.convert_types(Decimal)

        start_time = time.time()
        self.td_result = matrix.gauss_full()
        elapsed_time = time.time() - start_time
        td_time_method3 = elapsed_time

        # dla policzenia srednich czasow

        tf_time_sum_method1 += tf_time_method1
        tf_time_sum_method2 += tf_time_method2
        tf_time_sum_method3 += tf_time_method3

        # tc_time_sum_method1 += tc_time_method1
        # tc_time_sum_method2 += tc_time_method2
        # tc_time_sum_method3 += tc_time_method3

        td_time_sum_method1 += td_time_method1
        td_time_sum_method2 += td_time_method2
        td_time_sum_method3 += td_time_method3


        plt.scatter([1, 2, 3], [tf_time_sum_method1, tf_time_sum_method2, tf_time_sum_method3],
                    label="dla dokladnosci pojedynczej precyzji", marker=".")
        plt.scatter([1, 2, 3], [td_time_sum_method1, td_time_sum_method2, td_time_sum_method3],
                    label="dla dokladnosci podwojnej precyzji", marker=".")
        # plt.scatter([tc_time_sum_method1, tc_time_sum_method2, tc_time_sum_method3], [1, 2, 3],
        #             label="dla dokladnosci własnego typu",
        #             marker=".")
        plt.xlabel('Numery metod 1-G, 2-PG, 3-FG')
        plt.ylabel('Czas potrzbeny na wyliczenie metod')
        plt.title('Wykres pokazujacy czas obliczania\n w zaleznosci od metody i rodzaju dokladnosci danych')
        plt.legend()
        plt.show()

# MAIN

sizes_array = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
results_Q1_method1 = []
results_Q1_method2 = []

results_Q2_tf = []
results_Q2_td = []
results_Q2_tc = []

test = Tests()
test.test_1()
test.test_2()
test.test_3() # trzeba odpalic po test_2, bo tam sa wyliczane czesciowe wyniki uzyte w test_3
test.test_4()

