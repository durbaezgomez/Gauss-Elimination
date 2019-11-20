from gauss import *
import matplotlib.pyplot as plt
import time

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
