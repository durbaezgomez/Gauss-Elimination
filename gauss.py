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