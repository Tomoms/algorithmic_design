#!/usr/bin/env python3

from __future__ import annotations

from numbers import Number
from typing import List, Tuple
from random import random, seed
import math


def gauss_matrix_mult(A: Matrix, B: Matrix) -> Matrix:
    ''' Multiply two matrices by using Gauss's algorithm

    Parameters
    ----------
    A: Matrix
        The first matrix to be multiplied
    B: Matrix
        The second matrix to be multiplied

    Returns
    -------
    Matrix
        The row-column multiplication of the matrices passed as parameters

    Raises
    ------
    ValueError
        If the number of columns of `A` is different from the number of
        rows of `B`
    '''
    if(A.num_of_cols != B.num_of_rows):
        raise ValueError("The two matrix can not be multiplied")

    result = [[0 for col in range(B.num_of_cols)]
              for row in range(A.num_of_rows)]

    for row in range(A.num_of_rows):
        for col in range(B.num_of_cols):
            value = 0
            for k in range(A.num_of_cols):
                value += A[row][k] * B[k][col]
            result[row][col] = value

    return Matrix(result, clone_matrix=False)


def get_matrix_quadrands(A: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    A11 = A.submatrix(0, A.num_of_rows//2, 0, A.num_of_cols//2)
    A12 = A.submatrix(0, A.num_of_rows//2, A.num_of_cols//2, A.num_of_cols//2)
    A21 = A.submatrix(A.num_of_rows//2, A.num_of_rows//2, 0, A.num_of_cols//2)
    A22 = A.submatrix(A.num_of_rows//2, A.num_of_rows//2, A.num_of_cols//2, A.num_of_cols//2)
    return A11, A12, A21, A22


def get_new_dim(n: int) -> int:
    exp = math.ceil(math.log(n, 2))
    return int(pow(2, exp))


def zero_pad(A: Matrix, new_dim: int) -> Matrix:
    new_rows = new_dim - A.num_of_rows
    new_cols = new_dim - A.num_of_cols
    values = A[:]
    for i in range(new_cols):
        for row in values:
            row.append(0)
    for j in range(new_rows):
        values.append([0 for i in range(new_dim)])
    return Matrix(values, clone_matrix = False)


def strassen_matrix_general(A: Matrix, B: Matrix) -> Matrix:
    m, n, p = A.num_of_rows, A.num_of_cols, B.num_of_cols
    q = get_new_dim(max(m, n, p))
    values = strassen_matrix_mult(zero_pad(A, q), zero_pad(B, q))[:]
    for i in range(q - p):
        for row in values:
            del row[q - i - 1]
    for j in range(q - m):
        del values[q - j - 1]
    return Matrix(values, clone_matrix = False)


def strassen_matrix_mult(A: Matrix, B: Matrix) -> Matrix:

    if A.is_null():
        return A
    if B.is_null():
        return B

    # Base case
    if max(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 32:  # by hypothesis B.num_of_rows = A.num_of_cols
        return gauss_matrix_mult(A, B)
    
    # recursive step
    A11, A12, A21, A22 = get_matrix_quadrands(A)
    B11, B12, B21, B22 = get_matrix_quadrands(B)

    # recursion calls
    P1 = strassen_matrix_mult(A11, B12 - B22)
    P2 = strassen_matrix_mult(A11 + A12, B22)
    P3 = strassen_matrix_mult(A21 + A22, B11)
    P4 = strassen_matrix_mult(A22, B21 - B11)
    P5 = strassen_matrix_mult(A11 + A22, B11 + B22)
    P6 = strassen_matrix_mult(A12 - A22, B21 + B22)
    P7 = strassen_matrix_mult(A11 - A21, B11 + B12)

    # second batch of sums Theta(n^2)
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7

    # build the resulting matrix
    result = Matrix([[0 for x in range(B.num_of_cols)] for y in range(A.num_of_rows)],
                    clone_matrix=False)

    # copying Cij into the resulting matrix
    result.assign_submatrix(0, 0, C11)
    result.assign_submatrix(0, result.num_of_cols//2, C12)
    result.assign_submatrix(result.num_of_rows//2, 0, C21)
    result.assign_submatrix(result.num_of_rows//2, result.num_of_cols//2, C22)

    return result


class Matrix(object):
    ''' A simple naive matrix class

    Members
    -------
    _A: List[List[Number]]
        The list of rows that store all the matrix values

    Parameters
    ----------
    A: List[List[Number]]
        The list of rows that store all the matrix values
    clone_matrix: Optional[bool]
        A flag to require a full copy of `A`'s data structure.

    Raises
    ------
    ValueError
        If there are two lists having a different number of values
    '''
    def __init__(self, A: List[List[Number]], clone_matrix: bool = True):
        num_of_cols = None

        for i, row in enumerate(A):
            if num_of_cols is not None:
                if num_of_cols != len(row):
                    raise ValueError('This is not a matrix')
            else:
                num_of_cols = len(row)

        if clone_matrix:
            self._A = [[value for value in row] for row in A]
        else:
            self._A = A

    @property
    def num_of_rows(self) -> int:
        return len(self._A)

    @property
    def num_of_cols(self) -> int:
        if len(self._A) == 0:
            return 0

        return len(self._A[0])

    def copy(self):
        A = [[value for value in row] for row in self._A]

        return Matrix(A, clone_matrix=False)

    def __getitem__(self, y: int):
        ''' Return one of the rows

        Parameters
        ----------
        y: int
            the index of the rows to be returned

        Returns
        -------
        List[Number]
            The `y`-th row of the matrix
        '''
        return self._A[y]

    def __iadd__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] += A[y][x]

        return self

    def __add__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res += A

        return res

    def __isub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] -= A[y][x]

        return self

    def __sub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res -= A

        return res

    def __mul__(self, A: Matrix) -> Matrix:
        ''' Multiply one matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix which multiplies this matrix

        Returns
        -------
        Matrix
            The row-column multiplication between this matrix and that passed
            as parameter

        Raises
        ------
        ValueError
            If the number of columns of this matrix is different from the
            number of rows of `A`
        '''
        return gauss_matrix_mult(self, A)

    def __rmul__(self, value: Number) -> Matrix:
        ''' Multiply one matrix by a numeric value

        Parameters
        ----------
        value: Number
            The numeric value which multiplies this matrix

        Returns
        -------
        Matrix
            The multiplication between `value` and this matrix

        Raises
        ------
        ValueError
            If `value` is not a number
        '''

        if not isinstance(value, Number):
            raise ValueError('{} is not a number'.format(value))

        return Matrix([[value*elem for elem in row] for row in self._A],
                      clone_matrix=False)

    def submatrix(self, from_row: int, num_of_rows: int,
                  from_col: int, num_of_cols: int) -> Matrix:
        ''' Return a submatrix of this matrix

        Parameters
        ----------
        from_row: int
            The first row to be included in the submatrix to be returned
        num_of_rows: int
            The number of rows to be included in the submatrix to be returned
        from_col: int
            The first col to be included in the submatrix to be returned
        num_of_cols: int
            The number of cols to be included in the submatrix to be returned

        Returns
        -------
        Matrix
            A submatrix of this matrix
        '''
        A = [row[from_col:from_col+num_of_cols]
             for row in self._A[from_row:from_row+num_of_rows]]

        return Matrix(A, clone_matrix=False)

    def assign_submatrix(self, from_row: int, from_col: int, A: Matrix):
        for y, row in enumerate(A):
            self_row = self[y + from_row]
            for x, value in enumerate(row):
                self_row[x + from_col] = value

    def __repr__(self):
        return '\n'.join('{}'.format(row) for row in self._A)

    def is_null(self):
        for row in self._A:
            for n in row:
                if n != 0:
                    return False
        return True


class IdentityMatrix(Matrix):
    ''' A class for identity matrices

    Parameters
    ----------
    size: int
        The size of the identity matrix
    '''
    def __init__(self, size: int):
        A = [[1 if x == y else 0 for x in range(size)]
             for y in range(size)]

        super().__init__(A)


if __name__ == '__main__':
    from random import random, seed
    from sys import stdout
    from timeit import timeit
    
    seed(0)  # to have reproducibility
    for i in range(0, 13):
        size = 2**i
        stdout.write(f'{size}')
        A = Matrix([[random() for x in range(size)] for y in range(size)])
        B = Matrix([[random() for x in range(size)] for y in range(size)])
        for funct in ['strassen_matrix_general']:
            T = timeit(f'{funct}(A,B)', globals=locals(), number=1)
            stdout.write('\t{:.3f}'.format(T))
            stdout.flush()
        stdout.write('\n')
