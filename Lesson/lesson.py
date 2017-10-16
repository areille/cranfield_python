from math import log as ln
from math import *


def present_amount(A0, p, n):
    return A0 * (1 + p / (360.0 * 100))**n


def initial_amount(A, p, n):
    return A * (1 + p / (360.0 * 100))**(-n)


def days(A0, A, p):
    return ln(A / A0) / ln(1 + p / (360.0 * 100))


def annual_rate(A0, A, n):
    return 360 * 100 * ((A / A0)**(1.0 / n) - 1) # ok le commentaire


if __name__ == '__main__':
    print(present_amount(100, 2.4, 30))
    print(initial_amount(200, 2.4, 30))
    print(days(100, 200, 2.4))
    print(annual_rate(100, 200, 30))
