import sys
sys.path.append('..')


from Fuzzy_set import Fuzzy_set, Fuzzy_field

''' Initialization of trapezoidal fuzzy sets '''
set1 = (5,6,1,1)
set2 = (5,6,1,1)
A = Fuzzy_set(5,6,1,1)
B = Fuzzy_set(7,8,2,2)

# Equivalent to

A1 = Fuzzy_set(*set1)
B1 = Fuzzy_set(*set2)

''' Addition of trapezoidal fuzzy sets '''
C = A + B
A1 += B1

''' Subtraction trapezoidal fuzzy sets '''
D = A - B
A1 -= B1

''' Trapezoidal multiplication of fuzzy sets '''
E = A * B
A1 *= B1

''' Division of trapezoidal fuzzy sets '''
F = A / B
A1 /= B1

''' Exponentiation '''
G = A**3
A1 **= 3

''' Mean and variance of fuzzy set'''
A.mean()
A.var()

''' Supremum / Infimum '''
A.supr(B)
A.inf(B)

''' The probability of occurrence of a number in a set '''
A.probability(4.5)
A.probability(5)
A.probability(6.5)

''' Fuzzy set visualization '''
A.draw_set()

''' Fuzzy fields initiation '''
Q = Fuzzy_field(A, B)

# Equivalent to

Q = Fuzzy_field()
Q.add(A, B)

''' Fuzzy field visualization '''

Q.draw_field(bounds=True)
