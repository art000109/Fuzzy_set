import matplotlib.pyplot as plt
import numpy as np
    
class Fuzzy_set:  
    def __init__(self, m: float, M: float, a: float, b: float, inverted: bool = False):
        if not(all([1 if isinstance(n, (float, int)) else 0 for n in (m, M, a, b)])):
            raise ValueError('(m, M, a, b) - float or int numbers')
        if m > m:
            raise ValueError('The condition m > M - Invalid')
        if m == M and (a == b == 0):
            raise ValueError('Configuration (m, m, 0, 0) - Invalid')
        if a < 0 or b < 0:
            raise ValueError('Uncertainty bounds a, b cannot be < 0')
        self.m, self.M = m, M
        self.a, self.b = a, b
        self.inverted = inverted
        self.bounds = (self.m - self.a, self.M + self.b)
        self._calculateCurves()

    def _calculateCurves(self) -> None:
        self.kn, self.bn = [], []
        if self.m == self.bounds[0]:
            self.kn.append(0)
        else:
            self.kn.append(round(-1/(self.bounds[0] - self.m), 7))

        if self.M == self.bounds[1]:
            self.kn.append(0)
        else:
            self.kn.append(round(1/(self.M - self.bounds[1]), 7))

        self.bn = [round(-self.kn[0] * self.bounds[0], 7),
                   round(1 - self.kn[1] * self.M, 7)]

        if self.inverted:
            for i in range(len(self.kn)):
                self.kn[i] = -self.kn[i]
                self.bn[i] = 1 - self.bn[i]
        self.kn = tuple(self.kn)
        self.bn = tuple(self.bn)
    
    def _updateSet(self) -> None:
        self.bounds = (self.m - self.a, self.M + self.b)
        self._calculateCurves()

    def _linearProbability(self, x: float, n: int) -> float:
        return self.kn[n] * x + self.bn[n]
    
    def _boundsIntersection(self, other) -> tuple:
        X = []
        for i in range(2):
            for j in range(2):
                if self.kn[i] != other.kn[j]:
                    X.append((other.bn[j] - self.bn[i]) / (self.kn[i] - other.kn[j]))
        return tuple(x for x in X if 0 < self.probability(x) <= 1)
    
    def _plot(self, other = None, type: str = None, accuracy: int = 2) -> None:
        if isinstance(other, Fuzzy_set):
            bounds = (min(self.bounds[0], other.bounds[0]),
                    max(self.bounds[1], other.bounds[1]))
        elif isinstance(other, (int, float)):
            bounds = self.bounds
        elif isinstance(other, tuple):
            bounds = (min(fs.bounds[0] for fs in list(other)+[self]),
                    max(fs.bounds[1] for fs in list(other)+[self]))
        
        X = np.arange(bounds[0], bounds[1] + 0.1**accuracy, 0.1**accuracy)
        
        if type == 'con' or type == 'dil':
            Y = self.probability(X) ** other
        elif type == 'eq':
            Y = 1 - abs(self.probability(X) - other.probability(X))
        elif type == 'ne':
            Y = abs(self.probability(X) - other.probability(X))
        elif type == 'contains':
            other = tuple(list(other) + [self])
            min_array = (1 - np.array([fs.probability(X) for fs in other]).sum(axis=0))
            Y = np.amin(np.stack([min_array, np.ones_like(min_array, dtype=float)]), axis=0)
        elif type == 'matmul_number':
            min_array = self.probability(X) * other
            Y = np.amin(np.stack([min_array, np.ones_like(min_array, dtype=float)]), axis=0)
        elif type == 'matmul_set':
            other = tuple(list(other) + [self])
            Y = np.prod([fs.probability(X) for fs in other], axis=0)
        elif type == 'addition':
            Y = self.probability(X) + other.probability(X) - self.probability(X) * other.probability(X)
        elif type == 'division':
            temp = np.array([fs.probability(X) for fs in other])
            if temp.shape[0] == 1:
                temp = temp.reshape(temp.shape[1],)
            temp0 = np.prod(temp, axis=0)
            temp1 = np.where(temp0 == 0, self.probability(X), temp0)
            min_array = self.probability(X) / np.where(temp1 == 0, 1, temp1)
            Y = np.amin(np.stack([min_array, np.ones_like(min_array)]), axis=0)
        elif type == 'intersect' or type == 'union':
            other = tuple(list(other) + [self])
            if type == 'intersect':
                Y = np.amin(np.array([fs.probability(X) for fs in other]), axis=0)
            else:
                Y = np.amax(np.array([fs.probability(X) for fs in other]), axis=0)
        elif type == 'substract':
            min_array = self.probability(X) - np.sum(np.array([fs.probability(X) for fs in other]), axis=0)
            Y = np.amax(np.stack([min_array, np.zeros_like(min_array, dtype=float)]), axis=0)
        elif type == 'trunc':
            Y = np.amin(np.stack([self.probability(X), np.full_like(self.probability(X), other, dtype=float)]), axis=0)
        elif type == 'extension':
            Y = np.amax(np.stack([self.probability(X), np.full_like(self.probability(X), other, dtype=float)]), axis=0)
        elif type == 'inclusion':
            min_array = 1 - self.probability(X) + other.probability(X)
            Y = np.amin(np.stack([min_array, np.ones_like(min_array)]), axis=0)
                
        _, ax = plt.subplots()
        ax.set_xlim(bounds[0]-1, bounds[1]+1)
        ax.set_ylim(0, 1.1)
        ax.plot(X, Y)
        ax.grid()
        plt.show()
    
    def _add(self, other) -> tuple:
        a = self.a + other.a
        b = self.b + other.b
        m = self.m + other.m
        M = self.M + other.M
        return (m, M, a, b)
    
    def __add__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Addition is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        return Fuzzy_set(*self._add(other))

    def __iadd__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Addition is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        self.m, self.M, self.a, self.b = self._add(other)
        self._updateSet()
        return self

    def _sub(self, other) -> tuple:
        a = self.a + other.b
        b = self.b + other.a
        m = self.m - other.M
        M = self.M - other.m
        return (m, M, a, b)
		
    def __sub__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Subtraction is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        return Fuzzy_set(*self._sub(other))
    
    def __isub__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Subtraction is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        self.m, self.M, self.a, self.b = self._sub(other)
        self._updateSet()
        return self

    def _mul(self, other) -> tuple:
        a = self.m * other.m - (self.bounds[0]) * (other.bounds[0])
        b = (self.bounds[1]) * (other.bounds[1]) - self.M * other.M
        m = self.m * other.m
        M = self.M * other.M
        return (m, M, a, b)
		
    def __mul__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Multiplication is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        return Fuzzy_set(*self._mul(other))

    def __imul__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Multiplication is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        self.m, self.M, self.a, self.b = self._mul(other)
        self._updateSet()
        return self

    def _truediv(self, other) -> tuple:
        a = (self.m * other.b + other.M * self.a) / (other.M**2 + other.M * other.b)
        b = (other.m * self.b + self.M * other.a) / (other.m**2 + other.m * other.a)
        m = self.m / other.M
        M = self.M / other.m
        return (m, M, a, b)

    def __truediv__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Division is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        return Fuzzy_set(*self._truediv(other))

    def __itruediv__(self, other):
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Division is performed only with fuzzy sets')
        if self.inverted != other.inverted:
            raise ArithmeticError('Unable to perform operation')
        self.m, self.M, self.a, self.b = self._truediv(other)
        self._updateSet()
        return self
    
    def _pow(self, exp) -> tuple:
        m, M, a, b, _ = self.params()
        for _ in range(exp-1):
            a = m**2 - (m - a)**2
            b = (M + b)**2 - M**2
            m *= exp
            M *= exp
        return (m, M, a, b)

    def __pow__(self, exp):
        if not isinstance(exp, int):
            raise ValueError('A**X, X - integer')
        return Fuzzy_set(*self._pow(exp))

    def __ipow__(self, exp):
        if not isinstance(exp, int):
            raise ValueError('A**X, X - integer')   
        self.a, self.m, self.M, self.b = self._pow(exp)
        self._updateSet()
        return self

    def __len__(self) -> float:
        return self.bounds[1] - self.bounds[0] + 1

    def __pos__(self):
        return self

    def __eq__(self, other) -> None:
        self._plot(other, type='eq')

    def __ne__(self, other) -> None:
        self._plot(other, type='ne')

    def __contains__(self, other) -> bool:
        return self.bounds[0] < other < self.bounds[1]

    def __hash__(self) -> int:
        return hash((self.m, self.M, self.a, self.b, self.inverted))

    def __repr__(self) -> str:
        return f'Fuzzy_set(m: {self.m}, M: {self.M}, a: {self.a}, b: {self.b})'

    def __round__(self, n=0):
        a = round(self.a, n)
        b = round(self.b, n)
        m = round(self.m, n)
        M = round(self.M, n)
        return Fuzzy_set(m, M, a, b)

    def __invert__(self):
        ''' Supplement A (Inversion A) '''
        return Fuzzy_set(self.m, self.M, self.a, self.b, inverted=True)

    def mean(self) -> float:
        ''' Mean of A '''
        return (self.bounds[0] + 2 * self.m +
                2 * self.M + self.bounds[1])/6

    def var(self) -> float:
        ''' Variance of A '''
        return ((self.bounds[1] - self.bounds[0])**2
            + 2 * (self.bounds[1] - self.bounds[0]) * (self.M - self.m) +
            3*(self.M - self.m)**2)/24
    
    def trunc(self, k: float, accuracy: float = 2) -> None:
        ''' Truncation of fuzzy set '''
        if not (isinstance(k, float) and 0 < k < 1):
            raise ValueError('k must be between (0, 1)')
        self._plot(k, type='trunc', accuracy=accuracy)
    
    def extension(self, k: float, accuracy: float = 2) -> None:
        ''' Extension of fuzzy set '''
        if not (isinstance(k, float) and 0 < k < 1):
            raise ValueError('k must be between (0, 1)')
        self._plot(k, type='extension', accuracy=accuracy)

    def con(self, k: float = 1.5, accuracy: float = 2) -> None:
        ''' Algebraic concentration of fuzzy set '''
        if not isinstance(k, (float, int)):
            raise TypeError('k - float or int number')
        if k <= 1:
            raise ValueError('k must be grater then 1')
        self._plot(k, type='con', accuracy=accuracy)
    
    def dil(self, k: float = 0.5, accuracy: float = 2) -> None:
        ''' Algebraic dilatation of fuzzy set '''
        if not isinstance(k, (float, int)):
            raise TypeError('k - float or int number')
        if k <= 0 or k >= 1:
            raise ValueError('k must be between (0, 1)')
        self._plot(k, type='dil', accuracy=accuracy)

    def contains(self, *other, accuracy: float = 2) -> None:
        if not all(tuple(1 if isinstance(i, Fuzzy_set) else 0 for i in other)):
            raise TypeError('Only Fuzzy sets allowed')
        self._plot(other, type='contains', accuracy=accuracy)

    def intersect(self, *other, accuracy: float = 2) -> None:
        ''' Intersection of A '''
        if not all(tuple(1 if isinstance(i, Fuzzy_set) else 0 for i in other)):
            raise TypeError('Intersection is performed only with fuzzy sets')
        self._plot(other, type='intersect', accuracy=accuracy)

    def union(self, *other, accuracy: float = 2) -> None:
        ''' Union of A '''
        if not all(tuple(1 if isinstance(i, Fuzzy_set) else 0 for i in other)):
            raise TypeError('Union is performed only with fuzzy sets')
        self._plot(other, type='union', accuracy=accuracy)
    
    def addition(self, other, accuracy: float = 2) -> None:
        ''' Algebraic addition of fuzzy sets '''
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Addition is performed only with fuzzy set')
        self._plot(other, type='addition', accuracy=accuracy)
    
    def substract(self, *other, accuracy: float = 2) -> None:
        ''' Algebraic substraction of fuzzy sets '''
        if not (len(tuple(i for i in other if isinstance(i, Fuzzy_set))) == len(other)):
            raise TypeError('Substracton is performed only with fuzzy sets')
        self._plot(other, type='substract', accuracy=accuracy)
    
    def mul(self, *other, accuracy: float = 2) -> None:
        ''' Algebraic multiplication of fuzzy sets '''
        if not ( isinstance(*other, (float, int)) or tuple(i for i in other if isinstance(i, Fuzzy_set))):
            raise TypeError('Only Fuzzy sets or numbers allowed')
        if isinstance(*other, (float, int)):
            if other[0] <= 0:
                raise ValueError('k must be grater then 0')
            self._plot(*other, type='matmul_number', accuracy=accuracy)
        else:
            self._plot(other, type='matmul_set', accuracy=accuracy)
    
    def division(self, *other, accuracy: float = 2) -> None:
        ''' Algebraic division of fuzzy sets '''
        if not ( len( tuple(i for i in other if isinstance(i, Fuzzy_set)) ) == len(other)):
            raise TypeError('Division is performed only with fuzzy sets')
        self._plot(other, type='division', accuracy=accuracy)

    def inclusion(self, other, accuracy: float = 2) -> None:
        ''' Inclusion of fuzzy set '''
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Addition is performed only with fuzzy set')
        self._plot(other, type='inclusion', accuracy=accuracy)

    def probability(self, x: float, accuracy: int = 4) -> float:
        if not isinstance(x, (int, float, np.ndarray)):
            raise TypeError('x must be integer or float number')
        if not isinstance(accuracy, int):
            raise TypeError('Accuracy must be integer number')
        if isinstance(x, np.ndarray):
            if self.inverted:
                x1 = np.ones_like(x[np.where(np.less_equal(x, self.bounds[0]))])
                x5 = np.ones_like(x[np.where(np.greater_equal(x, self.bounds[1]))])
            else:
                x1 = np.zeros_like(x[np.where(np.less_equal(x, self.bounds[0]))])
                x5 = np.zeros_like(x[np.where(np.greater_equal(x, self.bounds[1]))])
            x2 = np.around(self._linearProbability(x[np.where(np.logical_and(x > self.bounds[0], x < self.m))[0]], 0), accuracy)
            x3 = np.ones_like(np.where(np.logical_and(x >= self.m, x <= self.M))[0])
            x4 = np.around(self._linearProbability(x[np.where(np.logical_and(x > self.M, x < self.bounds[1]))[0]], 1), accuracy)
            return np.concatenate([x1, x2, x3, x4, x5], axis=0)         
        else:
            if x <= self.bounds[0] or x >= self.bounds[1]:
                if self.inverted:
                    return 1
                return 0
            elif self.bounds[0] < x < self.m:
                return round(self._linearProbability(x, 0), accuracy)
            elif self.m <= x <= self.M:
                return 1
            elif self.M < x < self.bounds[1]:
                return round(self._linearProbability(x, 1), accuracy)

    def supr(self, other) -> float:
        ''' Supremum of two fuzzy sets '''
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Only Fuzzy sets allowed')
        if self.inverted == other.inverted:
            if self.inverted:
                return 1
            if (self.m <= other.m <= self.M or
                self.m <= other.M <= self.M or
                other.m <= self.m <= other.M or
                other.m <= self.M <= other.M):
                    return 1
            if (self.bounds[1] <= other.bounds[0] or
                self.bounds[0] >= other.bounds[1]):
                if self.inverted:
                    return 1
                return 0
        else:
            return 1
        
        X = self._boundsIntersection(other)
        return max(self.probability(x) for x in X)

    def inf(self, other) -> float:
        ''' Infimum of two fuzzy sets '''
        if not isinstance(other, Fuzzy_set):
            raise TypeError('Only Fuzzy sets allowed')
        if self.inverted == other.inverted:
            if (not self.inverted or
                self.m <= other.m <= self.M or
                self.m <= other.M <= self.M or
                other.m <= self.m <= other.M or
                other.m <= self.M <= other.M):
                return 0
        else:
            if (self.bounds[0] >= other.bounds[1] or
                self.bounds[1] <= other.bounds[0]):
                return 1
    
        X = self._boundsIntersection(other)
        return min(self.probability(x) for x in X)
    
    def core(self, step: float = None) -> tuple:
        ''' Get 2 points (m, M) '''
        if not step:
            return (self.m, self.M)
        else:
            if step == 0:
                raise ValueError('step must not be 0')
            if step > 0:
                return np.arange(self.m, self.M + step, step)
            else:
                return np.arange(self.M, self.m + step, step)
    
    def support(self, step: float = None) -> tuple:
        ''' Get 2 points (m-a, M+b) '''
        if not step:
            return (self.bounds[0], self.bounds[1])
        else:
            if step == 0:
                raise ValueError('step must not be 0')
            if step > 0:
                return np.arange(self.bounds[0], self.bounds[1] + step, step)
            else:
                return np.arange(self.bounds[1], self.bounds[0] + step, step)

    def a_level(self, a: float, step: float = None) -> tuple:
        ''' Get such 2 points for which the probability of occurrence is equal to a '''
        if not(0 < a < 1):
            raise ValueError('a must be between (0, 1)')
        if not step:
            return (round((self.bounds[0]+self.m)*a, 3),
                round((self.M+self.bounds[1])*a, 3))
        else:
            if step == 0:
                raise ValueError('step must not be 0')
            if step > 0:
                return np.arange(round((self.bounds[0]+self.m)*a, 3), round((self.M+self.bounds[1])*a, 3) + step, step)
            else:
                return np.arange(round((self.M+self.bounds[1])*a, 3), round((self.bounds[0]+self.m)*a, 3) + step, step)
        
    def transition_points(self, step: float = None) -> tuple:
        ''' Get such 2 points for which the probability of occurrence is equal to 0.5 '''
        return self.a_level(0.5, step)

    def params(self) -> tuple:
        ''' Get main parameters of the set '''
        return (self.m, self.M, self.a, self.b, self.inverted)

    def coefs(self) -> tuple:
        ''' coefficients of straight lines of set boundaries (k0, k1), (b0, b1) '''
        return ((self.kn[0], self.kn[1]), (self.bn[0], self.bn[1]))

    def draw_set(self, bounds: bool = True, title: str = 'Fuzzy set mapping') -> None:
        _, ax = plt.subplots()
        X = (self.bounds[0], self.m, self.M, self.bounds[1])
        Y = (0, 1, 1, 0)
        if self.inverted:
            Y = (1, 0, 0, 1)
        ax.plot(X, Y)
        
        if bounds and not (self.m == self.M == self.b == 0 or
                self.m == self.M == self.a == 0):
            if self.bounds[0] != self.m:
                ax.plot((self.m, self.m), (0, 1), dashes=[6, 2])
            if self.bounds[1] != self.M:
                ax.plot((self.M, self.M), (0, 1), dashes=[6, 2])
        ax.set(xlabel='X', ylabel='Probability',
                title=title)
        ax.grid()
        plt.show()

class Fuzzy_field:
    def __init__(self, *fuzzy_sets):
        if not all(tuple(1 if isinstance(i, Fuzzy_set) else 0 for i in fuzzy_sets)):
            raise TypeError('Only Fuzzy sets allowed')
        self.field = []
        for fuzzy_set in fuzzy_sets:
            self.field.append(fuzzy_set)
        self.field = list(set(tuple(self.field)))
        
    def __add__(self, other):
        if not isinstance(other, Fuzzy_field):
            raise TypeError('Only Fuzzy field allowed')
        return Fuzzy_field(*list(set(tuple(self.field+other.field))))

    def __iadd__(self, other):
        if not isinstance(other, Fuzzy_field):
            raise TypeError('Only Fuzzy field allowed')
        self.field += other.field
        self.field = list(set(tuple(self.field)))
        return self
    
    def __repr__(self) -> str:
        if len(self.field) != 1:
            return f'Fuzzy_field({len(self.field)} sets)'
        return f'Fuzzy_field({len(self.field)} set)'

    def clear(self) -> None:
        ''' Clear the fuzzy field '''
        self.field = []

    def add(self, *fuzzy_sets) -> None:
        ''' Add other fuzzy sets '''
        if not all(tuple(1 if isinstance(i, Fuzzy_set) else 0 for i in fuzzy_sets)):
            raise TypeError('Only Fuzzy sets allowed')
        for fuzzy_set in fuzzy_sets:
            self.field.append(fuzzy_set)
        self.field = list(set(tuple(self.field)))

    def field_info(self) -> None:
        ''' Print information about fuzzy sets '''
        if not len(self.field):
            print('Field is empty')
        else:
            print(f'{len(self.field)} sets:')
            i = 1
            for fuzzy_set in self.field:
                print(f'{i}: {fuzzy_set.params()}')
                i += 1

    def draw_field(self, bounds: bool = False, title: str = 'Fuzzy set\'s field mapping') -> None:
        _, ax = plt.subplots()
        for fuzzy_set in self.field:
            if not fuzzy_set.inverted:
                ax.plot((fuzzy_set.bounds[0],
                         fuzzy_set.m,
                         fuzzy_set.M,
                         fuzzy_set.bounds[1]),
                        (0, 1, 1, 0))
            else:
                ax.plot((fuzzy_set.bounds[0],
                         fuzzy_set.m,
                         fuzzy_set.M,
                         fuzzy_set.bounds[1]),
                        (1, 0, 0, 1))
            if bounds:
                if not (fuzzy_set.m == fuzzy_set.M == fuzzy_set.b == 0 or
                        fuzzy_set.m == fuzzy_set.M == fuzzy_set.a == 0):
                    if fuzzy_set.bounds[0] != fuzzy_set.m:
                        ax.plot((fuzzy_set.m, fuzzy_set.m),
                                (0, 1), dashes=[6, 2])
                    if fuzzy_set.bounds[1] != fuzzy_set.M:
                        ax.plot((fuzzy_set.M, fuzzy_set.M), (0, 1),
                                dashes=[6, 2])
                
        ax.set(xlabel='X', ylabel='Probability',
                title=title)
        ax.grid()
        plt.show()