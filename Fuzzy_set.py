import matplotlib.pyplot as plt
import numpy as np
    
class Fuzzy_set:  
    def __init__(self, m: float, M: float, a: float, b: float, inverted: bool = False):
        assert all([n for n in (m, M, a, b) if isinstance(n, float) or isinstance(n, int)]), '(m, M, a, b) - float or int numbers'
        assert m <= M, 'The condition m > M - Invalid'
        assert m != M or not (a == b == 0), 'Configuration (m, m, 0, 0) - Invalid'
        assert a > 0 and b > 0, 'Uncertainty bounds a, b cannot be < 0'
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
            self.kn.append(round(1/self.a, 7))
            
        if self.M == self.bounds[1]:
            self.kn.append(0)
        else:
            self.kn.append(round(-1/self.b, 7))

        self.bn = [round(-self.kn[0] * (self.bounds[0]), 7),
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
    
    def _boundsIntersection(self, other):
        x1 = (other.bn[0] - self.bn[1]) / (self.kn[1] - other.kn[0])
        x2 = (other.bn[1] - self.bn[0]) / (self.kn[0] - other.kn[1])
        return (x1, x2)
    
    def _plot(self, other = None, type: str = None) -> None:
        if isinstance(other, Fuzzy_set):
            bounds = (min(self.bounds[0], other.bounds[0]),
                    max(self.bounds[1], other.bounds[1]))
        elif (len(other) == 1 and (isinstance(other, int) or isinstance(other, float) or
                isinstance(*other, int) or isinstance(*other, float))):
            bounds = self.bounds
        elif isinstance(other, tuple):
            bounds = (min(fs.bounds[0] for fs in list(other)+[self]),
                    max(fs.bounds[1] for fs in list(other)+[self]))
        
        X = np.arange(bounds[0], bounds[1]+0.1, 0.1)
        
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
        elif type == 'floordiv':
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
            Y = tuple(max(self.probability(i) - sum(tuple(fs.probability(i) for fs in other)), 0) for i in X)
        elif type == 'trunc':
            Y = np.amin(np.stack([self.probability(X), np.full_like(self.probability(X), other, dtype=float)]))
        elif type == 'extension':
            Y = np.amax(np.stack([self.probability(X), np.full_like(self.probability(X), other, dtype=float)]))
                
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
        assert isinstance(other, Fuzzy_set), 'Addition is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
        return Fuzzy_set(*self._add(other))

    def __iadd__(self, other):
        assert isinstance(other, Fuzzy_set), 'Addition is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
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
        assert isinstance(other, Fuzzy_set), 'Subtraction is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
        return Fuzzy_set(*self._sub(other))
    
    def __isub__(self, other):
        assert isinstance(other, Fuzzy_set), 'Subtraction is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
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
        assert isinstance(other, Fuzzy_set), 'Multiplication is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
        return Fuzzy_set(*self._mul(other))

    def __imul__(self, other):
        assert isinstance(other, Fuzzy_set), 'Multiplication is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
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
        assert isinstance(other, Fuzzy_set), 'Division is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
        return Fuzzy_set(*self._truediv(other))

    def __itruediv__(self, other):
        assert isinstance(other, Fuzzy_set), 'Division is performed only with fuzzy sets'
        assert self.inverted == other.inverted, 'Unable to perform operation'
        self.m, self.M, self.a, self.b = self._truediv(other)
        self._updateSet()
        return self
    
    def _pow(self, exp) -> tuple:
        m, M, a, b = self.params()
        for _ in range(exp-1):
            a = m**2 - (m - a)**2
            b = (M + b)**2 - M**2
            m *= exp
            M *= exp
        return (m, M, a, b)

    def __pow__(self, exp):
        assert isinstance(exp, int), 'A**X, X - integer'
        return Fuzzy_set(*self._pow(exp))

    def __ipow__(self, exp):
        assert isinstance(exp, int), 'A**X, X - integer'        
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

    def __hash__(self):
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
        return (self.bounds[0] + 
                2 * (self.m + self.M) + self.bounds[1])/6

    def var(self) -> float:
        ''' The variance of A '''
        return ((self.bounds[1] - self.bounds[0])**2
            + 2 * (self.bounds[1] - self.bounds[0]) *
                (self.M - self.m) + 3*(self.M - self.m)**2)/24

    def con(self, k: float = 2) -> None:
        ''' Algebraic concentration of fuzzy set '''
        assert isinstance(k, float) and k > 1, 'k < 1'
        self._plot(k, type='con')
    
    def dil(self, k: float = 0.5) -> None:
        ''' Algebraic dilatation of fuzzy set '''
        assert isinstance(k, float) and k < 1, 'k > 1'
        assert k > 0, 'k < 0'
        self._plot(k, type='dil')

    def contains(self, *other) -> None:
        assert tuple(i for i in other if isinstance(i, Fuzzy_set))
        self._plot(other, type='contains')

    def intersect(self, *other) -> None:
        ''' Intersection of A '''
        assert tuple(i for i in other if isinstance(i, Fuzzy_set)), 'Only Fuzzy sets allowed'
        self._plot(other, type='intersect')

    def union(self, *other) -> None:
        ''' Union of A '''
        assert tuple(i for i in other if isinstance(i, Fuzzy_set)), 'Only Fuzzy sets allowed'
        self._plot(other, type='union')
    
    def addition(self, other) -> None:
        ''' Algebraic addition of fuzzy sets '''
        self._plot(other, type='addition')
    
    def substract(self, *other) -> None:
        ''' Algebraic substraction of fuzzy sets '''
        assert tuple(i for i in other if isinstance(i, Fuzzy_set)), 'Only Fuzzy sets allowed'
        self._plot(other, type='substract')
    
    def mul(self, *other) -> None:
        ''' Algebraic multiplication of fuzzy sets '''
        print(other)
        assert isinstance(*other, float) or isinstance(*other, int) or tuple(i for i in other if isinstance(i, Fuzzy_set))
        if isinstance(*other, float) or isinstance(*other, int):
            self._plot(other, type='matmul_number')
        else:
            self._plot(other, type='matmul_set')
    
    def division(self, *other):
        assert tuple(i for i in other if isinstance(i, Fuzzy_set)), 'Division is performed only with fuzzy sets'
        self._plot(other, type='floordiv')
        
    def trunc(self, k: float) -> None:
        ''' Truncation of fuzzy set '''
        assert isinstance(k, float) and k < 1, 'k > 1'
        self._plot(k, type='trunc')
    
    def extension(self, k: float) -> None:
        assert isinstance(k, float) and k < 1, 'k > 1'
        self._plot(k, type='extension')

    def probability(self, x: float, accuracy: int = 4) -> float:
        assert isinstance(x, int) or isinstance(x, float) or isinstance(x, np.ndarray)
        assert isinstance(accuracy, int)
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
        assert isinstance(other, Fuzzy_set)
        if self.inverted == other.inverted:
            if (self.m <= other.m and self.M >= other.m or
                self.m >= other.m  and self.m <= other.M):
                if self.inverted:
                    return 0
                return 1
            elif (self.bounds[1] <= other.bounds[0] or
                self.bounds[0] >= other.bounds[1]):
                if self.inverted:
                    return 1
                return 0
        else:
            if (self.bounds[0] > other.m or
                self.bounds[1] < other.M):
                if self.inverted:
                    return 1
                return 0
            if (self.m <= other.bounds[0] or
                self.M >= other.bounds[1]):
                if self.inverted:
                    return 0
                return 1
        
        x1, x2 = self._boundsIntersection(other)
        y1 = round(self.kn[1] * x1 + self.bn[1], 4)
        y2 = round(self.kn[0] * x2 + self.bn[0], 4)
        if 0 <= y1 <= 1 and 0 <= y2 <= 1:
            return max(y1, y2)
        elif 0 <= y1 <= 1:
            return y1
        return y2

    def inf(self, other) -> float:
        ''' Infimum of two fuzzy sets '''
        assert isinstance(other, Fuzzy_set)
        if self.inverted == other.inverted:
            return 0
        else:
            if (self.m > other.bounds[0] or
                self.M < other.bounds[1]):
                return 0
            elif (self.bounds[0] >= other.m or
                self.bounds[1] <= other.M):
                return 1

        x1, x2 = self._boundsIntersection(other)
        y1 = round(self.kn[1] * x1 + self.bn[1], 4)
        y2 = round(self.kn[0] * x2 + self.bn[0], 4)
        if 0 <= y1 <= 1 and 0 <= y2 <= 1:
            return round(max(y1, y2), 4)
        elif 0 <= y1 <= 1:
            return y1
        return y2
    
    def core(self) -> tuple:
        return (self.m, self.M)
    
    def a_level(self, a: float) -> tuple:
        assert a < 1, 'a >= 1'
        #TODO a
        return ((self.bounds[0]+self.m)/2,
                (self.M+self.bounds[1])/2)
    
    def support(self) -> tuple:
        return (self.bounds[0], self.bounds[1])

    def params(self) -> tuple: 
        return (self.m, self.M, self.a, self.b)

    def coefs(self) -> tuple:
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
        self.field = []
        for fuzzy_set in fuzzy_sets:
            self.field.append(fuzzy_set)
        self.field = list(set(tuple(self.field)))
        
    def __add__(self, other):
        return Fuzzy_field(*list(set(tuple(self.field+other.field))))

    def __iadd__(self, other):
        self.field += other.field
        self.field = list(set(tuple(self.field)))
        return self

    def clear(self) -> None:
        self.field = []

    def add(self, *fuzzy_sets) -> None:
        for fuzzy_set in fuzzy_sets:
            self.field.append(fuzzy_set)
        self.field = list(set(tuple(self.field)))

    def field_info(self) -> None:
        if not len(self.field):
            print('Field is empty')
        else:
            print(f'{len(self.field)} sets:')
            i = 1
            for fuzzy_set in self.field:
                print(f'{i}: {fuzzy_set.params()}')
                i += 1

    def draw_field(self, bounds=False, title: str = 'Fuzzy set\'s field mapping') -> None:
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