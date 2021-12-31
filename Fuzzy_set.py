try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Модуль matplotlib недоступен")
    print("Метод draw_set недоступен")
    print("Класс Fuzzy_filed недоступен")
    
class Fuzzy_set:  
    def __init__(self, m: float, M: float, a: float, b: float, inverted: bool = False):
        assert m != M or not (a == b == 0), 'Конфигурация (m, m, 0, 0) - недопустима'
        assert m <= M, 'Должно выполняться условие m <= M'
        assert not (a < 0 or b < 0), 'Границы неопределённости a, b не могут быть < 0'
        self.m, self.M = m, M
        self.a, self.b = a, b
        self.inverted = inverted
        self.bounds = (self.m - self.a, self.M + self.b)
        self.calculateCurves()

    def calculateCurves(self):
        self.kn, self.bn = [], []
        
        if self.m == self.bounds[0]:
            self.kn.append(0)
        else:
            self.kn.append(round(1/self.a, 7))
            
        if self.M == self.bounds[1]:
            self.kn.append(0)
        else:
            self.kn.append(round(-1/self.b, 7))

        self.bn = [-self.kn[0] * (self.bounds[0]),
                   1 - self.kn[1] * self.M]
        
        if self.inverted:
            for i in range(len(self.kn)):
                self.kn[i] = -self.kn[i]
                self.bn[i] = 1 - self.bn[i]

        self.kn = tuple(self.kn)
        self.bn = tuple(self.bn)
    
    def _add(self, other):
        a = self.a + other.a
        b = self.b + other.b
        m = self.m + other.m
        M = self.M + other.M
        return (m, M, a, b)
    
    def __add__(self, other):
        assert isinstance(other, Fuzzy_set), 'Сложение выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        return Fuzzy_set(*self._add(other))

    def __iadd__(self, other):
        assert isinstance(other, Fuzzy_set), 'Сложение выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        self.m, self.M, self.a, self.b = self.add(other)
        self.bounds = (self.m - self.a, self.M + self.b)
        self.calculateCurves()
        return self

    def _sub(self, other):
        a = self.a + other.b
        b = self.b + other.a
        m = self.m - other.M
        M = self.M - other.m
        return (m, M, a, b)
		
    def __sub__(self, other):
        assert isinstance(other, Fuzzy_set), 'Вычитание выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        return Fuzzy_set(*self._sub(other))
    
    def __isub__(self, other):
        assert isinstance(other, Fuzzy_set), 'Вычитание выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        self.m, self.M, self.a, self.b = self._sub(other)
        self.bounds = (self.m - self.a, self.M + self.b)
        self.calculateCurves()
        return self

    def _mul(self, other):
        a = self.m * other.m - (self.bounds[0])*(other.bounds[0])
        b = (self.bounds[1])*(other.bounds[1]) - self.M * other.M
        m = self.m * other.m
        M = self.M * other.M
        return (m, M, a, b)
		
    def __mul__(self, other):
        assert isinstance(other, Fuzzy_set), 'Умножение выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        return Fuzzy_set(*self._mul(other))

    def __imul__(self, other):
        assert isinstance(other, Fuzzy_set), 'Умножение выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        self.m, self.M, self.a, self.b = self._mul(other)
        self.bounds = (self.m - self.a, self.M + self.b)
        self.calculateCurves()
        return self

    def _truediv(self, other):
        a = (self.m * other.b + other.M * self.a) / (other.M**2 + other.M * other.b)
        b = (other.m * self.b + self.M * other.a) / (other.m**2 + other.m * other.a)
        m = self.m / other.M
        M = self.M / other.m
        return (m, M, a, b)

    def __truediv__(self, other):
        assert isinstance(other, Fuzzy_set), 'Деление выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        return Fuzzy_set(*self._truediv(other))

    def __itruediv__(self, other):
        assert isinstance(other, Fuzzy_set), 'Деление выполняется только с нечёткими множествами'
        assert self.inverted == other.inverted, 'Невозможно выполнить операцию'
        self.m, self.M, self.a, self.b = self._truediv(other)
        self.bounds = (self.m - self.a, self.M + self.b)
        self.calculateCurves()
        return self
    
    def _pow(self, exp):
        a, b, m, M = self.a, self.b, self.m, self.M
        for _ in range(exp-1):
            a = m**2 - (m - a)**2
            b = (M + b)**2 - M**2
            m *= exp
            M *= exp
        return (m, M, a, b)

    def __pow__(self, exp):
        assert isinstance(exp, int), 'A**X, X - целое число'
        return Fuzzy_set(*self._pow(exp))

    def __ipow__(self, exp):
        assert isinstance(exp, int), 'A**X, X - целое число'        
        self.a, self.m, self.M, self.b = self._pow(exp)
        self.bounds = (self.m - self.a, self.M + self.b)
        self.calculateCurves()
        return self

    def __len__(self) -> float:
        return self.bounds[1] - self.bounds[0] + 1

    def __pos__(self):
        return self

    def __eq__(self, other) -> bool:
        if (self.a == other.a and
            self.m == other.m and
            self.M == other.M and
            self.b == other.b):
                return True
        return False

    def __ne__(self, other):
        if (self.a != other.a or
            self.m != other.m or
            self.M != other.M or
            self.b != other.b):
                return True
        return False

    def __hash__(self):
        return hash((self.m, self.M, self.a, self.b, self.inverted))

    def __repr__(self) -> str:
        return f'Fuzzy_set(m: {self.m}, M: {self.M}, a: {self.a}, b: {self.b})'

    def __round__(self, n=0):
        m = round(self.m, n)
        M = round(self.M, n)
        a = round(self.a, n)
        b = round(self.b, n)
        return Fuzzy_set(m, M, a, b)

    def __contains__(self, item) -> bool:
        assert isinstance(item, int) or isinstance(item, float) or isinstance(item, Fuzzy_set)
        
        if isinstance(item, int) or isinstance(item, float):
            return self.bounds[0] <= item <= self.bounds[1]
        return (self.bounds[0] <= item.bounds[0]) and (self.bounds[1] >= item.bounds[1])

    def __invert__(self):
        ''' Дополнение А (Инверсия А) '''
        return Fuzzy_set(self.m, self.M, self.a, self.b, inverted=True)
    
    '''
    def __and__(self, other):
        #TODO
        #Пересечение A
        assert (self.m - self.a < other.M + other.b or
                self.M  + self.b < other.m - other.a), 'Недопустимое действие'
        return Fuzzy_set(self.m, self.M, self.a, self.b)

    def __iand__(self, other):
        #TODO
        #Пересечение A
        assert (self.m - self.a < other.M + other.b or
                self.M  + self.b < other.m - other.a), 'Недопустимое действие'
        return self

    def __or__(self, other):
        #TODO
        #Обьединение A
        assert (self.m - self.a < other.M + other.b or
                self.M  + self.b < other.m - other.a), 'Недопустимое действие'
        return Fuzzy_set(self.m, self.M, self.a, self.b)

    def __ior__(self, other):
        #TODO
        #Обьединение A
        assert (self.m - self.a < other.M + other.b or
                self.M  + self.b < other.m - other.a), 'Недопустимое действие'
        return self
    '''

    def E(self) -> float:
        ''' Средннее значение A '''
        a, b = self.bounds[0], self.bounds[1]
        return (a + 2*self.m + 2*self.M + b)/6

    def var(self) -> float:
        ''' Дисперсия A '''
        a, b = self.bounds[0], self.bounds[1]
        return ((b - a)**2 + 2*(b - a)*(self.M - self.m)
                + 3*(self.M - self.m)**2)/24

    def probability(self, x: float, accuracy: int = 4) -> float:
        assert isinstance(x, int) or isinstance(x, float)
        f = lambda x, n: self.kn[n] * x + self.bn[n]
        if self.m <= x <= self.M:
            return 1
        elif x <= self.bounds[0] or x >= self.bounds[1]:
            if self.inverted:
                return 1
            return 0
        elif self.bounds[0] < x < self.m:
            return round(f(x, 0), accuracy)
        else:
            return round(f(x, 1), accuracy)

    def suprInf(self, other, supr: bool = True) -> float:
        bounds = (min(self.bounds[0], other.bounds[0]) - 1,
                  max(self.bounds[1], other.bounds[1]) + 1)
        X = [i/10 for i in range(bounds[0]*10, bounds[1]*10+1)]
        A = [self.probability(i) for i in X]
        B = [other.probability(i) for i in X]
        if supr:
            return max(min(i) for i in zip(A, B))
        else:
            return min(max(i) for i in zip(A, B))
    
    def supr(self, other) -> float:
        return self.suprInf(other)

    def inf(self, other) -> float:
        return self.suprInf(other, supr=False)

    def params(self) -> tuple: 
        return (self.m, self.M, self.a, self.b)

    def boundsInfo(self) -> None:
        print(f'[{self.m-self.a}, {self.m}]; [{self.M}, {self.M+self.a}]')

    def pcoefs(self) -> None:
        print(f'Y0 = {self.kn[0]} * X1 + {self.bn[0]}')
        print(f'Y1 = {self.kn[1]} * X1 + {self.bn[1]}')

    def coefs(self) -> tuple:
        return ((self.kn[0], self.kn[1]),(self.bn[0], self.bn[1]))

    def draw_set(self) -> None:
        _, ax = plt.subplots()
        X = (self.bounds[0], self.m, self.M, self.bounds[1])
        Y = (0, 1, 1, 0)
        if self.inverted:
            Y = (1, 0, 0, 1)
        ax.plot(X, Y)
            
        if not (self.m == self.M == self.b == 0 or
                self.m == self.M == self.a == 0):
            if self.bounds[0] != self.m:
                ax.plot((self.m, self.m), (0, 1), dashes=[6, 2])
            if self.bounds[1] != self.M:
                ax.plot((self.M, self.M), (0, 1), dashes=[6, 2])
        ax.set(xlabel='X', ylabel='Probability',
                title='Fuzzy set mapping')
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
            print('Поле пустое')
        else:
            print(f'{len(self.field)} множества:')
            i = 1
            for fuzzy_set in self.field:
                print(f'{i}: {fuzzy_set.params()}')
                i += 1

    def draw_field(self, bounds=False) -> None:
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
                title='Fuzzy set\'s field mapping')
        ax.grid()
        plt.show()