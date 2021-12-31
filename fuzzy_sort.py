def chuPark(self,*fuzzy_sets, w: float = 1):
    cp = []
    for fuzzy_set in fuzzy_sets:
        t1 = (fuzzy_set.bounds[0] + fuzzy_set.m +
            fuzzy_set.M + fuzzy_set.bounds[1]) / 4
        t2 = w * (fuzzy_set.m + fuzzy_set.M) / 2
        cp.append((t1 + t2, fuzzy_set))
    return [i[1] for i in sorted(cp)]

def chung(self, *fuzzy_sets):
    ch = []
    for fuzzy_set in fuzzy_sets:
        t = (fuzzy_set.M**2 +
             fuzzy_set.M*fuzzy_set.bounds[1] +
             fuzzy_set.fuzzy_set.bounds[1]**2 -
             fuzzy_set.bounds[0]**2 -
             fuzzy_set.m*fuzzy_set.bounds[0] -
             fuzzy_set.m**2) / 6
        ch.append((t, fuzzy_set))
    return [i[1] for i in sorted(ch)]

def kaufmanGupt(self, *fuzzy_sets):
    #TODO
    pass

def jane(self, *fuzzy_sets):
    #TODO
    pass

def duboisPrades(self, *fuzzy_sets):
    #TODO
    pass