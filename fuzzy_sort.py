def сhewPark(*fuzzy_sets, w: float = 1) -> tuple:
    cp = []
    for fuzzy_set in fuzzy_sets:
        t1 = (fuzzy_set.bounds[0] + fuzzy_set.m +
            fuzzy_set.M + fuzzy_set.bounds[1]) / 4
        t2 = w * (fuzzy_set.m + fuzzy_set.M) / 2
        cp.append((t1 + t2, fuzzy_set))
    return tuple(i[1] for i in sorted(cp))

def chang(*fuzzy_sets) -> tuple:
    ch = []
    for fs in fuzzy_sets:
        t = (fs.M**2 +
             fs.M * fs.bounds[1] +
             fs.bounds[1]**2 -
             fs.bounds[0]**2 -
             fs.m * fs.bounds[0] -
             fs.m**2) / 6
        ch.append((t, fs))
    return tuple(i[1] for i in sorted(ch))

def kaufmanGupt(*fuzzy_sets) -> tuple:
    kg = [((fs.bounds[0] + fs.bounds[1] + 2 * (fs.m + fs.M))/6,
        (fs.M + fs.bounds[1])/2,
        fs.bounds[1] - fs.bounds[0]) for fs in fuzzy_sets]

    for i in range(len(fuzzy_sets) - 1):
       for j in range(len(fuzzy_sets) - i - 1):
           if kg[j][0] >= kg[j + 1][0] or \
               (kg[j][0], kg[j][1])  == (kg[j + 1][0], kg[j + 1][1]) \
                   and kg[j][2] > kg[j + 1][2]:
               fuzzy_sets[j], fuzzy_sets[j+1] = fuzzy_sets[j + 1], fuzzy_sets[j]
               kg[j], kg[j+1] = kg[j + 1], kg[j]
    return tuple(fuzzy_sets)

def jane(b0: float, b1: float, *fuzzy_sets) -> tuple:
    #TODO
    pass

def duboisPrades(*fuzzy_sets) -> tuple:
    #TODO
    pass