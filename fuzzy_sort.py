from Fuzzy_set import Fuzzy_set

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

def jane(m: float, M: float, a: float, b: float, *fuzzy_sets) -> tuple:
    assert (all(tuple(fs if fs.inverted else None for fs in fuzzy_sets)) or
            all(tuple(fs if not fs.inverted else None for fs in fuzzy_sets))), 'All sets must be equally inverted'
    if (m, a) == (float('inf'), float('inf')):
        m, a, side = 0, 0, 1
    elif (M, b) == (float('inf'), float('inf')):
        M, b, side = 0, 0, 0
    Jane = Fuzzy_set(m, M, a, b, fuzzy_sets[0].inverted)
    J = []
    for fs in fuzzy_sets:
        if fs.m >= Jane.m:
            J.append((fs.M, fs))
        elif fs.bounds[0] > Jane.bounds[0]: 
            x1 = round((Jane.bn[side] - fs.bn[1]) / (fs.kn[1] - Jane.bn[side]), 3)
            x2 = round((Jane.bn[side] - fs.bn[0]) / (fs.kn[0] - Jane.bn[side]), 3)
            J.append((max(x1, x2), fs))
        elif fs.bounds[1] > Jane.bounds[0]:
            x = round((Jane.bn[side] - fs.bn[1]) / (fs.kn[1] - Jane.bn[side]), 3)
            J.append((x, fs))
        else:
            J.append((Jane.bounds[0], fs))
    return tuple(fs[1] for fs in sorted(J))

def duboisPrades(*fuzzy_sets) -> tuple:
    #TODO
    pass