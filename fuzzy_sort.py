from Fuzzy_set import Fuzzy_set

def сhewPark(*fuzzy_sets, w: float = 1) -> tuple:
    if not all(tuple(1 if isinstance(i, Fuzzy_set) else None for i in fuzzy_sets)):
            raise TypeError('Only Fuzzy sets allowed')
    CP = []
    for fuzzy_set in fuzzy_sets:
        t1 = (fuzzy_set.bounds[0] + fuzzy_set.m +
            fuzzy_set.M + fuzzy_set.bounds[1]) / 4
        t2 = w * (fuzzy_set.m + fuzzy_set.M) / 2
        CP.append((t1 + t2, fuzzy_set))
    return tuple(i[1] for i in sorted(CP))

def chang(*fuzzy_sets) -> tuple:
    if not all(tuple(1 if isinstance(i, Fuzzy_set) else None for i in fuzzy_sets)):
            raise TypeError('Only Fuzzy sets allowed')
    CH = []
    for fs in fuzzy_sets:
        t = (fs.M**2 +
             fs.M * fs.bounds[1] +
             fs.bounds[1]**2 -
             fs.bounds[0]**2 -
             fs.m * fs.bounds[0] -
             fs.m**2) / 6
        CH.append((t, fs))
    return tuple(i[1] for i in sorted(CH))

def kaufmanGupt(*fuzzy_sets) -> tuple:
    if not all(tuple(1 if isinstance(i, Fuzzy_set) else None for i in fuzzy_sets)):
            raise TypeError('Only Fuzzy sets allowed')
    KG = [(fs.mean(),
        (fs.M + fs.bounds[1])/2,
        fs.bounds[1] - fs.bounds[0]) for fs in fuzzy_sets]
    fuzzy_sets = list(fuzzy_sets)
    for i in range(len(fuzzy_sets) - 1):
       for j in range(len(fuzzy_sets) - i - 1):
           if (KG[j][0] > KG[j + 1][0] or
               (KG[j][0]  == KG[j + 1][0] and KG[j][1] > KG[j + 1][1]) or 
                (KG[j][0], KG[j][1]) == (KG[j + 1][0], KG[j + 1][1]) and KG[j][2] > KG[j + 1][2]):
               fuzzy_sets[j], fuzzy_sets[j+1] = fuzzy_sets[j + 1], fuzzy_sets[j]
               KG[j], KG[j+1] = KG[j + 1], KG[j]
    return tuple(fuzzy_sets)

def jane(m: float, M: float, a: float, b: float, *fuzzy_sets) -> tuple:
    if not all(tuple(1 if isinstance(i, Fuzzy_set) else None for i in fuzzy_sets)):
            raise TypeError('Only Fuzzy sets allowed')
    if not(all(tuple(1 if fs.inverted else None for fs in fuzzy_sets)) or
            all(tuple(1 if not fs.inverted else None for fs in fuzzy_sets))):
        raise TypeError('All sets must be equally inverted')
    if m == float('inf'):
        m, a = 0, 0
    elif M == float('inf'):
        M, b = 0, 0
    side = 1 if m == 0 else 0
    Jane = Fuzzy_set(m, M, a, b, fuzzy_sets[0].inverted)
    J = []
    for fs in fuzzy_sets:
        if fs.m >= Jane.m:
            J.append((fs.M, fs))
        elif fs.bounds[0] > Jane.bounds[0]: 
            x1 = round((Jane.bn[side] - fs.bn[1]) / (fs.kn[1] - Jane.kn[side]), 3)
            x2 = round((Jane.bn[side] - fs.bn[0]) / (fs.kn[0] - Jane.kn[side]), 3)
            J.append((max(x1, x2), fs))
        elif fs.bounds[1] > Jane.bounds[0]:
            x = round((Jane.bn[side] - fs.bn[1]) / (fs.kn[1] - Jane.kn[side]), 3)
            J.append((x, fs))
        else:
            J.append((Jane.bounds[0], fs))
    return tuple(fs[1] for fs in sorted(J))

def duboisPrades(*fuzzy_sets) -> tuple:
    if not all(tuple(1 if isinstance(i, Fuzzy_set) else None for i in fuzzy_sets)):
            raise TypeError('Only Fuzzy sets allowed')
    if not(all(tuple(1 if fs.inverted else None for fs in fuzzy_sets)) or
            all(tuple(1 if not fs.inverted else None for fs in fuzzy_sets))):
        raise TypeError('All sets must be equally inverted')
    DP, DP_temp = [], set()
    for fs in fuzzy_sets:
        temp = []
        for other_fs in fuzzy_sets:
            points = []
            if fs.params() == other_fs.params():
                continue
            x1 = round((fs.bn[0] - other_fs.bn[1]) / (other_fs.kn[1] - fs.kn[0]), 3)
            x2 = round((fs.bn[0] - other_fs.bn[0]) / (other_fs.kn[0] - fs.kn[0]), 3)
            
            if 0 < fs.probability(x1) <= 1 and x1 <= fs.m:
                points.append(x1)
            elif 0 < fs.probability(x2) <= 1 and x2 <= fs.m:
                points.append(x2)
            if points:
                temp.append(points[0])
        if len(temp) == len(fuzzy_sets)-1:
            DP.append((1, fs))
            DP_temp.add(fs)

    for fs in fuzzy_sets:
        temp, points = [], []
        if fs in DP_temp:
            continue
        for other_fs in fuzzy_sets:
            if fs.params() == other_fs.params():
                continue
            if fs.kn[0] != other_fs.kn[0]:
                x1 = round((fs.bn[0] - other_fs.bn[0]) / (other_fs.kn[0] - fs.kn[0]), 3)
                x2 = round((fs.bn[0] - other_fs.bn[1]) / (other_fs.kn[1] - fs.kn[0]), 3)
                if (x1 < fs.m and 0 < fs.probability(x1) <= 1 or
                    x2 < fs.m and 0 < fs.probability(x2) <= 1):
                    points.append((x1, 0))
                    continue
            if fs.kn[1] != other_fs.kn[0]:
                x = round((fs.bn[1] - other_fs.bn[0]) / (other_fs.kn[0] - fs.kn[1]), 3)
                if x > fs.m and 0 < fs.probability(x) <= 1:
                    points.append((x, fs.probability(x)))
        if len(points) == len(fuzzy_sets) - 1:
            DP.append((max(points)[1], fs))

    return tuple(fs[1] for fs in sorted(DP))