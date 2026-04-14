def calc_speed_angle(row):
    import pandas as pd
    v, a = pd.to_numeric(row['HTNG_SPD_MI']), pd.to_numeric(row['HTNG_ANGL'])
    if pd.isna(v) or pd.isna(a) and row['PICH_JUDG'] != '타격' : return 0
    if (v*1.5 - a >= 117) and (v+a >= 124) and (v >= 98) and (4 <= a <= 50): return 6
    elif (v*1.5 - a >= 111) and (v+a >= 119) and (v >= 95) and (0 <= a <= 52): return 5
    elif v <= 59: return 1
    elif (v*2 - a >= 87) and (a <= 41) and (v*2 + a <= 175) and (v+a*1.3 >= 89) and (59 <= v <= 72): return 4
    elif (v+a*1.3 <= 112) and (v+a*1.55 >= 92) and (72 <= v <= 86): return 4
    elif (a <= 20) and (v+a*2.4 >= 98) and (86 <= v <= 95): return 4
    elif (v - a >= 76) and (v + a*2.4 >= 98) and (v >= 95) and (a <= 30): return 4
    elif (v + a*2 >= 116): return 3
    elif (v + a*2) <= 116: return 2
    else: return 0
    

def barrel(row):
    import pandas as pd
    s, a, j = pd.to_numeric(row['HTNG_SPD']), pd.to_numeric(row['HTNG_ANGL']), row['PICH_JUDG']
    if j != '타격': return 0
    conds = [
        (s >= 156 and 23.5 <= a < 32.9),
        (s >= 159.25 and 21.15 <= a < 37.6),
        (s >= 162.5 and 21.15 <= a < 39.95),
        (s >= 165.75 and 18.8 <= a < 47),
        (s >= 169 and a >= 18.8),
        (s >= 172.25 and a >= 16.45),
        (s >= 175.5 and a >= 14.1),
        (s >= 178.75 and a >= 11.75),
        (s >= 182 and a >= 9.4)
    ]
    return int(any(conds))
