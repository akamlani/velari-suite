def trsfrm_normalizer(x):
    return (x - min(x)) / (max(x) - min(x)) if max(x) != min(x) else 0
