"""
similarity measures
"""
def jaccard_similarity(d1, d2):
    s1 = set(d1)
    s2 = set(d2)
    intersect = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union == 0:
        return 0
    return float(intersect) / union

def jaccard_similarity_with_weight(d1, d2, weight):
    s1 = set(d1)
    s2 = set(d2)
    intersect = s1.intersection(s2)
    union = s1.union(s2)
    if len(union) == 0:
        return 0
    i = .0
    u = .0
    for x in intersect:
        i += weight[x]
    for x in union:
        u += weight[x]
    return i / u

def common_word_with_weight(d1, d2, weight):
    s1 = set(d1)
    s2 = set(d2)
    intersect = s1.intersection(s2)
    i = .0
    for x in intersect:
        i += weight[x]
    return i


