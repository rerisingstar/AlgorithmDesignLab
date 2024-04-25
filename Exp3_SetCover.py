import pulp
import random
import copy
from typing import Tuple, List
import pickle

def generate_data(X_size: int) -> Tuple[set, set]:
    """
    X是总集合
    F是X的子集合族
    C是求解的最小集族
    """
    X = set(random.sample(range(0, 10000), X_size))
    S0 = set(random.sample(list(X), 20))
    F = [S0]
    last_set = copy.deepcopy(X)
    last_set -= S0
    while len(last_set) > 20:
        S_i = set(random.sample(list(last_set), 20))
        last_set -= S_i
        F.append(S_i)
    F.append(last_set)
    
    return X, F

def Greedy(sets: Tuple[set, set]):
    X, F = sets
    remain_e = copy.deepcopy(X)
    selected_subsets = []
    
    while remain_e:
        best_subset = None
        max_cover = 0
        
        for subset in F:
            cover = len(remain_e & subset)
            if cover > max_cover:
                max_cover = cover
                best_subset = subset
            
        selected_subsets.append(best_subset)
        remain_e -= best_subset
    
    return selected_subsets

def LP(sets: Tuple[set, set], theta=0.5):
    X, F = sets
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(len(F)), lowBound=0, cat='Continuous')
    prob += pulp.lpSum(x[i] for i in range(len(F)))
    for e in X:
        prob += pulp.lpSum(x[i] for i in range(len(F)) if e in F[i]) >= 1
    prob.solve()
    
    selected_subsets = []
    for i in range(len(F)):
        if x[i].varValue >= theta:
            selected_subsets.append(F[i])

    return selected_subsets

def check(selected_subsets: List[set], sets: Tuple[set, set]):
    X, F = sets
    merged_set = set()
    for subset in selected_subsets:
        merged_set = merged_set.union(subset)
    
    return X == merged_set

if __name__ == "__main__":
    element_num = 5000
    sets = generate_data(element_num)
    data_path = './data/Exp2/'
    with open(data_path + f'sets{element_num}.pkl', 'wb') as f:
        pickle.dump(sets, f)
        
    # ss = Greedy(sets100)
    ss = LP(sets)
    
    # print(ss)
    print(check(ss, sets))