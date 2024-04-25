import numpy as np
import pickle
import random

def generate_lists(num_lists=11):
    lists = []
    size = 10**6  
    for i in range(num_lists):
        
        num_repeats = int(size * 10 * i / 100)
        
        if num_repeats > 0:
            repeated_element = np.random.randint(0, 1000000)
            list_data = np.array([repeated_element] * num_repeats + list(np.random.randint(0, 1000000, size - num_repeats)))
        else:
            list_data = np.random.randint(0, 1000000, size)

        np.random.shuffle(list_data)
        lists.append(list_data)
    return lists

def exchange(A, i, j):
    tmp = A[i]
    A[i] = A[j]
    A[j] = tmp

def Rand_Partition(A, p, r):
    i = random.randint(p, r)
    exchange(A, r, i)
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            exchange(A, i, j)
    exchange(A, i+1, r)
    return i + 1
            

def QuickSort(A, p, r):
    if p < r:
        q = Rand_Partition(A, p, r)
        QuickSort(A, p, q-1)
        QuickSort(A, q+1, r)
    return A

if __name__ == "__main__":
    data_path = "./data/Exp3/lists.pkl"
    # lists = generate_lists()
    # with open(data_path, 'wb') as f:
    #     pickle.dump(lists, f)
    with open(data_path, 'rb') as f:
        lists = pickle.load(f)
    
    # data1 = lists[0]
    data1 = np.array(list(range(20)))
    np.random.shuffle(data1)
    data_sort = QuickSort(data1, 0, len(data1)-1)
    print(data_sort)
    