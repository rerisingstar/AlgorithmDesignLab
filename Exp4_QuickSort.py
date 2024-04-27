import numpy as np
import pickle
import random
import time

# def generate_lists(num_lists=11):
#     lists = []
#     size = 10**6  
#     for i in range(num_lists):
        
#         num_repeats = int(size * 10 * i / 100)
        
#         if num_repeats > 0:
#             repeated_element = np.random.randint(0, 1000000)
#             list_data = np.array([repeated_element] * num_repeats + list(np.random.randint(0, 1000000, size - num_repeats)))
#         else:
#             list_data = np.random.randint(0, 1000000, size)

#         np.random.shuffle(list_data)
#         lists.append(list_data)
#     return lists
def generate_sort(size, num_list=11):
    lists = []
    for i in range(num_list):
        repeat_num = int(size * 10 * i / 100)
        list_tmp = list(range(size - repeat_num))
        repeat = random.sample(range(size), 1)[0]
        list_tmp += [repeat] * repeat_num
        random.shuffle(list_tmp)
        lists.append(list_tmp)
    return lists

def exchange(A, i, j):
    if i != j:
        tmp = A[i]
        A[i] = A[j]
        A[j] = tmp

def Rand_Partition(A, p, r):
    i = random.randint(p, r)
    # exchange(A, r, i)
    A[r], A[i] = A[i], A[r]
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    # exchange(A, i+1, r)
    A[i+1], A[r] = A[r], A[i+1]
    if i == r-1:
        return int((p + r) / 2)
    else:
        return i + 1
            

def QuickSort(A, p, r):
    if p < r:
        q = Rand_Partition(A, p, r)
        QuickSort(A, p, q-1)
        QuickSort(A, q+1, r)
    return A

def quicksort3way(arr, low, high):
    if low >= high:
        return

    pivot = arr[low]


    lt = low     
    gt = high   
    i = low      


    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
        else:
            i += 1


    quicksort3way(arr, low, lt - 1)
    quicksort3way(arr, gt + 1, high)
    
    return arr

if __name__ == "__main__":
    lists = generate_sort(1000000)
    data_path = "./data/Exp3/lists.pkl"
    # # lists = generate_lists()
    with open(data_path, 'wb') as f:
        pickle.dump(lists, f)
    # with open(data_path, 'rb') as f:
    #     lists = pickle.load(f)
    
    
    # data1 = lists[0]
    # data1 = np.array(list(range(5)) + [3, 3, 4, 4, 5])
    # np.random.shuffle(data1)
    # data_sort = QuickSort(data1, 0, len(data1)-1)
    # print(data_sort)
    
    # lists = [[1, 1, 1, 1, 1]]
    
    for i in range(0, 11):
        data = lists[i]
        print(f"{i}th data, Repeat number = {10**4 * 10 * i}")
        start_time = time.perf_counter()
        # data_sort = QuickSort(data, 0, len(data)-1)
        data_sort = QuickSort(data, 0, len(data)-1)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"    Algorithm execution time: {elapsed_time:.3f} s")