import random
import pickle
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import math
from typing import List, Optional

class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        
    def __eq__(self, value: object) -> bool:
        if isinstance(value, Point):
            if value.x == self.x and value.y == self.y:
                return True
        return False
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def __sub__(self, b):
        if not isinstance(b, Point):
            raise ValueError
        else:
            return Point(self.x - b.x, self.y - b.y)
    
def generate_points(num_points=100, wide=100, length=100, save=False):
    
    points = []
    for _ in range(num_points):
        x = random.uniform(0, wide)
        y = random.uniform(0, length)
        points.append(Point(x, y))
    
    if save:
        save_path = f"./data/Exp1/{num_points}points.pkl"    
        with open(save_path, "wb") as f:
            pickle.dump(points, f)
    
    return points

def get_points(path):
    with open(path, "rb") as f:
        points = pickle.load(f)
    return points

def cal_cross_product(v1: Point, v2: Point):
    """计算向量v1和向量v2的叉积，这里的向量用点表示，因为都是2维的

    Args:
        X (Point): _description_
        Y (Point): _description_
    """
    
    return v1.x * v2.y - v1.y * v2.x

def cal_g(A: Point, B: Point, P: Point):
    """计算g,g>0则P在AB上面
        实际上是向量AB和向量AP的叉积
    Args:
        A (Point)
        B (Point)
        P (Point)
    """
    # if A.x >= B.x:
    #     C = copy.deepcopy(B)
    #     B = copy.deepcopy(A)
    #     A = copy.deepcopy(C)
    
    AB = Point(B.x - A.x, B.y - A.y)
    AP = Point(P.x - A.x, P.y - A.y)
    
    g = cal_cross_product(AB, AP)
    return g

def in_triangle(A: Point, B: Point, C: Point, P: Point) -> bool:
    """判断P在不在ABC的三角形内

    Args:
        A (Point)
        B (Point)
        C (Point)
        P (Point)

    Returns:
        bool
    """
    
    if (cal_g(A, B, P) * cal_g(A, B, C) >= 0
        and cal_g(A, C, P) * cal_g(A, C, B) >= 0
        and cal_g(B, C, P) * cal_g(B, C, A) >= 0):
        return True
    else:
        return False

def left_turn(p1: Point, p2: Point, p3: Point) -> bool:
    v1 = p2 - p1
    v2 = p3 - p2
    return cal_cross_product(v1, v2) > 0

def get_inside_point(a: Point, b: Point, c: Point):
        # 找三角形的一个内点，凸包上连续三点的三角形的内点肯定也在凸包内
        core_x = (a.x + b.x + c.x) / 3
        core_y = (a.y + b.y + c.y) / 3
        return Point(core_x, core_y)

def cal_angle(p1: Point, p2: Point):
        return math.atan2(p2.y - p1.y, p2.x - p1.x)


## DEBUG
def draw_seqs(seq_1, seq_2, seq_3):
    for i, p in enumerate(seq_1): 
        plt.scatter(p.x, p.y, color="blue") 
        plt.text(p.x, p.y, str(i+1), fontsize=12, ha="right")
    for i, p in enumerate(seq_2): 
        plt.scatter(p.x, p.y, color="yellow") 
        plt.text(p.x, p.y, str(i+1), fontsize=12, ha="right")
    for i, p in enumerate(seq_3): 
        plt.scatter(p.x, p.y, color="red") 
        plt.text(p.x, p.y, str(i+1), fontsize=12, ha="right")

def draw_points(merged_seq):
    for i, p in enumerate(merged_seq): 
        plt.scatter(p.x, p.y, color="blue") 
        # plt.text(p.x, p.y, str(i+1), fontsize=12, ha="right")
        
def draw_merged(merged_seq):
    for i, p in enumerate(merged_seq): 
        plt.scatter(p.x, p.y, color="blue") 
        plt.text(p.x, p.y, str(i+1), fontsize=12, ha="right")

def draw_cov_hull(edge_sort):
    draw_points(edge_sort)
    for i, p in enumerate(edge_sort):
        if (i + 1) >= len(edge_sort):
            plt.arrow(p.x, p.y, edge_sort[0].x - p.x, edge_sort[0].y - p.y, color="red")
        else:
            plt.arrow(p.x, p.y, edge_sort[i+1].x - p.x, edge_sort[i+1].y - p.y, color="red")


def BruteForceCH(points):
    in_flag = set()
    for A in tqdm(points):
        for B in points:
            if A == B:
                continue
            for C in points:
                if C == A or C == B:
                    continue
                for D in points:
                    if D == A or D == B or D == C:
                        continue
                    
                    if in_triangle(A, B, C, D):
                        in_flag.add(D)
                    if in_triangle(A, B, D, C):
                        in_flag.add(C)
                    if in_triangle(A, C, D, B):
                        in_flag.add(B)
                    if in_triangle(B, C, D, A):
                        in_flag.add(A)
                        
                        
    edge_points = [p for p in points if p not in in_flag]
    edge_points.sort(key=lambda p: p.x)
    A = edge_points[0]
    B = edge_points[-1]
    SL = []
    SU = []
    for P in edge_points[1:-1]:
        if cal_g(A, B, P) < 0:
            SL.append(P)
        if cal_g(A, B, P) > 0:
            SU.append(P)
    SL.sort(key=lambda p: p.x)
    SU.sort(key=lambda p: p.x, reverse=True)
    edge_sort = [A] + SL + [B] + SU
    
    return edge_sort

def GrahamScan(points: List[Point], have_sorted: bool=False):
    edge_sort = []
    
    if not have_sorted:
        # find p0
        min_y_ind = 0
        for ind, point in enumerate(points):
            if point.y < points[min_y_ind].y:
                min_y_ind = ind
        edge_sort.append(points[min_y_ind])
        points = points[:min_y_ind] + points[min_y_ind+1:]

        #按极角排序
        def angle(p: Point):
                p0 = edge_sort[0]
                return math.atan2(
                    p.y - p0.y,
                    p.x - p0.x
                )
        sorted_points = sorted(points, key=angle)
    else:
        edge_sort.append(points[0])
        sorted_points = points[1:]

    #挨个考虑
    edge_sort.append(sorted_points[0])
    # edge_sort.append(sorted_points[1])
    for p in sorted_points[1:]:
        top_p = edge_sort[-1]
        nttop_p = edge_sort[-2]
        while (not left_turn(nttop_p, top_p, p)) and len(edge_sort) >= 3:
            edge_sort.pop()
            top_p = edge_sort[-1]
            nttop_p = edge_sort[-2]
        edge_sort.append(p)
    
    return edge_sort

def DevideAndConquer(points: List[Point]) -> List[Point]:
    debug = False
    
    # preprocess
    if len(points) == 3:
        if left_turn(points[0], points[1], points[2]):
            return points
        else:
            return [points[0], points[2], points[1]]
    elif len(points) <= 2:
        if len(points) == 2:
            p_inside = Point((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2)
            if cal_angle(Point(0,0), points[0]-p_inside) > cal_angle(Point(0,0), points[1]-p_inside):
                points = [points[1], points[0]]
        return points
    
    
    min_x = float("inf")
    max_x = float("-inf")
    for p in points:
        if p.x > max_x:
            max_x = p.x
        if p.x < min_x:
            min_x = p.x
    mid_x = (min_x + max_x) / 2
    
    QL = []
    QR = []
    for p in points:
        if p.x <= mid_x:
            QL.append(p)
        else:
            QR.append(p)
    
    QL = DevideAndConquer(QL)
    QR = DevideAndConquer(QR)
    
    if debug:
        draw_cov_hull(QL)
        draw_cov_hull(QR)
        plt.show()
        plt.clf()
    # 构造三个序列
    try:
        p_inside = get_inside_point(QL[0], QL[1], QL[2]) # 找凸包QL的一个内点
    except IndexError:
        if len(QL) == 2:
            p_inside = Point((QL[0].x + QL[1].x) / 2, (QL[0].y + QL[1].y) / 2)
        elif len(QL) == 1:
            p_inside = QL[0]
    # 在CH(QR)中找与p的极角最大和最小顶点u和v
    min_angle = math.pi
    max_angle = -math.pi
    for ind, p in enumerate(QR):
        angle = cal_angle(Point(0,0), p-p_inside)
        if angle > max_angle:
            v_id = ind
            max_angle = angle
        if angle < min_angle:
            u_id = ind
            min_angle = angle
    
    #按逆时针方向排列的CH(QL)的所有顶点
    min_angel = math.pi
    for ind, p in enumerate(QL):
        if cal_angle(Point(0,0), p-p_inside) < min_angel:
            min_ind_in_seq1 = ind
            min_angel = cal_angle(Point(0,0), p-p_inside)
    seq_1 = (QL + QL)[min_ind_in_seq1:min_ind_in_seq1+len(QL)]
    
    if v_id < u_id: #按逆时针方向排列的CH(QR)从u到v的顶点
        seq_2 = (QR + QR)[u_id:v_id+len(QR)+1]
    else:
        seq_2 = QR[u_id:v_id+1]
        
    if v_id < u_id: #按顺时针方向排列的CH(QR)从u到v的顶点
        seq_3 = list(reversed(QR[v_id+1:u_id]))
    else:
        seq_3 = list(reversed((QR + QR)[v_id+1:u_id+len(QR)]))
    
    def _find_min_angle(p1: Optional[Point]=None, p2: Optional[Point]=None, p3: Optional[Point]=None) -> int:
        """找到与p_inside极角最小的点，返回它的index
        """
        p_list = [p for p in [p1, p2, p3] if p]
        min_angle = math.pi
        for ind, p in enumerate(p_list):
            angle = cal_angle(p_inside, p)
            if angle < min_angle:
                min_ind = ind
                min_angle = angle
        for ind, p in enumerate([p1, p2, p3]):
            if p_list[min_ind] == p:
                return ind+1
    
    if debug:
        draw_seqs(seq_1, seq_2, seq_3)
        plt.show()
        plt.clf()
    merged_seq = [p_inside]
    while True:
        if len(seq_1) == 0 and len(seq_2) == 0 and len(seq_3) == 0:
            break
        
        top_seq_1 = seq_1[0] if len(seq_1) != 0 else None
        top_seq_2 = seq_2[0] if len(seq_2) != 0 else None
        top_seq_3 = seq_3[0] if len(seq_3) != 0 else None
        
        min_ind = _find_min_angle(top_seq_1, top_seq_2, top_seq_3)
        # seq_name = f"seq_{min_ind}"
        ind2seq = {
            1: seq_1,
            2: seq_2,
            3: seq_3
        }
        merged_seq.append(ind2seq[min_ind][0])
        ind2seq[min_ind].pop(0)

    if debug:
        draw_merged(merged_seq)
        plt.show()
        plt.clf()
    cov_hull = GrahamScan(merged_seq, have_sorted=True)      
    
    if debug:
        draw_cov_hull(cov_hull[1:])
        plt.show()
        plt.clf()
    return cov_hull[1:]

if __name__ == "__main__":
    point_num = 50
    alg = "DAC"
    points = get_points(f"./data/Exp1/{point_num}points.pkl")
    # points = generate_points(point_num, save=True)
    
    formatted_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    start_time = time.perf_counter()
    
    if alg == "BF":
        edge_sort = BruteForceCH(points)
    elif alg == 'GS':
        edge_sort = GrahamScan(points)
    elif alg == 'DAC':
        edge_sort = DevideAndConquer(points)
    else:
        raise ValueError("Not correct algotirhm")
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Algorithm execution time: {elapsed_time * 1000:.3f} ms")
    
    for p in points:
        plt.scatter(p.x, p.y, color="blue")
    for i, p in enumerate(edge_sort):
        if (i + 1) >= len(edge_sort):
            plt.arrow(p.x, p.y, edge_sort[0].x - p.x, edge_sort[0].y - p.y, color="red")
        else:
            plt.arrow(p.x, p.y, edge_sort[i+1].x - p.x, edge_sort[i+1].y - p.y, color="red")
    
    result_path = f'./results/Exp1/{formatted_time}_{alg}_{point_num}points'
    os.makedirs(result_path)
    
    with open(result_path + '/edge_points.txt', 'w') as f:
        f.write(str(edge_sort)+'\n')
        f.write(f'{elapsed_time * 1000:.3f}ms')
    plt.savefig(result_path + '/figure.png')
    plt.show()
