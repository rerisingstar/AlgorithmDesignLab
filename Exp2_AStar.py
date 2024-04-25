import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

NodeLocation = Tuple[int, int]

class Graph:
    def __init__(self, 
        width: int,
        height: int,
    ):
        self.width = width
        self.height = height
        
        self.map = np.full((width, height), 0)
        
    
    def _in_graph(self, node: NodeLocation) -> bool:
        x, y = node
        if x >= 0 and x < self.width:
            if y >= 0 and y < self.height:
                return True
        return False
    
    def add_geography(self, 
        deserts: List[NodeLocation]=None, 
        streams: List[NodeLocation]=None, 
        obstacles: List[NodeLocation]=None,
    ):
        if deserts:
            for de in deserts:
                self.map[de[0]-1, de[1]-1] = 1
        if streams:
            for st in streams:
                self.map[st[0]-1, st[1]-1] = 2
        if obstacles:
            for ob in obstacles:
                self.map[ob[0]-1, ob[1]-1] = 3
    
    def neighbors(self, node: NodeLocation) -> List[NodeLocation]:
        dirs = [
            [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ]
        result = []
        for dir in dirs:
            neighbor = [node[0] + dir[0], node[1] + dir[1]]
            if self._in_graph(neighbor):
                result.append(neighbor)
        return result

    def cost(self, current: NodeLocation, next: NodeLocation):
        movement = [next[0] - current[0], next[1] - current[1]]
        if movement in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
            move_cost = 1
        elif movement in [[1, 1], [1, -1], [-1, 1], [-1, -1]]:
            move_cost = math.sqrt(2)
        
        if self.map[next[0], next[1]] == 1: # deserts
            geo_cost = 4
        elif self.map[next[0], next[1]] == 2: # streams
            geo_cost = 2
        elif self.map[next[0], next[1]] == 3: # obstacle
            geo_cost = int('inf')
        else:
            geo_cost = 0
        
        return move_cost + geo_cost
    
    def draw(self):
        geo_colors = {
            1: 'wheat',
            2: 'blue',
            0: 'white',
            3: 'gray'
        }
        fig, ax = plt.subplots()
        
        ax.set_aspect('equal', adjustable='box')
        
        for row in range(self.width):
            for col in range(self.height):
                color = geo_colors[self.map[row, col]] 
                ax.add_patch(patches.Rectangle((row, col), 1, 1, color=color)) 
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.xaxis.set_ticks_position('top')
        
        for x in range(1, self.width):
            plt.axvline(x=x, linewidth=0.5, color='gray')
        for y in range(1, self.height):
            plt.axhline(y=y, linewidth=0.5, color='gray')

        interval = 5
        label_list = list(i+1 for i in range(interval-1, self.width, interval))
        xtick_list = [i-0.5 for i in label_list]
        ax.set_xticks(xtick_list)
        ax.set_xticklabels(label_list)
        label_list = list(i+1 for i in range(interval-1, self.height, interval))
        ytick_list = [i-0.5 for i in label_list]
        ax.set_yticks(ytick_list)
        ax.set_yticklabels(label_list)
        
        plt.show()