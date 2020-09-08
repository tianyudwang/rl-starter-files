import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import utils

class ExpertAgent:
    """
    An agent that sees the full grid, defines a cost function,
    and choose actions according to shortest path from Dijkstra algorithm
    """

    def __init__(self):
        # object to index
        self.obj_to_idx = {
            "Empty":    1,
            "Wall":     2,
            "Goal":     8,
            "Lava":     9,
            "Lawn":     11,
        }

        # cost to arrive at this object
        self.idx_to_cost = {
            1:      1,      # empty
            2:      None,   # wall, impassable
            8:      1,      # goal
            9:      10,     # lava
            11:     0.5,    # lawn
        }
        
        controls = [0, 1, 2, 3]
        displacements = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # control to displacement
        self.cont_to_disp = dict(zip(controls, displacements))

        # displacement to control
        self.disp_to_cont = dict(zip(displacements, controls))

        self.agent_pos = None
        self.goal_pos = None

    def reset(self, grid, agent_pos):
        """
        Reset the expert at the beginning of the episode
        """
        assert grid.ndim == 2
        self.grid = grid
        self.grid_width, self.grid_height = self.grid.shape
        self.num_nodes = np.prod(self.grid.shape)
        self.agent_pos = agent_pos

        self.cost_graph = self.encode_cost()

    def update_agent_pos(self, agent_pos):
        """
        Updates agent position at each step
        """
        self.agent_pos = agent_pos

    def encode_cost(self):
        """
        Defines a cost function for one step transitions
        state to next state cost is equal to cost of arriving at next state
        """
        # construct cost graph
        cost_graph = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                idx = self.sub2ind(i, j)
                cost = self.idx_to_cost[self.grid[i, j]]
                
                # skip assigning cost to walls
                if not cost:
                    continue

                for a in range(4): 
                    pred_i, pred_j = np.array([i, j]) - np.array(self.cont_to_disp[a])
                    
                    # skip nodes that are out of grid
                    if (pred_i < 0 or pred_i >= self.grid_width or
                        pred_j < 0 or pred_j >= self.grid_height):
                        continue

                    pred_idx = self.sub2ind(pred_i, pred_j)
                    cost_graph[pred_idx, idx] = cost
        
        return cost_graph

    def sub2ind(self, i, j):
        """
        Convert subscripts to linear indices in row major
        """
        assert i >= 0 and i < self.grid_width
        assert j >= 0 and j < self.grid_height

        idx = i * self.grid_width + j
        return idx
    
    def ind2sub(self, idx):
        """
        Convert linear indices in row major to subscripts
        """
        assert idx >= 0 and idx < self.num_nodes

        i = idx // self.grid_width
        j = idx % self.grid_width
        return (i, j)

    def shortest_path(self):
        """
        Run shortest path algorithm to get paths from all nodes to goal
        """
        goal_i, goal_j = np.where(self.grid == self.obj_to_idx["Goal"])
        self.goal_pos = [goal_i, goal_j]
        goal_idx = self.sub2ind(goal_i, goal_j)

        cost_graph = csr_matrix(self.encode_cost())
        dist_matrix, predecessors = shortest_path(csgraph=cost_graph, method='BF',
                directed=True, indices=goal_idx, return_predecessors=True)

        self.predecessors = np.squeeze(predecessors, axis=0)

    def get_expert_controls(self):
        """
        Extracts the optimal controls from expert shortest path
        starting from agent position
        Returns empty list if no shortest path exists
        """
        path = []
        idx = self.sub2ind(*self.agent_pos)
        while idx != -9999:
            path.append(idx)
            idx = self.predecessors[idx]
        path = [self.ind2sub(idx) for idx in path]

        expert_controls = []
        for t in range(len(path) - 1):
            curr_state = path[t]
            next_state = path[t + 1]
            expert_controls.append(self.get_control_from_states(curr_state, next_state))

        return expert_controls
   
    def get_expert_control(self):
        """
        Returns the first control in expert controls
        """
        expert_controls = self.get_expert_controls()
        return expert_controls[0]

    def get_control_from_states(self, curr_state, next_state):
        """
        Get control from current state to next state
        """
        disp = tuple(np.array(next_state) - np.array(curr_state))
        return self.disp_to_cont[disp]
        
