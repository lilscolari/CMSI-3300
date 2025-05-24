'''
Pacman Agent employing a PacNet trained in another module to
navigate perilous ghostly pellet mazes.
'''

import time
import random
import numpy as np
import torch
from torch import nn
from pathfinder import *
from queue import Queue
from constants import *
from pac_trainer import *

class PacmanAgent:
    '''
    Deep learning Pacman agent that employs PacNets trained in the pac_trainer.py
    module.
    '''

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, really just needs the model and
        its plan Queue.
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained. (Will be
        same format as Constants.MAZE)
        """

        self.model = PacNet(maze)
        self.model.load_state_dict(torch.load(f'{Constants.PARAM_PATH}', weights_only=True))

        self.model.eval()

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: String action choice from the set of legal_actions
        """

        perception = ["".join(row) for row in perception]

        vectorized_perception = PacmanMazeDataset.vectorize_maze(perception)

        outputs = self.model(vectorized_perception)

        best_move = None
        best_score = -float('inf')
        for move, _ in legal_actions:
            i = PacmanMazeDataset.move_indexes[move]
            if outputs[i] > best_score:
                best_score = outputs[i]
                best_move = move
        return best_move