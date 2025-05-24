from maze_knowledge_base import *
from typing import Optional, Set, Tuple, List
from constants import *

class MazeAgent:

    def __init__(self, env: "Environment", perception: dict) -> None:
        self.env = env
        self.goal: tuple[int, int] = env.get_goal_loc()
        self.maze = env.get_agent_maze()
        self.kb = MazeKnowledgeBase()
        self.possible_pits: set[tuple[int, int]] = set()
        self.safe_tiles: set[tuple[int, int]] = set()
        self.pit_tiles: set[tuple[int, int]] = set()

        # Add goal and starting location to safe tiles
        self.safe_tiles.add(self.goal)
        self.safe_tiles.add(perception["loc"])

        if perception["tile"] == '.':
            # Safe tile - all adjacent are safe
            for tile in self.env.get_cardinal_locs(perception["loc"], 1):
                self.kb.tell(MazeClause([(("P", tile), False)]))
                self.safe_tiles.add(tile)

        if len(self.env.get_cardinal_locs(self.goal, 1)) == 1:
            for tile in self.env.get_cardinal_locs(self.goal, 1):
                self.kb.tell(MazeClause([(("P", tile), False)]))
                self.safe_tiles.add(tile)

    def manhattan_distance(self, loc1: tuple[int, int], loc2: tuple[int, int]) -> int:
        # Calculate Manhattan distance between two points.
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def think(self, perception: dict) -> tuple[int, int]:
        # Process perception and choose next move
        current_loc = perception["loc"]
        tile_type = perception["tile"]
        frontier = self.env.get_frontier_locs()

        # Process current tile information
        adjacent_tiles = self.env.get_cardinal_locs(current_loc, 1)
        self.safe_tiles.add(current_loc)

        if tile_type == '.':
            # Safe tile - all adjacent are safe
            for tile in adjacent_tiles:
                self.kb.tell(MazeClause([(("P", tile), False)]))
                self.safe_tiles.add(tile)

        elif tile_type.isdigit():
            warning_value = int(tile_type)
            unknown_adj = [loc for loc in adjacent_tiles if loc not in self.safe_tiles]

            if warning_value == len(unknown_adj):
                # All unknown adjacent are pits
                for tile in unknown_adj:
                    self.kb.tell(MazeClause([(("P", tile), True)]))
                    self.pit_tiles.add(tile)

            else:
                # Add to possible pits
                self.possible_pits.update(unknown_adj)

        # Update knowledge
        self.possible_pits = self.possible_pits - self.safe_tiles
        self.kb.simplify_self(self.pit_tiles, self.safe_tiles)

        # Choose next move
        next_move = self.choose_optimal_move(current_loc, frontier)
        return next_move

    def choose_optimal_move(self, current_loc: tuple[int, int], frontier: set) -> tuple[int, int]:
        # Choose optimal move based on manhattan distance to goal.
        # Direct path to goal if available and safe
        if self.goal in frontier and self.is_safe_tile(self.goal):
            return self.goal

        # Calculate current distance to goal
        current_to_goal = self.manhattan_distance(current_loc, self.goal)

        # Get all moves sorted by manhattan distance to goal
        possible_moves: list[tuple[tuple[int, int], int]] = []
        for loc in frontier:
            if loc not in self.pit_tiles or loc not in self.possible_pits:
                dist_to_goal = self.manhattan_distance(loc, self.goal)
                move_cost = self.manhattan_distance(current_loc, loc)

                # Only consider moves that don't increase distance to goal too much
                if dist_to_goal <= current_to_goal + 1:
                    # Calculate score (lower is better)
                    score = dist_to_goal * 2  # Primary factor is distance to goal

                    # Add move cost
                    score += move_cost

                    # Add risk penalty
                    if loc in self.possible_pits:
                        score += Constants.get_pit_penalty() # 20

                    # Add safety bonus
                    if self.is_safe_tile(loc):
                        score -= 10

                    possible_moves.append((loc, score))

        if possible_moves:
            # Sort by score and return best move
            return min(possible_moves, key=lambda x: x[1])[0]

        # If no good moves found, just take closest safe move to goal
        safe_moves: list[tuple[int, int]] = [loc for loc in frontier if self.is_safe_tile(loc)]
        if safe_moves:
            return min(safe_moves, key=lambda x: self.manhattan_distance(x, self.goal))

        # Last resort: closest non-pit move to goal
        fallback_moves: list[tuple[int, int]] = [loc for loc in frontier if loc not in self.pit_tiles]
        if fallback_moves:
            return min(fallback_moves, key=lambda x: self.manhattan_distance(x, self.goal))

        return current_loc

    def is_safe_tile(self, loc: tuple[int, int]) -> Optional[bool]:
        """Determine if a tile is safe, unsafe, or unknown."""  
        if loc in self.safe_tiles:
            return True
        if loc in self.pit_tiles:
            return False
        
        # Query KB
        if self.kb.ask(MazeClause([(("P", loc), False)])):
            self.safe_tiles.add(loc)
            return True
        if self.kb.ask(MazeClause([(("P", loc), True)])):
            self.pit_tiles.add(loc)
            return False

        return None

from environment import Environment