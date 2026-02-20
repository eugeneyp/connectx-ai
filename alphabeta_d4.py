def alphabeta_agent(observation, configuration):
    """
    A Level 3 Strategic AI using Depth-Limited Minimax with Alpha-Beta Pruning.
    
    This agent evaluates the game tree to a specified depth, using Alpha-Beta
    pruning to discard branches that cannot improve the final decision. It 
    prioritizes central columns to maximize pruning efficiency.
    
    Args:
        observation: Kaggle object containing 'board' (1D list) and 'mark' (1 or 2).
        configuration: Kaggle object containing 'rows', 'columns', and 'inarow'.
        
    Returns:
        int: The selected column index (0 to configuration.columns-1).
    """
    import numpy as np
    import random

    # --- SETTINGS ---
    # Alpha-beta typically allows for Depth 4-5 within the 2s time limit.
    DEPTH = 4 

    # --- HELPER: DROP PIECE ---
    def local_drop_piece(grid, col, piece, config):
        """
        Simulates the gravity-fed 'drop' of a piece into the board.
        
        Args:
            grid (np.ndarray): The current 2D board state.
            col (int): The column index to drop the piece into.
            piece (int): The player's mark (1 or 2).
            config: Game configuration for board dimensions.
            
        Returns:
            np.ndarray: A copy of the grid with the new piece placed.
        """
        next_grid = grid.copy()
        for r in range(config.rows - 1, -1, -1):
            if next_grid[r][col] == 0:
                next_grid[r][col] = piece
                return next_grid
        return next_grid

    # --- HELPER: IS WIN ---
    def local_is_win(grid, piece, config):
        """
        Determines if the specified piece has achieved the required N-in-a-row.
        
        Args:
            grid (np.ndarray): The 2D board state.
            piece (int): The player mark to check (1 or 2).
            config: Game configuration for 'inarow' requirement.
            
        Returns:
            bool: True if a winning connection is found, False otherwise.
        """
        # Horizontal check
        for r in range(config.rows):
            for c in range(config.columns - (config.inarow - 1)):
                if all(grid[r, c+i] == piece for i in range(config.inarow)): return True
        # Vertical check
        for c in range(config.columns):
            for r in range(config.rows - (config.inarow - 1)):
                if all(grid[r+i, c] == piece for i in range(config.inarow)): return True
        # Positive Diagonal check
        for r in range(config.rows - (config.inarow - 1)):
            for c in range(config.columns - (config.inarow - 1)):
                if all(grid[r+i, c+i] == piece for i in range(config.inarow)): return True
        # Negative Diagonal check
        for r in range(config.inarow - 1, config.rows):
            for c in range(config.columns - (config.inarow - 1)):
                if all(grid[r-i, c+i] == piece for i in range(config.inarow)): return True
        return False

    # --- HELPER: HEURISTIC SCORER ---
    def local_get_score(grid, piece, config):
        """
        Evaluates the strategic value of a non-terminal board state.
        
        Args:
            grid (np.ndarray): The 2D board state.
            piece (int): The agent's mark.
            config: Game configuration for window scanning.
            
        Returns:
            float: A weighted score representing board 'goodness'.
        """
        score = 0
        enemy = 1 if piece == 2 else 2
        
        def evaluate_window(window, p, e):
            w_score = 0
            # Prioritize wins
            if window.count(p) == 4: w_score += 1000000
            # Reward setups
            elif window.count(p) == 3 and window.count(0) == 1: w_score += 100
            elif window.count(p) == 2 and window.count(0) == 2: w_score += 10
            # Penalize enemy threats heavily
            if window.count(e) == 3 and window.count(0) == 1: w_score -= 10000
            if window.count(e) == 4: w_score -= 1000000
            return w_score

        # Scan Horizontal
        for r in range(config.rows):
            for c in range(config.columns - (config.inarow-1)):
                score += evaluate_window(list(grid[r, c:c+config.inarow]), piece, enemy)
        # Scan Vertical
        for c in range(config.columns):
            for r in range(config.rows - (config.inarow-1)):
                score += evaluate_window(list(grid[r:r+config.inarow, c]), piece, enemy)
        # Scan Diagonals
        for r in range(config.rows - (config.inarow-1)):
            for c in range(config.columns - (config.inarow-1)):
                score += evaluate_window([grid[r+i, c+i] for i in range(config.inarow)], piece, enemy)
        for r in range(config.inarow-1, config.rows):
            for c in range(config.columns - (config.inarow-1)):
                score += evaluate_window([grid[r-i, c+i] for i in range(config.inarow)], piece, enemy)
        return score

    # --- CORE: ALPHA-BETA RECURSION ---
    def alphabeta(grid, depth, alpha, beta, is_maximizing, piece, config):
        """
        Executes the recursive search with pruning logic.
        
        Args:
            grid (np.ndarray): Current hypothetical board.
            depth (int): Current depth in search tree.
            alpha (float): Best score found for Maximizer.
            beta (float): Best score found for Minimizer.
            is_maximizing (bool): Whose turn it is in the simulation.
            piece (int): The agent's mark.
            config: Game configuration.
            
        Returns:
            float: The heuristic or terminal value of the branch.
        """
        enemy = 1 if piece == 2 else 2
        
        # Base Cases
        if local_is_win(grid, piece, config): return 1000000
        if local_is_win(grid, enemy, config): return -1000000
        if depth == 0: return local_get_score(grid, piece, config)
        
        valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
        if not valid_moves: return 0

        if is_maximizing:
            value = -float('inf')
            for col in valid_moves:
                temp_grid = local_drop_piece(grid, col, piece, config)
                value = max(value, alphabeta(temp_grid, depth - 1, alpha, beta, False, piece, config))
                alpha = max(alpha, value)
                if alpha >= beta: break 
            return value
        else:
            value = float('inf')
            for col in valid_moves:
                temp_grid = local_drop_piece(grid, col, enemy, config)
                value = min(value, alphabeta(temp_grid, depth - 1, alpha, beta, True, piece, config))
                beta = min(beta, value)
                if alpha >= beta: break
            return value

    # --- AGENT EXECUTION ---
    grid = np.array(observation.board).reshape(configuration.rows, configuration.columns)
    me = observation.mark
    
    # 1. Get and sort moves (Center-out heuristic)
    valid_moves = [c for c in range(configuration.columns) if observation.board[c] == 0]
    valid_moves = sorted(valid_moves, key=lambda x: abs(x - (configuration.columns // 2)))

    best_score = -float('inf')
    best_move = valid_moves[0]
    
    # 2. Root-Level Pruning: Initialize Alpha and Beta at the top
    alpha = -float('inf')
    beta = float('inf')
    
    for col in valid_moves:
        temp_grid = local_drop_piece(grid, col, me, configuration)
        # Check the score of this move
        score = alphabeta(temp_grid, DEPTH-1, alpha, beta, False, me, configuration)
        
        if score > best_score:
            best_score = score
            best_move = col
        
        # Update alpha for the next column check
        alpha = max(alpha, best_score)
            
    return int(best_move)