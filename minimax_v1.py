def minimax_agent(observation, configuration):
    """
    A Level 2 Strategic AI using Depth-Limited Minimax.
    Contains all necessary logic to be fully self-contained for Kaggle.
    """
    import numpy as np
    import random

    # --- SETTINGS ---
    DEPTH = 3 # Lookahead depth
    
    # --- HELPER: DROP PIECE ---
    def local_drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for r in range(config.rows - 1, -1, -1):
            if next_grid[r][col] == 0:
                next_grid[r][col] = piece
                return next_grid
        return next_grid

    # --- HELPER: IS WIN ---
    def local_is_win(grid, piece, config):
        # Horizontal
        for r in range(config.rows):
            for c in range(config.columns - (config.inarow - 1)):
                if all(grid[r, c+i] == piece for i in range(config.inarow)): return True
        # Vertical
        for c in range(config.columns):
            for r in range(config.rows - (config.inarow - 1)):
                if all(grid[r+i, c] == piece for i in range(config.inarow)): return True
        # Diagonals
        for r in range(config.rows - (config.inarow - 1)):
            for c in range(config.columns - (config.inarow - 1)):
                if all(grid[r+i, c+i] == piece for i in range(config.inarow)): return True
        for r in range(config.inarow - 1, config.rows):
            for c in range(config.columns - (config.inarow - 1)):
                if all(grid[r-i, c+i] == piece for i in range(config.inarow)): return True
        return False

    # --- HELPER: HEURISTIC SCORER ---
    def local_get_score(grid, piece, config):
        score = 0
        enemy = 1 if piece == 2 else 2
        
        def evaluate_window(window, p, e):
            w_score = 0
            if window.count(p) == 4: w_score += 1000000
            elif window.count(p) == 3 and window.count(0) == 1: w_score += 100
            elif window.count(p) == 2 and window.count(0) == 2: w_score += 10
            if window.count(e) == 3 and window.count(0) == 1: w_score -= 10000
            if window.count(e) == 4: w_score -= 1000000
            return w_score

        # Scan Horizontal, Vertical, Diagonals (Simplified for brevity)
        for r in range(config.rows):
            for c in range(config.columns - (config.inarow-1)):
                score += evaluate_window(list(grid[r, c:c+config.inarow]), piece, enemy)
        for c in range(config.columns):
            for r in range(config.rows - (config.inarow-1)):
                score += evaluate_window(list(grid[r:r+config.inarow, c]), piece, enemy)
        for r in range(config.rows - (config.inarow-1)):
            for c in range(config.columns - (config.inarow-1)):
                score += evaluate_window([grid[r+i, c+i] for i in range(config.inarow)], piece, enemy)
        for r in range(config.inarow-1, config.rows):
            for c in range(config.columns - (config.inarow-1)):
                score += evaluate_window([grid[r-i, c+i] for i in range(config.inarow)], piece, enemy)
        return score

    # --- CORE: RECURSIVE MINIMAX ---
    def local_minimax(grid, depth, is_maximizing, piece, config):
        enemy = 1 if piece == 2 else 2
        
        if local_is_win(grid, piece, config): return 1000000
        if local_is_win(grid, enemy, config): return -1000000
        if depth == 0: return local_get_score(grid, piece, config)
        
        valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
        if not valid_moves: return 0

        if is_maximizing:
            best_score = -float('inf')
            for col in valid_moves:
                temp_grid = local_drop_piece(grid, col, piece, config)
                score = local_minimax(temp_grid, depth - 1, False, piece, config)
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for col in valid_moves:
                temp_grid = local_drop_piece(grid, col, enemy, config)
                score = local_minimax(temp_grid, depth - 1, True, piece, config)
                best_score = min(score, best_score)
            return best_score

    # --- AGENT EXECUTION ---
    grid = np.array(observation.board).reshape(configuration.rows, configuration.columns)
    me = observation.mark
    valid_moves = [c for c in range(configuration.columns) if observation.board[c] == 0]
    
    best_score = -float('inf')
    best_move = random.choice(valid_moves)
    
    for col in valid_moves:
        temp_grid = local_drop_piece(grid, col, me, configuration)
        score = local_minimax(temp_grid, DEPTH-1, False, me, configuration)
        if score > best_score:
            best_score = score
            best_move = col
            
    return int(best_move)