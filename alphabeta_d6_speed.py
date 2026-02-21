def alphabeta_d6_speed_agent(observation, configuration):
    """
    A Level 4 Strategic AI using Depth-Limited Minimax with Alpha-Beta Pruning.
    
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
    import time

    # --- SETTINGS ---
    start_time = time.time()
    # We use a dictionary for counters so inner functions can modify them
    # stats = {"nodes_evaluated": 0, "nodes_pruned": 0}
    DEPTH = 6 

    # --- HELPER: IS WIN ---
    def is_win_local(grid, row, col, piece, config):
        # 1. Cache configuration values
        n = config.inarow
        rows = config.rows
        cols = config.columns
        target = (piece,) * n
        
        # 2. VERTICAL (Check only downwards)
        # We only need to check down because pieces are placed from bottom-up
        if row <= rows - n:
            if tuple(grid[row:row+n, col]) == target:
                return True
    
        # 3. HORIZONTAL
        # Range is (n-1) steps left and (n-1) steps right
        c_start, c_end = max(0, col - (n - 1)), min(cols, col + n)
        row_segment = tuple(grid[row, c_start:c_end])
        if len(row_segment) >= n:
            for i in range(len(row_segment) - (n - 1)):
                if row_segment[i:i+n] == target: return True
    
        # 4. POSITIVE DIAGONAL (/)
        pos_diag = []
        for i in range(-(n - 1), n):
            r, c = row - i, col + i
            if 0 <= r < rows and 0 <= c < cols:
                pos_diag.append(grid[r, c])
        if len(pos_diag) >= n:
            pos_diag = tuple(pos_diag)
            for i in range(len(pos_diag) - (n - 1)):
                if pos_diag[i:i+n] == target: return True
    
        # 5. NEGATIVE DIAGONAL (\)
        neg_diag = []
        for i in range(-(n - 1), n):
            r, c = row + i, col + i
            if 0 <= r < rows and 0 <= c < cols:
                neg_diag.append(grid[r, c])
        if len(neg_diag) >= n:
            neg_diag = tuple(neg_diag)
            for i in range(len(neg_diag) - (n - 1)):
                if neg_diag[i:i+n] == target: return True
    
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

        # cache for faster access
        rows = config.rows
        cols = config.columns
        n = config.inarow
        
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
        for r in range(rows):
            row_list = list(grid[r, :]) # Convert ONLY ONCE per row
            for c in range(cols - (n-1)):
                # Slicing a list is much faster than slicing an array + list() conversion
                score += evaluate_window(row_list[c:c+n], piece, enemy)
        # Scan Vertical
        for c in range(cols):
            col_list = list(grid[:, c]) # Convert ONLY ONCE per column
            for r in range(rows - (n-1)):
                score += evaluate_window(col_list[r:r+n], piece, enemy)
        # Scan Diagonals
        for r in range(rows - (n-1)):
            for c in range(cols - (n-1)):
                score += evaluate_window([grid[r+i, c+i] for i in range(n)], piece, enemy)
        for r in range(n-1, rows):
            for c in range(cols - (n-1)):
                score += evaluate_window([grid[r-i, c+i] for i in range(n)], piece, enemy)
        return score

    # --- CORE: ALPHA-BETA RECURSION ---
    def alphabeta(grid, depth, alpha, beta, is_maximizing, piece, config, last_row=None, last_col=None):
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
            last_row: the row where the last piece landed.
            last_col: the col where the last piece landed.
            
        Returns:
            float: The heuristic or terminal value of the branch.
        """
        # stats["nodes_evaluated"] += 1
        enemy = 1 if piece == 2 else 2
        
        # Check if the last move resulted in a win
        if last_row is not None:
            # Who made the last move? 
            # If it's currently maximizing turn, the previous (minimizing) move was the enemy.
            prev_piece = enemy if is_maximizing else piece
            if is_win_local(grid, last_row, last_col, prev_piece, config):
                # Weighting by depth ensures we pick the FASTEST win
                return (1000000 + depth) if prev_piece == piece else (-1000000 - depth)
        # Reached the end node of the tree search, return heuristics score
        if depth == 0: return local_get_score(grid, piece, config)
        
        valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
        if not valid_moves: return 0
        # Use the "Static" Center-Out Sort to maximize alpha-beta pruning
        valid_moves = sorted(valid_moves, key=lambda x: abs(x - (config.columns // 2)))

        if is_maximizing:
            value = -float('inf')
            for i, col in enumerate(valid_moves):
                # Find the row for this drop
                row = next(r for r in range(config.rows-1, -1, -1) if grid[r][col] == 0)
                grid[row][col] = piece # Drop
                
                value = max(value, alphabeta(grid, depth - 1, alpha, beta, False, piece, config, row, col))
                grid[row][col] = 0     # Undo the drop
                
                alpha = max(alpha, value)
                if alpha >= beta: 
                    # Increment pruning count for remaining moves in this branch
                    #stats["nodes_pruned"] += (len(valid_moves) - (i + 1))
                    break 
            return value
        else:
            value = float('inf')
            for i, col in enumerate(valid_moves):
                # Find the row for this drop
                row = next(r for r in range(config.rows-1, -1, -1) if grid[r][col] == 0)
                grid[row][col] = enemy # Drop
                
                value = min(value, alphabeta(grid, depth - 1, alpha, beta, True, piece, config, row, col))
                grid[row][col] = 0     # Undo the drop
                
                beta = min(beta, value)
                if alpha >= beta: 
                    #stats["nodes_pruned"] += (len(valid_moves) - (i + 1))
                    break
            return value

    # --- AGENT EXECUTION ---
    grid = np.array(observation.board).reshape(configuration.rows, configuration.columns)
    me = observation.mark

    # 1. Get and sort moves (Center-out heuristic)
    valid_moves = [c for c in range(configuration.columns) if observation.board[c] == 0]
    valid_moves = sorted(valid_moves, key=lambda x: abs(x - (configuration.columns // 2)))

    # # 1. Killer Sort the moves to prioritize better ones first to optimize alpha-beta pruning
    # # 1a. Get valid moves
    # valid_moves = [c for c in range(configuration.columns) if observation.board[c] == 0]
    # # 1b. Define the Priority Function (Killer Heuristic)
    # def move_priority(col):
    #     enemy = 1 if me == 2 else 2
    #     # Priority 1: Winning move (Highest)
    #     if local_is_win(local_drop_piece(grid, col, me, configuration), me, configuration):
    #         return 100
    #     # Priority 2: Blocking move (Critical)
    #     if local_is_win(local_drop_piece(grid, col, enemy, configuration), enemy, configuration):
    #         return 50
    #     # Priority 3: Center-out (Standard strategy)
    #     return -abs(col - (configuration.columns // 2))
    # # 1c. Sort moves by priority (Highest priority first)
    # valid_moves = sorted(valid_moves, key=move_priority, reverse=True)

    best_score = -float('inf')
    best_move = valid_moves[0]
    
    # 2. Root-Level Pruning: Initialize Alpha and Beta at the top
    alpha = -float('inf')
    beta = float('inf')
    
    for col in valid_moves:
        # Find the row manually of where the piece dropped to
        row = next(r for r in range(configuration.rows-1, -1, -1) if grid[r][col] == 0)
        grid[row][col] = me
        # Check the score of this move
        score = alphabeta(grid, DEPTH-1, alpha, beta, False, me, configuration, row, col)
        grid[row][col] = 0 # Undo
        
        if score > best_score:
            best_score = score
            best_move = col
        
        # Update alpha for the next column check
        alpha = max(alpha, best_score)

    # --- REPORTING ---
    # elapsed = time.time() - start_time
    # total_possible = stats["nodes_evaluated"] + stats["nodes_pruned"]
    # prune_rate = (stats["nodes_pruned"] / total_possible) * 100 if total_possible > 0 else 0
    # nps = stats["nodes_evaluated"] / elapsed if elapsed > 0 else 0
    # print(f"Turn Time: {elapsed:.3f}s | NPS: {nps:.0f} | Nodes Evaluated: {stats['nodes_evaluated']} | "
    #       f"Pruned: {stats['nodes_pruned']} ({prune_rate:.1f}%)")
    
    return int(best_move)