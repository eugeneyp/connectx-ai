import time

# --- PERSISTENT GLOBAL MEMORY ---
tt = {} 

def aspwin_agent(observation, configuration):
    """
    Elite Connect 4 Agent: Pure Minimax Edition.
    Cures the Negamax Asymmetry Trap to properly evaluate defensive threats.
    """
    start_time = time.time()
    rows, cols = configuration.rows, configuration.columns
    me = observation.mark
    
    # --- 1. SEARCH PARAMETERS ---
    MAX_DEPTH = 21
    SOFT_TIME_LIMIT = 1.0   
    HARD_TIME_LIMIT = 1.8   
    ASPIRATION_WINDOW = 400 

    stats = {"nodes": 0}

    # --- 2. SYNCHRONIZED WIN MASKS ---
    def generate_win_masks():
        masks = []
        for r in range(rows):
            for c in range(cols - 3):
                m = (1<<(c*7+r)) | (1<<((c+1)*7+r)) | (1<<((c+2)*7+r)) | (1<<((c+3)*7+r))
                masks.append(m)
        for c in range(cols):
            for r in range(rows - 3):
                m = (1<<(c*7+r)) | (1<<(c*7+r+1)) | (1<<(c*7+r+2)) | (1<<(c*7+r+3))
                masks.append(m)
        for r in range(rows - 3):
            for c in range(cols - 3):
                m = (1<<(c*7+r)) | (1<<((c+1)*7+r+1)) | (1<<((c+2)*7+r+2)) | (1<<((c+3)*7+r+3))
                masks.append(m)
        for r in range(3, rows):
            for c in range(cols - 3):
                m = (1<<(c*7+r)) | (1<<((c+1)*7+r-1)) | (1<<((c+2)*7+r-2)) | (1<<((c+3)*7+r-3))
                masks.append(m)
        return masks

    WIN_MASKS = generate_win_masks()

    # --- 3. BITBOARD UTILITIES ---
    def get_bitboards(board):
        my_bits, en_bits = 0, 0
        for r in range(rows):
            for c in range(cols):
                piece = board[r * cols + c]
                if piece == 0: continue
                idx = c * 7 + (rows - 1 - r)
                if piece == me: my_bits |= (1 << idx)
                else: en_bits |= (1 << idx)
        return my_bits, en_bits

    def is_win(pos):
        for s in [7, 1, 6, 8]:
            m = pos & (pos >> s)
            if m & (m >> (2 * s)): return True
        return False

    # --- 4. THE MIRROR HEURISTIC ---
    def get_heuristic(my_p, en_p, mask):
        """
        ALWAYS evaluated from 'My' perspective. No more flipping!
        """
        zeros = ~mask
        score = 0
        
        for m in WIN_MASKS:
            p_c = (my_p & m).bit_count()
            e_c = (en_p & m).bit_count()
            z_c = (zeros & m).bit_count()

            if p_c == 4: score += 1000000
            elif p_c == 3 and z_c == 1: score += 100
            elif p_c == 2 and z_c == 2: score += 10
            
            if e_c == 3 and z_c == 1: score -= 10000
            elif e_c == 4: score -= 1000000
        return score

    # --- 5. PURE MINIMAX SEARCH ENGINE ---
    def alphabeta(my_bits, en_bits, mask, depth, alpha, beta, is_maximizing):
        stats["nodes"] += 1
        key = (my_bits, en_bits)
        
        entry = tt.get(key)
        if entry and entry[0] >= depth:
            v, f, m = entry[1], entry[2], entry[3]
            if f == 0: return v
            elif f == 1: alpha = max(alpha, v) # Lower bound
            elif f == 2: beta = min(beta, v)   # Upper bound
            if alpha >= beta: return v

        # Terminal checks based on who just moved
        if is_maximizing:
            # Enemy just moved, check if they won
            if is_win(en_bits): return -1000000 - depth
        else:
            # I just moved, check if I won
            if is_win(my_bits): return 1000000 + depth

        if depth == 0: return get_heuristic(my_bits, en_bits, mask)

        valid_moves = [c for c in range(cols) if not (mask & (1 << (c * 7 + 5)))]
        if not valid_moves: return 0
        
        tt_m = entry[3] if entry else None
        valid_moves.sort(key=lambda x: (x != tt_m, abs(x - 3)))

        best_m = valid_moves[0]
        
        if is_maximizing:
            best_v = -float('inf')
            alpha_orig = alpha
            for col in valid_moves:
                new_bit = (mask + (1 << (col * 7))) & (0x7F << (col * 7))
                v = alphabeta(my_bits | new_bit, en_bits, mask | new_bit, depth - 1, alpha, beta, False)
                if v > best_v:
                    best_v, best_m = v, col
                alpha = max(alpha, best_v)
                if alpha >= beta: break
                if time.time() - start_time > HARD_TIME_LIMIT: break
            
            flag = 0 if alpha_orig < best_v < beta else (1 if best_v >= beta else 2)
            tt[key] = (depth, best_v, flag, best_m)
            return best_v
            
        else:
            best_v = float('inf')
            beta_orig = beta
            for col in valid_moves:
                new_bit = (mask + (1 << (col * 7))) & (0x7F << (col * 7))
                v = alphabeta(my_bits, en_bits | new_bit, mask | new_bit, depth - 1, alpha, beta, True)
                if v < best_v:
                    best_v, best_m = v, col
                beta = min(beta, best_v)
                if alpha >= beta: break
                if time.time() - start_time > HARD_TIME_LIMIT: break
                
            flag = 0 if alpha < best_v < beta_orig else (2 if best_v <= alpha else 1)
            tt[key] = (depth, best_v, flag, best_m)
            return best_v

    # --- 6. EXECUTION ---
    my_bits, en_bits = get_bitboards(observation.board)
    total_mask = my_bits | en_bits
    best_move, last_score, max_d = 3, 0, 0

    for d in range(1, MAX_DEPTH + 1):
        if time.time() - start_time > SOFT_TIME_LIMIT: break
        
        if d < 4:
            alpha, beta = -float('inf'), float('inf')
        else:
            alpha = last_score - ASPIRATION_WINDOW
            beta = last_score + ASPIRATION_WINDOW

        score = alphabeta(my_bits, en_bits, total_mask, d, alpha, beta, True)
        
        if score <= alpha or score >= beta:
            score = alphabeta(my_bits, en_bits, total_mask, d, -float('inf'), float('inf'), True)
        
        entry = tt.get((my_bits, en_bits))
        if entry:
            best_move, last_score = entry[3], entry[1]
            max_d = d
            if last_score >= 1000000: break

    elapsed = time.time() - start_time
    move_num = 42 - observation.board.count(0)
    print(f"Move: {move_num:2} | Depth: {max_d:2} | Score: {last_score:8} | Time: {elapsed:.3f}s")
    
    return int(best_move)