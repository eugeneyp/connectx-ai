import numpy as np
import time
import random

# --- PERSISTENT GLOBAL MEMORY ---
tt = {} 

def bitboard_agent(observation, configuration):
    """
    Balanced Bitboard Agent: High NPS + Targeted Strategic Heuristics.
    Uses bitwise 'threat' detection instead of counting every sequence.
    """
    start_time = time.time()
    
    # --- 1. CONFIGURATION ---
    rows, cols = configuration.rows, configuration.columns
    me = observation.mark
    enemy = 1 if me == 2 else 2
    
    MAX_DEPTH = 21
    SOFT_TIME_LIMIT = 1.0
    HARD_TIME_LIMIT = 1.8
    
    stats = {"nodes_evaluated": 0, "nodes_pruned": 0, "tt_hits": 0}

    # --- 2. BITBOARD UTILITIES ---
    def get_bitboards(board, player):
        position, mask = 0, 0
        for c in range(cols):
            for r in range(rows):
                index = c * 7 + (rows - 1 - r)
                piece = board[r * cols + c]
                if piece != 0:
                    mask |= (1 << index)
                    if piece == player:
                        position |= (1 << index)
        return position, mask

    def is_win(pos):
        """Standard 4-in-a-row bitboard check."""
        m = pos & (pos >> 7)
        if m & (m >> 14): return True
        m = pos & (pos >> 1)
        if m & (m >> 2): return True
        m = pos & (pos >> 6)
        if m & (m >> 12): return True
        m = pos & (pos >> 8)
        if m & (m >> 16): return True
        return False

    def count_open_threes(p, mask):
        """
        Detects playable XXX_ and X_XX patterns using fast bit_count().
        """
        empty = ~mask
        def check_dir(p, s):
            # Pattern: 3-in-a-row with an empty space at either end
            # We look for 3 consecutive bits and AND with the empty mask shifted
            # This detects: [X X X .] and [. X X X]
            v1 = (p & (p >> s) & (p >> (2*s))) & (empty >> (3*s))
            v2 = (p & (p << s) & (p << (2*s))) & (empty << (3*s))
            
            # Pattern: Gapped threats like [X X . X] and [X . X X]
            g1 = (p & (p >> s) & (p >> (3*s))) & (empty >> (2*s))
            g2 = (p & (p >> (2*s)) & (p >> (3*s))) & (empty >> s)
            
            # bit_count() is a C-optimized call available in Python 3.10+
            return (v1 | v2 | g1 | g2).bit_count()

        return check_dir(p, 7) + check_dir(p, 1) + check_dir(p, 6) + check_dir(p, 8)

    def get_heuristic(pos, mask, is_me):
        if is_me:
            my_pos, opp_pos = pos, mask ^ pos
        else:
            my_pos, opp_pos = mask ^ pos, pos

        score = 0
        
        # Only count 'Live' threats (ones that can actually become wins)
        my_3 = count_open_threes(my_pos, mask)
        opp_3 = count_open_threes(opp_pos, mask)

        score += my_3 * 100
        score -= opp_3 * 10000 
        
        # Center Column Control (Hex mask for Column 3)
        center_mask = 0x1FC000000
        score += bin(my_pos & center_mask).count('1') * 5
        score -= bin(opp_pos & center_mask).count('1') * 5
        
        return score

    # --- 3. RECURSIVE NEGAMAX ---
    def alphabeta(pos, mask, depth, alpha, beta, is_me):
        stats["nodes_evaluated"] += 1
        key = (pos, mask)
        
        entry = tt.get(key)
        if entry and entry[0] >= depth:
            v, f, m = entry[1], entry[2], entry[3]
            stats["tt_hits"] += 1
            if f == 0: return v
            elif f == 1: alpha = max(alpha, v)
            elif f == 2: beta = min(beta, v)
            if alpha >= beta: return v

        enemy_pos = mask ^ pos
        # Terminal Check: Did the person who just moved win?
        if is_win(enemy_pos):
            return -1000000 - depth

        if depth == 0:
            return get_heuristic(pos, mask, is_me)

        moves = [c for c in range(cols) if not (mask & (1 << (c * 7 + 5)))]
        if not moves: return 0
        
        tt_move = entry[3] if entry else None
        moves.sort(key=lambda x: (x != tt_move, abs(x - (cols // 2))))

        best_val = -float('inf')
        best_move_found = moves[0]
        alpha_orig = alpha

        for i, col in enumerate(moves):
            new_move_bit = (mask + (1 << (col * 7))) & (0x7F << (col * 7))
            
            # Recurse: Flip player and negate score
            val = -alphabeta(enemy_pos, mask | new_move_bit, depth - 1, -beta, -alpha, not is_me)

            if val > best_val:
                best_val, best_move_found = val, col
            alpha = max(alpha, best_val)
            
            if alpha >= beta:
                stats["nodes_pruned"] += (len(moves) - (i + 1))
                break
            if time.time() - start_time > HARD_TIME_LIMIT: break

        flag = 0 if alpha_orig < best_val < beta else (1 if best_val >= beta else 2)
        tt[key] = (depth, best_val, flag, best_move_found)
        return best_val

    # --- 4. EXECUTION ---
    current_pos, current_mask = get_bitboards(observation.board, me)
    valid_cols = [c for c in range(cols) if not (current_mask & (1 << (c * 7 + 5)))]
    if not valid_cols: return 0 

    # Find center-most available move as default
    best_move = sorted(valid_cols, key=lambda x: abs(x - (cols // 2)))[0]
    final_score = 0
    max_d = 0

    for d in range(1, MAX_DEPTH + 1):
        if time.time() - start_time > SOFT_TIME_LIMIT: break
        
        max_d = d
        score = alphabeta(current_pos, current_mask, d, -float('inf'), float('inf'), True)
        
        entry = tt.get((current_pos, current_mask))
        if entry and time.time() - start_time < HARD_TIME_LIMIT:
            best_move = entry[3]
            final_score = entry[1]
            if final_score >= 1000000: break

    # --- 5. FINAL REPORTING ---
    # elapsed = time.time() - start_time
    # nps = stats["nodes_evaluated"] / (elapsed if elapsed > 0 else 0.001)
    # print(f"Depth: {max_d} | Score: {final_score} | Time: {elapsed:.3f}s | NPS: {nps:.0f} | TT Hits: {stats['tt_hits']}")
    
    return int(best_move)