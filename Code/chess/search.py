import chess


# =============================
# Evaluation cache
# =============================

_eval_cache = {}


def evaluate_board(engine, model, feature_fn):
    fen = engine.board.fen()
    if fen in _eval_cache:
        return _eval_cache[fen]

    features = feature_fn(engine.board)
    score = model.predict(features)

    _eval_cache[fen] = score
    return score


# =============================
# Move ordering
# =============================

def ordered_moves(board):
    captures = []
    quiets = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            quiets.append(move)

    return captures + quiets


# =============================
# Minimax with alpha-beta
# =============================

def minimax(engine, model, feature_fn, depth, alpha, beta, maximizing):
    board = engine.board

    # Terminal or depth limit
    if depth == 0 or board.is_game_over():
        if board.is_checkmate():
            return 1.0 if maximizing else -1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        return evaluate_board(engine, model, feature_fn)

    if maximizing:
        max_eval = -1e9
        for move in ordered_moves(board):
            board.push(move)
            eval_score = minimax(
                engine, model, feature_fn,
                depth - 1, alpha, beta, False
            )
            board.pop()

            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval

    else:
        min_eval = 1e9
        for move in ordered_moves(board):
            board.push(move)
            eval_score = minimax(
                engine, model, feature_fn,
                depth - 1, alpha, beta, True
            )
            board.pop()

            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


# =============================
# Root move selection
# =============================

def select_best_move(engine, model, feature_fn, depth=2):
    """
    Root move selection using minimax.
    depth=1 → old behavior
    depth=2 → real tactics
    """
    board = engine.board
    maximizing = board.turn == chess.WHITE

    best_move = None
    best_score = -1e9 if maximizing else 1e9

    for move in ordered_moves(board):
        board.push(move)

        score = minimax(
            engine,
            model,
            feature_fn,
            depth - 1,
            alpha=-1e9,
            beta=1e9,
            maximizing=not maximizing
        )

        board.pop()

        if maximizing and score > best_score:
            best_score = score
            best_move = move
        elif not maximizing and score < best_score:
            best_score = score
            best_move = move

    return best_move.uci()
