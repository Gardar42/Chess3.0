import chess
from engines.minimax.evaluator import evaluate

INF = 10000000


def negamax(board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0) -> float: # с точки зрения текущего игрока
    if board.is_game_over():
        if board.is_checkmate():
            return -INF + ply
        return 0

    if depth == 0:
        score = evaluate(board)
        return score if board.turn == chess.WHITE else -score

    captures = []
    quiet = []
    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            quiet.append(move)

    for move in captures:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    for move in quiet:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def best_move(board: chess.Board, depth: int) -> chess.Move | None:
    best = None
    alpha = -INF

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -INF, -alpha)
        board.pop()

        if score > alpha:
            alpha = score
            best = move

    return best



# fen = "7k/4Q3/8/6K1/8/8/8/8 w - - 0 1"
# board = chess.Board(fen)
# print(best_move(board, 5))

# fen = "8/5r2/8/1r6/4N2k/1P6/PKP5/8 w - - 0 1"
# board = chess.Board(fen)
# print(best_move(board, 5))

