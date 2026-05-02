import chess
import tensorflow as tf
from batch_to_tensor import fen_batch_to_tensor


model = tf.keras.models.load_model("chess_model.keras")

def evaluate(board, model):
    fen = board.fen()
    x = fen_batch_to_tensor([fen])  # (1, 8, 8, 18)
    value = model(x, training=False).numpy()[0][0]
    return value


def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float: # с точки зрения текущего игрока
    if board.is_game_over():
        if board.is_checkmate():
            return -1
        return 0

    if depth == 0:
        score = evaluate(board, model)
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
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    for move in quiet:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def best_move(board: chess.Board, depth: int) -> chess.Move | None:
    best = None
    alpha = -1

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -1, -alpha)
        board.pop()

        if score > alpha:
            alpha = score
            best = move

    return best
