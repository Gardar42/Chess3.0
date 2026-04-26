import sys
import chess
from nn_search import best_move

DEPTH = 4

board = chess.Board()


def send(msg: str) -> None:
    print(msg, flush=True)


def set_position(line: str):
    global board

    parts = line.split()
    board = chess.Board()

    if "startpos" in parts:
        if "moves" in parts:
            i = parts.index("moves") + 1
            moves = parts[i:]
        else:
            moves = []

    elif "fen" in parts:
        fen_i = parts.index("fen") + 1

        if "moves" in parts:
            moves_i = parts.index("moves")
            fen = " ".join(parts[fen_i:moves_i])
            moves = parts[moves_i + 1:]
        else:
            fen = " ".join(parts[fen_i:])
            moves = []

        board = chess.Board(fen)

    else:
        moves = []

    for m in moves:
        board.push_uci(m)


def main():
    global board

    while True:
        line = sys.stdin.readline().strip()

        if line == "uci":
            send("id name MyBot")
            send("id author You")
            send("uciok")

        elif line == "isready":
            send("readyok")

        elif line == "ucinewgame":
            board = chess.Board()

        elif line.startswith("position"):
            set_position(line)

        elif line.startswith("go"):
            move = best_move(board, DEPTH)

            if move is None:
                send("bestmove 0000")
            else:
                send(f"bestmove {move.uci()}")

        elif line == "stop":
            # игнорируем (нет time management)
            pass

        elif line == "quit":
            break


if __name__ == "__main__":
    main()