import pandas as pd
import numpy as np

PATH = "C:/Chess3/data/chessData.csv"

def normalize_eval(ev): # [-1,1]
    ev = max(-1000, min(1000, ev))
    return ev / 1000.0


def batch_generator(csv_path=PATH, batch_size=256, chunk_size=10000):
    fen_buffer = []
    eval_buffer = []

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk["Evaluation"] = pd.to_numeric(chunk["Evaluation"], errors="coerce")
        chunk = chunk.dropna(subset=["Evaluation"])

        for fen, ev in zip(chunk["FEN"].values, chunk["Evaluation"].values):
            fen_buffer.append(fen)
            eval_buffer.append(normalize_eval(ev))

            if len(fen_buffer) == batch_size:
                yield fen_buffer, np.array(eval_buffer, dtype=np.float32)
                fen_buffer = []
                eval_buffer = []