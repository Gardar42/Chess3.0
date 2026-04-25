import numpy as np

PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}


def fen_batch_to_tensor(fen_batch):
    batch_size = len(fen_batch)
    tensor = np.zeros((batch_size, 8, 8, 18), dtype=np.float32)

    for i, fen in enumerate(fen_batch):
        parts = fen.split()
        board_str, side, castling, enpassant = parts[:4]

        # фигуры (12 каналов)
        rows = board_str.split('/')
        for r in range(8):
            c = 0
            for char in rows[r]:
                if char.isdigit():
                    c += int(char)
                else:
                    ch = PIECE_TO_CHANNEL[char]
                    tensor[i, r, c, ch] = 1.0
                    c += 1

        #side to move (1 канал)
        if side == 'w':
            tensor[i, :, :, 12] = 1.0

        #рокировки (4 канала)
        if 'K' in castling:
            tensor[i, :, :, 13] = 1.0
        if 'Q' in castling:
            tensor[i, :, :, 14] = 1.0
        if 'k' in castling:
            tensor[i, :, :, 15] = 1.0
        if 'q' in castling:
            tensor[i, :, :, 16] = 1.0

        #en passant (1 канал)
        if enpassant != '-':
            file = ord(enpassant[0]) - ord('a')
            rank = 8 - int(enpassant[1])
            tensor[i, rank, file, 17] = 1.0

    return tensor