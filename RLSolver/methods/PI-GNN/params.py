import numpy as np

INPUT_DIM = lambda n: (np.round(np.sqrt(n) if n > 1e5 else n ** (1 / 3))).astype(int)
HIDDEN_DIM = lambda n: INPUT_DIM(n) // 2
OUTPUT_DIM = 1

DEVICE = "cpu"  # 'cpu' or 'mps'

LEARNING_RATE = 1e-4
NUM_EPOCH = int(1e5)
TOLERANCE = 1e-4
PATIENCE = int(1e3)
NUM_EVAL = 10
