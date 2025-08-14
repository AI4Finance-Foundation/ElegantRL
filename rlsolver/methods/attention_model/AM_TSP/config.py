# config.py
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_HEAD = 4
C = 10.0
SEQ_LEN = 30
NUM_TR_DATASET = 50000
NUM_TE_DATASET = 2000

NUM_EPOCHS = 100
BATCH_SIZE = 128  # Per GPU batch size
LR = 0.0003       # Fixed learning rate, no scaling with world size
GRAD_CLIP = 1.5
BETA = 0.9

USE_CUDA = True
NUM_WORKERS = 4   # Can be increased based on machine (e.g., 8-16)
SEED = 111