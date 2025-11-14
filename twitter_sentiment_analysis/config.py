import torch

class TrainingConfig:
    """Configuration for training"""
    def __init__(self):
        self.SEED = 42
        self.MODEL_NAME = "distilbert-base-uncased"
        self.MAX_LEN = 128
        self.BATCH_SIZE = 32
        self.ACCUMULATION_STEPS = 8
        self.EPOCHS = 12
        self.LEARNING_RATE = 2e-5
        self.WARMUP_PROPORTION = 0.1
        self.EARLY_STOPPING_PATIENCE = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = TrainingConfig()