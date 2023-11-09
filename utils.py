import numpy as np
import random
import torch
import os


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    MAX_SEED = np.iinfo(np.int32).max
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed_everything(seed)
    return seed