"""재현성용 전역 시드 고정. 신경망 보정기 main에서 --seed 지정 시 호출.
seed=None 이면 아무것도 안 함(기존 비시드 동작 유지)."""
import os, random


def set_all_seeds(seed):
    if seed is None:
        return
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 결정적 실행(속도 약간 손해, 재현성 우선)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
