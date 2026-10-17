from collections import deque
import hashlib
import math
import os

import numpy as np


JOB = os.environ.get("EXPERIMENT192_JOB", "unsubmitted")
NODES = 6
WORLD = 12
GLOBAL_BATCH = 1536
MICROBATCH = 2
UPDATE_BUDGET = 1000
PARENT_STEP = 328
LR_START = 0.00006
LR_END = 0.000006


def learning_rate(update_index):
    fraction = min(max(update_index, 0), UPDATE_BUDGET - 1) / (UPDATE_BUDGET - 1)
    return LR_END + (LR_START - LR_END) * (1 + math.cos(math.pi * fraction)) / 2


def is_held_out(market, ticker):
    value = int.from_bytes(hashlib.sha256((market + ticker).encode("utf-8")).digest(), "big")
    return value * 10 < 1 << 256


def market_probabilities(arm, possible, weights):
    if arm == "A":
        return None
    if arm != "B" or not possible:
        raise ValueError("Expected arm A or B with feasible markets")
    probabilities = np.asarray([weights[market] for market in possible], dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("Invalid eligible stock-day weights")
    return probabilities / probabilities.sum()


class SmoothedSelection:
    def __init__(self, baseline):
        self.values = deque([float(baseline)], maxlen=5)
        self.best_smoothed = float(baseline)
        self.patience_reference = float(baseline)
        self.stale_validations = 0

    def observe(self, score):
        if not math.isfinite(score):
            raise ValueError("Nonfinite selection score")
        self.values.append(float(score))
        if len(self.values) < 5:
            return {"smoothed": None, "best": False, "stop": False, "stale_validations": 0}
        smoothed = sum(self.values) / len(self.values)
        best = smoothed > self.best_smoothed
        self.best_smoothed = max(self.best_smoothed, smoothed)
        if smoothed - self.patience_reference > 0.002:
            self.patience_reference = smoothed
            self.stale_validations = 0
        else:
            self.stale_validations += 1
        return {"smoothed": smoothed, "best": best, "stop": self.stale_validations >= 6,
                "stale_validations": self.stale_validations}

    def state(self):
        return {"values": list(self.values), "best_smoothed": self.best_smoothed,
                "patience_reference": self.patience_reference, "stale_validations": self.stale_validations}