import math
from src.ml_noise.noise_inject import inject_noise


def test_noise_deterministic():
    x = [0.0, 0.0, 0.0]
    out1 = inject_noise(x, enob=4, seed=123)
    out2 = inject_noise(x, enob=4, seed=123)
    assert out1 == out2

    out3 = inject_noise(x, enob=4, seed=124)
    assert out1 != out3
