from src.ml_noise.noise_inject import enob_to_noise_std, noise_std_to_enob
import pytest
import math


def test_enob_roundtrip():
    for enob in [2, 4, 8]:
        std = enob_to_noise_std(enob)
        back = noise_std_to_enob(std)
        assert back == pytest.approx(enob, rel=1e-12)
