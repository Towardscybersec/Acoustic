from src.universality.rule110 import evolve


def to_list(arr):
    return arr.tolist() if hasattr(arr, "tolist") else arr


def test_rule110_small_example():
    init = [0, 1, 1, 0, 1, 1, 1, 0, 0]
    expected = [
        [0, 1, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 0, 1],
    ]
    result = to_list(evolve(init, 5))
    assert result == expected
