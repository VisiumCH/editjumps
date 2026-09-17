"""The scaffolding example test that ships with the project template."""

import pytest


@pytest.mark.parametrize(
    "a, b, expected_sum",
    [
        (1, 1, 2),  # First test, 1 + 1 = 2
        (-1, -1, -2),  # Second test, (-1) + (-1) = -2
    ],
)
def test_example(a: int, b: int, expected_sum: int) -> None:
    """Test example of addition a + b == expected_sum."""
    assert a + b == expected_sum
