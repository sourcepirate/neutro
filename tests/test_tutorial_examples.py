import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "example, expected",
    [
        ("examples/03_attention_tour.py", "That is attention"),
        ("examples/04_tiny_transformer_lm.py", "Generated:"),
        ("examples/05_manual_backprop.py", "Both paths use the same"),
        ("examples/06_tokenization_tour.py", "Unicode round-trip"),
        ("examples/07_lora_tour.py", "Only lora_A and lora_B"),
    ],
)
def test_tutorial_example_runs(example, expected):
    result = subprocess.run(
        [sys.executable, example],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
