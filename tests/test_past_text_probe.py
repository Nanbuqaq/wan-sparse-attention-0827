import json
from pathlib import Path

import pytest

from scripts.run_official_interactive_local import restate_past_identity


def test_only_past_instruction_features_are_restated():
    root=Path(__file__).resolve().parents[1]
    segments=json.loads((root/'configs/system/memory_revisit_development.json').read_text())['scenarios'][1]['segments']
    current=segments[-1]['prompt'];past=segments[1]['prompt'];new=restate_past_identity(past,current)
    assert new.startswith(current)
    assert 'small red tin robot with a square head, a yellow triangular chest badge, and white feet' in new
    assert 'lid of the same' not in new and 'steps onto' not in new
    with pytest.raises(ValueError):restate_past_identity('no matching frozen reveal',current)
