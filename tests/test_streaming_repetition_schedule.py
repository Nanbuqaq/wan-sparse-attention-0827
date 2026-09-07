import pytest
from scripts.profile_streaming_pipeline import build_repetition_schedule


def test_single_repetition_preserves_existing_explicit_order():
    variants=[('batch','current_stream'),('async','current_stream')]
    assert build_repetition_schedule(variants,1,8)==[(0,*v) for v in variants]


def test_repeats_are_frozen_balanced_and_non_mutating():
    variants=[('batch','current_stream'),('async','current_stream'),('async_priority','current_stream')]
    original=list(variants)
    schedule=build_repetition_schedule(variants,5,20260908)
    assert schedule==build_repetition_schedule(variants,5,20260908)
    assert len(schedule)==15 and variants==original
    for i in range(5):
        assert {row[1:] for row in schedule if row[0]==i}==set(variants)
    assert len({tuple(row[1:] for row in schedule if row[0]==i) for i in range(5)})>1


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError): build_repetition_schedule([],1,0)
    with pytest.raises(ValueError): build_repetition_schedule([('batch','device')],0,0)


def test_factorial_schedule_keeps_both_axes_separate():
    arms=[(mode,'current_stream',layout) for mode in ('batch','async_priority') for layout in ('upstream','direct_output')]
    schedule=build_repetition_schedule(arms,3,821)
    assert len(schedule)==12
    for i in range(3): assert {row[1:] for row in schedule if row[0]==i}==set(arms)
