from scripts import run_shared_compiler_repeats as scheduler


def test_waits_for_utilization_window_without_relaunch_or_kill(monkeypatch):
    samples = iter(['2, 61', '2, 0'])
    sleeps = []
    monkeypatch.setattr(scheduler.subprocess, 'check_output', lambda *a, **k: next(samples))
    monkeypatch.setattr(scheduler.time, 'sleep', sleeps.append)
    rows = scheduler.wait_for_idle(1)
    assert [r['utilization'] for r in rows] == [61, 0]
    assert sleeps == [2]
