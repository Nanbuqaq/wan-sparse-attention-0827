"""Shared, non-neural contract for event-window face diagnostics."""
import hashlib
import json
import math

GROUPS = ('reference_anchor', 'away', 'return_anchor', 'return_continuation')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_task(task):
    if task.get('schema') != 'face_return_windows_v2':
        raise ValueError('Unknown face diagnostic schema')
    if task.get('identity_threshold_adopted') is not False or task.get('formal_memory_admitted') is not False:
        raise ValueError('Windowed feature scores do not establish identity or formal memory')
    if len(task.get('video_sha256', '')) != 64 or not task.get('associations'):
        raise ValueError('Frozen media and model associations required')
    if set(task['windows']) != set(GROUPS):
        raise ValueError('All requested groups must be explicit, including missing continuation')
    for name, window in task['windows'].items():
        if window is None:
            if name != 'return_continuation':
                raise ValueError('Reference, away and return windows are required')
            continue
        if (len(window) != 2 or not all(type(x) in (int, float) and math.isfinite(x) for x in window)
                or not 0 <= window[0] < window[1]):
            raise ValueError('Invalid requested video time window')
    a, b, c = [task['windows'][key] for key in GROUPS[:3]]
    if a[1] > b[0]+1e-8 or b[1] > c[0]+1e-8:
        raise ValueError('Reference, away and return windows overlap')
    d = task['windows']['return_continuation']
    if d is not None and d[0] < c[1]-1e-8:
        raise ValueError('First return and continuation must be separate')
    expected = digest({k:v for k,v in task.items() if k not in ('task_id','associations','video')})[:24]
    if task.get('task_id') != expected:
        raise ValueError('Frozen face request changed')
    return task


def requested_samples(task):
    validate_task(task)
    return [(name, i, window[0]+(window[1]-window[0])*fraction, window[1])
            for name, window in task['windows'].items() if window is not None
            for i, fraction in enumerate((.2, .5, .8))]


def cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError('Feature shapes differ or are empty')
    if not all(math.isfinite(float(x)) for x in list(left)+list(right)):
        raise ValueError('Nonfinite face features')
    an = math.sqrt(sum(float(x)*float(x) for x in left))
    bn = math.sqrt(sum(float(x)*float(x) for x in right))
    if an <= 1e-12 or bn <= 1e-12:
        raise ValueError('Zero face feature norm')
    return max(-1., min(1., sum(float(a)*float(b) for a,b in zip(left,right))/(an*bn)))
