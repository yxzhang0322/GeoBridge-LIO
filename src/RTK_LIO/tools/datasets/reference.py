import math

def classify_pvt(valid_fix, carrier):
    if not valid_fix:
        return 'INVALID'
    return {0: 'SINGLE', 1: 'FLOAT', 2: 'FIXED'}.get(carrier, 'INVALID')

def pseudo_position(stamp_ns, position, now_ns, intervals):
    if stamp_ns > now_ns:
        raise ValueError('Future reference measurement')
    if any((start <= stamp_ns < end for start, end in intervals)):
        return None
    if len(position) != 3 or not all((math.isfinite(v) for v in position)):
        raise ValueError('Invalid reference position')
    return {'stamp_ns': stamp_ns, 'position': list(position), 'source': 'pseudo_rtk', 'noise_enabled': False}
