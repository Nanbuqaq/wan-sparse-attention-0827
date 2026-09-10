"""Explicit GPU identity policy for homogeneous native inference batches."""


def validate_hardware_names(names,*,required='H200',allow_h800=False,expected_count):
    if len(names)!=expected_count or not names:
        raise ValueError('assigned GPU count differs from the frozen batch')
    if len(set(names))!=1:
        raise ValueError('mixed GPU models cannot share a native paired batch')
    if allow_h800 and required!='H200':
        raise ValueError('H800 opt-in is only defined for the H200/H800 Hopper policy')
    accepted=(required,'H800') if allow_h800 else (required,)
    if not all(any(prefix in name for prefix in accepted) for name in names):
        raise ValueError('assigned GPU model is outside the explicit accepted set: '+repr(names))
    return dict(actual_model=names[0],device_count=len(names),accepted_model_substrings=list(accepted),
        homogeneous_model=True,performance_must_be_grouped_by_actual_model=True,
        H800_is_not_labeled_H200=True)
