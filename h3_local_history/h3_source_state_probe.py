"""Read-only first-NFE state inspection after history projection and zeroing."""
import torch


def inspect_state(value,mode,expected_rows,tile_rows=4096):
    if (mode not in ['carry','zero'] or value.ndim!=3 or value.shape[:2]!=(1,expected_rows) or
        value.shape[2]<=0 or not value.is_floating_point() or tile_rows<1):
        raise ValueError('Actual H3 history input geometry/policy invalid')
    maxima=[];finite=[]
    for start in range(0,expected_rows,tile_rows):
        part=value[:,start:start+tile_rows]
        finite.append(torch.isfinite(part).all());maxima.append(part.abs().amax())
    valid=bool(torch.stack(finite).all().item()) if finite else True
    if not valid:raise ValueError('Nonfinite effective H3 history input')
    maximum=float(torch.stack(maxima).amax().item()) if maxima else 0.
    if mode=='zero' and maximum!=0.:raise ValueError('Requested zero history reached H3 with nonzero projected content')
    return dict(observed=True,scope='first NFE input to H3 transformer after projection/zeroing',mode=mode,
        rows=expected_rows,channels=value.shape[2],dtype=str(value.dtype),finite=valid,max_abs=maximum,
        all_zero=maximum==0.,signal_present=bool(expected_rows and maximum>0.),reduction_tile_rows=tile_rows,
        identity_or_memory_effect_proven=False)


class EffectiveHistoryProbe:
    def __init__(self,transformer,mode,expected_rows):
        self.mode,self.rows=mode,expected_rows;self.result=None
        self.handle=transformer.register_forward_pre_hook(self.observe,with_kwargs=True)

    def observe(self,module,args,kwargs):
        if self.result is None:
            if 'state_hidden_states' not in kwargs:raise ValueError('H3 actual projected history input missing')
            self.result=inspect_state(kwargs['state_hidden_states'],self.mode,self.rows)
        # No returned arguments: the model input is never replaced or masked here.

    def close(self):self.handle.remove()

    def summary(self):
        if self.result is None:raise ValueError('No actual H3 history input was observed')
        return self.result
