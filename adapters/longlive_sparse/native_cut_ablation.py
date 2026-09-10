"""Declared cut-component ablation, not a new native baseline or pure layout."""


def condition_aliases(prompts,*,block_frames=8,after_frame=48,prefix='The scene transitions. '):
    return {prompt:prompt.removeprefix(prefix) for i,prompt in enumerate(prompts)
            if i*block_frames>=after_frame and prompt.startswith(prefix)}


class NativeRopePhaseFreeze:
    def __init__(self,pipe,after_frame=48):
        self.pipe=pipe;self.after_frame=after_frame;self.last_before=None;self.frozen=None
        self.seen=set();self.events=[];self.handle=None

    def before(self,owner,values,kwargs):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        actual=float(self.pipe._dit_model.rope_temporal_offset)
        if frame<self.after_frame:
            self.last_before=actual;return
        if self.frozen is None:
            if self.last_before is None:raise RuntimeError('no observed pre-ablation RoPE phase')
            self.frozen=self.last_before
        self.pipe._dit_model.rope_temporal_offset=self.frozen
        if frame not in self.seen:
            self.events.append(dict(frame=frame,before=actual,used=self.frozen));self.seen.add(frame)

    def attach(self):self.handle=self.pipe.generator.register_forward_pre_hook(self.before,with_kwargs=True)

    def detach(self):
        if self.handle is not None:self.handle.remove();self.handle=None

    def audit(self):
        if not self.events:raise RuntimeError('RoPE phase ablation did not execute')
        return dict(frozen_after_frame=self.after_frame,phase=self.frozen,events=self.events,
                    old_KV_not_rotated=True,raw_cut_prefix_and_native_pin_not_changed=True)
