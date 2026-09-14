"""Explicit fixed-off source policy needs no CPU raw archive or selector."""


class NoSourceArchive:
    def __init__(self):
        self.archives=[]

    def before(self,*args,**kwargs):
        return None

    def after(self,*args,**kwargs):
        return None

    def audit(self):
        return dict(method='fixed_off_no_source_archive',archives=[],decisions=[],installations=[],
            archive_budget_bytes=0,raw_source_frames_or_target_frames_supplied_to_selector=False,
            future_text_or_generated_outputs_read_by_selector=False,
            scope='current-text return filter only; no past-source selection or raw CPU bank',
            ledger=dict(archive_D2H_KV_bytes=0,history_H2D_KV_bytes=0,condition_summary_D2H_bytes=0,
                CPU_archive_peak_tensor_bytes=0,archive_wall_s=0.,condition_summary_wall_including_readiness_s=0.,
                selector_CPU_wall_s=0.,history_install_wall_s=0.,temporal_rebind_wall_s=0.,evicted_archives=0))
