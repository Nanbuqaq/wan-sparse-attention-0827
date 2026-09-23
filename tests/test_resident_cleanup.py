import ast
from pathlib import Path

def test_resident_runner_has_explicit_codec_cleanup_after_each_case():
 s=Path('adapters/longlive_sparse/resident_output_experiment.py').read_text()
 assert 'gc.collect()' in s and 'torch.cuda.empty_cache()' in s
 assert s.index('gc.collect()')>s.index('del pipeline,latent,cpu,sink')
