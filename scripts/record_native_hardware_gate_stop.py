#!/usr/bin/env python3
"""Do not turn an unrun conditional video stage into missing or success."""
import argparse
import json
from pathlib import Path

from scripts.collect_native_episode_batch import DESIGN,MODES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    rows=[dict(scenario=scenario,seed=seed,mode=mode,status='negative',execution='not_launched',
        reason='hardware numerical/backend gate failed; no video promotion') for scenario,seed in DESIGN for mode in MODES]
    with (args.root/'conditional_full_stage_stop.json').open('x') as handle:
        json.dump(dict(status='negative',cases=rows,missing_unaccounted=0,not_a_method_quality_result=True),handle,indent=2);handle.write('\n')


if __name__=='__main__':main()
