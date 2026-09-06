#!/usr/bin/env python3
"""Instrument a frozen completed generation without redefining its case identity.

Generation uses the original checkout, method order, RNG and hardware lane.
Only diagnostic namespace/capture flags change. This is not a new quality trial.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess


def build(suite, expected, states, *, orchestrator_commit):
    if suite.get('formal_prompts_used') is not False:
        raise ValueError('only isolated development references are allowed')
    if {r['id'] for r in expected['cases']} != {r['id'] for r in states['cases']}:
        raise ValueError('reference matrix incomplete')
    if any(r['status'] != 'pass' for r in states['cases']):
        raise ValueError('reference cases must have passed')
    source = suite['experiment_commit']
    if any(r['case_key']['commit'] != source for r in states['cases']):
        raise ValueError('mixed reference source commits')
    result=copy.deepcopy(suite)
    result['status']='frozen_same_trajectory_capture_diagnostic'
    for case in result['cases']:
        case['complete_capture']=True
    protocol={'scope':'same_generation_case_distinct_diagnostic_namespace',
        'generation_commit':source,'orchestrator_commit':orchestrator_commit,
        'layers':[0,9,19,29],'starts':[28080,46800],'passes':5,
        'reference_cases':len(expected['cases']),'expected_captures_per_case':40,
        'required_gates':['same_initial_noise','bitwise_latents','ordered_routes','complete_capture_coverage'],
        'new_quality_trial':False,'formal_prompts_used':False}
    return result,copy.deepcopy(expected),protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-root',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(args.reference_root).resolve()
    paths=[root/'control/suite.json',root/'control/expected.json',root/'states.json']
    data=[json.loads(p.read_text()) for p in paths]
    commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    suite,expected,protocol=build(*data,orchestrator_commit=commit)
    protocol['reference_root']=str(root)
    protocol['reference_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    for name,value in [('suite',suite),('expected',expected),('protocol',protocol)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
