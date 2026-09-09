#!/usr/bin/env python3
"""Read-only official parser validation of exported native two-device CUPTI trace."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import traceback


BASE_SQL='FROM slice s JOIN thread_track tt ON s.track_id=tt.id JOIN thread t ON tt.utid=t.utid JOIN process p ON t.upid=p.upid'
OVERLAP_SQL='''WITH edges AS (
 SELECT s.ts AS t,CASE WHEN p.pid=100 THEN 1 ELSE 0 END AS a,CASE WHEN p.pid=101 THEN 1 ELSE 0 END AS b
 '''+BASE_SQL+''' WHERE s.category='kernel' AND p.pid IN (100,101)
 UNION ALL
 SELECT s.ts+s.dur AS t,CASE WHEN p.pid=100 THEN -1 ELSE 0 END AS a,CASE WHEN p.pid=101 THEN -1 ELSE 0 END AS b
 '''+BASE_SQL+''' WHERE s.category='kernel' AND p.pid IN (100,101)
), changes AS (SELECT t,SUM(a) AS a,SUM(b) AS b FROM edges GROUP BY t),
active AS (SELECT t,LEAD(t) OVER (ORDER BY t) AS next_t,
 SUM(a) OVER (ORDER BY t) AS a,SUM(b) OVER (ORDER BY t) AS b FROM changes)
SELECT COALESCE(SUM(next_t-t),0) AS overlap_ns FROM active WHERE a>0 AND b>0'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--parser',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',new_GPU_run=False,official_parser_validation=True)
    try:
        expected=json.loads(args.audit.read_text());assert expected['status']=='pass'
        result['parser_version']=subprocess.check_output([str(args.parser),'--version'],text=True).strip()
        for label,path in [('parser',args.parser),('trace',args.trace),('audit',args.audit)]:
            with path.open('rb') as handle:result[label+'_sha256']=hashlib.file_digest(handle,'sha256').hexdigest()
        queries=dict(services='SELECT p.pid,p.name AS process,s.category,COUNT(*) AS events,SUM(s.dur) AS duration_ns '+BASE_SQL+' GROUP BY p.pid,s.category ORDER BY p.pid,s.category',
            errors="SELECT name,severity,value FROM stats WHERE value>0 AND severity IN ('error','fatal')",overlap=OVERLAP_SQL)
        rows={}
        for name,query in queries.items():
            run=subprocess.run([str(args.parser),'query',str(args.trace),query],capture_output=True,text=True,timeout=60)
            (args.output/f'{name}.sql').write_text(query+'\n')
            (args.output/f'{name}.csv').write_text(run.stdout)
            (args.output/f'{name}.stderr.log').write_text(run.stderr)
            assert run.returncode==0,(name,run.stderr)
            rows[name]=list(csv.DictReader(io.StringIO(run.stdout)))
        assert not rows['errors']
        for device in (0,1):
            got=[r for r in rows['services'] if int(r['pid'])==100+device]
            assert sum(int(r['events']) for r in got)==expected['per_device'][str(device)]['activity_count']
            kernel=next(r for r in got if r['category']=='kernel')
            assert abs(int(kernel['duration_ns'])/1e9-expected['per_device'][str(device)]['GPU_kernel_service_sum_s'])<.001
        overlap=int(rows['overlap'][0]['overlap_ns'])/1e9
        assert abs(overlap-expected['cross_device_kernel_overlap_s'])<.001
        result.update(status='pass',services=rows['services'],parser_errors=rows['errors'],
            parsed_cross_device_kernel_overlap_s=overlap,comparison_tolerance_s=.001,
            GPU_activity_counts_match_source_audit=True,CPU_nested_scopes_are_not_additive_wall_time=True,
            not_a_repeated_speedup_experiment=True)
    except Exception:result.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
