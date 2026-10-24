"""Run a frozen JSON list of Docker benchmark conditions consecutively."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--only',nargs='+')
    args=parser.parse_args()
    spec=json.loads(args.manifest.read_text())
    root=Path(spec['output_root']);root.mkdir(exist_ok=True,parents=True)
    results=[]
    for item in spec['conditions']:
        if args.only and item['id'] not in args.only:continue
        assert re.fullmatch(r'[a-z0-9-]+',item['id'])
        record_path=root/(item['id']+'_record.json')
        if record_path.exists():
            result=json.loads(record_path.read_text())
            assert result['exit_code']==0
            results.append(result);continue
        if item.get('reuse_from'):
            out=Path(item['reuse_from'])
            rows=json.loads((out/'measurements.json').read_text())
            result={**item,'exit_code':0,'reused':True,'measurements':rows}
        else:
            out=root/item['id'];out.mkdir(exist_ok=False)
            # Pin the image ID; the displayed tag remains useful provenance.
            cmd=['docker','run','--gpus','all','--ipc=host','--shm-size=64g',
                 '--name',spec['container_prefix']+'-'+item['id']]
            for key,value in {**spec['environment'],**item.get('environment',{})}.items():
                cmd+=['-e',key+'='+value]
            mounts=[{'source':str(out),'target':'/output'},
                    {'source':spec['tools_directory'],'target':'/tools','readonly':True},
                    {'source':item['query'],'target':'/query.json','readonly':True},
                    {'source':item['msa'],'target':'/frozen-msas','readonly':True},
                    *item.get('extra_mounts',[])]
            for mount in mounts:
                text=f"type=bind,src={mount['source']},dst={mount['target']}"
                if mount.get('readonly'):text+=',readonly'
                cmd+=['--mount',text]
            command=['python3','/tools/benchmark_resources.py','--query-json=/query.json',
                     '--output-root=/output','--warmups=1','--repeats=2','--interval=0.1']
            if item.get('wrapper'):command+=['--inference-wrapper=/tools/'+item['wrapper']]
            cmd+=[item['image_id']]
            if item.get('setup_command'):
                cmd+=['bash','-c','set -e; '+shlex.join(item['setup_command'])+'; '+shlex.join(command)]
            else:cmd+=command
            started=time.monotonic()
            with (root/(item['id']+'.log')).open('w') as log:
                code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
            result={**item,'exit_code':code,'elapsed_including_setup_and_warmup_seconds':time.monotonic()-started,'reused':False}
            if (out/'measurements.json').exists():result['measurements']=json.loads((out/'measurements.json').read_text())
        result['query_sha256']=hashlib.sha256(Path(item['query']).read_bytes()).hexdigest()
        assert result['exit_code']==0,result
        rows=result['measurements']
        assert len(rows)==3 and sum(not r['warmup'] for r in rows)==2
        assert all(r['cli_returned_successfully'] and r['prediction_files']==item['expected_prediction_files'] for r in rows),item['id']
        record_path.write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        (root/'results_241026.json').write_text(json.dumps(results,indent=2)+'\n')
        print(item['id']+' COMPLETE',flush=True)
    print('BENCHMARK_CONDITIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
