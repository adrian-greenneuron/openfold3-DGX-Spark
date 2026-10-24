"""Diagnostic stage/attention timings without altering model configuration.

CUDA event intervals include stream-idle gaps; they are not isolated kernel busy
 time. Diagnostic timings must not be mixed with uninstrumented benchmarks.
"""
import argparse
from collections import defaultdict
import functools
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--capture-attention', type=Path)
    args, cli = parser.parse_known_args()
    import torch
    import openfold3.core.model.primitives.attention as attention
    from openfold3.projects.of3_all_atom.model import OpenFold3

    events = defaultdict(list)
    shapes = defaultdict(lambda: defaultdict(int))
    capture_done = set()

    def timed(name, function, capture=False):
        @functools.wraps(function)
        def wrapper(*pos, **kw):
            if capture:
                q = pos[0]
                key = str((tuple(q.shape), str(q.dtype), tuple(q.stride())))
                shapes[name][key] += 1
                # One representative real input per attention shape category.
                category = 'large' if q.shape[-2] > 128 else 'small'
                if (args.capture_attention and category not in capture_done
                        and q.numel() > 10000):
                    args.capture_attention.mkdir(parents=True, exist_ok=True)
                    def cpu(x):
                        if isinstance(x, torch.Tensor): return x.detach().cpu()
                        if isinstance(x, list): return [cpu(v) for v in x]
                        return x
                    torch.save({'args': tuple(cpu(v) for v in pos),
                                'kwargs': {k: cpu(v) for k,v in kw.items()},
                                'function': name}, args.capture_attention/(category+'.pt'))
                    capture_done.add(category)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            start = time.perf_counter()
            result = function(*pos, **kw)
            end.record()
            events[name].append((begin, end, time.perf_counter()-start))
            return result
        return wrapper

    for name in ['_attention', '_triton_evo_attn', '_deepspeed_evo_attn', '_cueq_triangle_attn']:
        if hasattr(attention,name):
            setattr(attention,name,timed(name,getattr(attention,name),capture=True))
    for name in ['run_trunk','_rollout']:
        setattr(OpenFold3,name,timed(name,getattr(OpenFold3,name)))
    original_init = OpenFold3.__init__
    @functools.wraps(original_init)
    def init(self,*pos,**kw):
        original_init(self,*pos,**kw)
        for name in ['input_embedder','template_embedder','msa_module','pairformer_stack','sample_diffusion','aux_heads']:
            module=getattr(self,name,None)
            if module is not None:
                module.forward=timed(name,module.forward)
    OpenFold3.__init__=init
    torch.cuda.init()
    sys.argv=['run_openfold',*cli]
    try:
        runpy.run_module('openfold3.run_openfold',run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None,0):raise
    torch.cuda.synchronize()
    result={}
    for name,rows in events.items():
        elapsed=[a.elapsed_time(b)/1000 for a,b,_ in rows]
        result[name]={'calls':len(rows),'cuda_stream_elapsed_seconds':sum(elapsed),
                      'host_call_seconds':sum(t for _,_,t in rows),
                      'first_call_stream_seconds':elapsed[0],
                      'remaining_stream_seconds':sum(elapsed[1:]),
                      'shapes':dict(shapes[name])}
    args.profile_output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
