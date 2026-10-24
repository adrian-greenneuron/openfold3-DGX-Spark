# Portions follow OpenFold3's DiffusionConditioning.
# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in, rollout-scoped reuse of timestep-independent diffusion calculations.

No weights, precision, model configuration or sampling settings are changed.
The cache is invalidated at every rollout and on tensor identity/version changes.
Actual cache hits are checked exactly against the original implementation.
"""
import contextvars
import functools
import hashlib
import inspect
import json
from pathlib import Path


def install(record_path):
    import torch
    from openfold3.core.model.structure.diffusion_module import SampleDiffusion, DiffusionModule
    from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
    from openfold3.core.model.layers.attention_pair_bias import DiffusionAttentionPairBias
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer

    # Fail before prediction if the relevant upstream implementation changes.
    # These are the unmodified files in pinned revision a83cb119f58a86306f9d66ca528ef2a0fffad250.
    expected={SampleDiffusion:'3b5ed9fb090d53df2b5ad16be517cfdca7f5290a6bb8274e71408a3ff465155d',
              DiffusionConditioning:'5bf5170d836c3b56cf665a9dc68bb935018ef21f779f516bc97ee0be5b1c6305',
              DiffusionAttentionPairBias:'a7622c8a315e917480586047af7fb35ed2283ade0a16bcf8b68f8f5f22f5cdee',
              DiffusionTransformer:'a2f1abff3c59a141311f1408b084f96235332067a95a35e5133a909aff9796e6'}
    for cls,digest in expected.items():
        if hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Diffusion cache is unqualified for this upstream source: '+cls.__name__)

    state=contextvars.ContextVar('of3_rollout_cache',default=None)
    record={'mechanism':'rollout-scoped invariant conditioning and bias cache',
            'precision_or_model_settings_changed':False,'rollouts':0,
            'conditioning_hits':0,'bias_hits':0,'norm_hits':0,'exact_checks':0,
            'max_bias_entries':0,'max_norm_entries':0}
    def tensor_key(x):
        if not isinstance(x,torch.Tensor):return x
        version=None if torch.is_inference(x) else x._version
        return (id(x),version,tuple(x.shape),str(x.dtype),str(x.device))
    def batch_key(batch):
        return (id(batch),tuple((k,tensor_key(v)) for k,v in sorted(batch.items())
                               if isinstance(v,torch.Tensor)))

    original_sample=SampleDiffusion.forward
    @functools.wraps(original_sample)
    def sample(self,*args,**kwargs):
        if self.training or torch.is_grad_enabled():return original_sample(self,*args,**kwargs)
        token=state.set({'conditioning':{},'bias':{},'norm':{}})
        record['rollouts']+=1
        try:return original_sample(self,*args,**kwargs)
        finally:state.reset(token)
    SampleDiffusion.forward=sample

    # Cache only the token diffusion transformer's normalised invariant pair
    # tensor. Atom encoder/decoder pair tensors depend on noisy coordinates.
    original_init=DiffusionModule.__init__
    @functools.wraps(original_init)
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        module=self.diffusion_transformer.layer_norm_z
        original_norm=module.forward
        @functools.wraps(original_norm)
        def norm(z):
            cache=state.get()
            if cache is None or module.training or torch.is_grad_enabled():return original_norm(z)
            key=tensor_key(z)
            saved=cache['norm'].get(id(module))
            if saved is None or saved['key']!=key:
                result=original_norm(z)
                cache['norm'][id(module)]={'key':key,'input':z,'output':result,'checked':False}
                record['max_norm_entries']=max(record['max_norm_entries'],len(cache['norm']))
                return result
            if not saved['checked']:
                if not torch.equal(saved['output'],original_norm(z)):
                    raise RuntimeError('Normalisation cache differs from original')
                saved['checked']=True;record['exact_checks']+=1
            record['norm_hits']+=1
            return saved['output']
        module.forward=norm
    DiffusionModule.__init__=init

    original_condition=DiffusionConditioning.forward
    @functools.wraps(original_condition)
    def condition(self,batch,t,si_input,si_trunk,zij_trunk,use_conditioning,chunk_size=None):
        cache=state.get()
        if cache is None or self.training or torch.is_grad_enabled() or not use_conditioning:
            return original_condition(self,batch,t,si_input,si_trunk,zij_trunk,use_conditioning,chunk_size)
        key=(id(self),batch_key(batch),tensor_key(si_input),tensor_key(si_trunk),
             tensor_key(zij_trunk),chunk_size,torch.get_float32_matmul_precision())
        saved=cache['conditioning'].get(id(self))
        if saved is None or saved['key']!=key:
            output=original_condition(self,batch,t,si_input,si_trunk,zij_trunk,use_conditioning,chunk_size)
            cache['conditioning'][id(self)]={'key':key,'inputs':(batch,si_input,si_trunk,zij_trunk),
                                            'pair':output[1],'checked':False}
            return output
        # This single representation depends on t and must be recomputed.
        # Only the pair representation and its transitions are invariant.
        si=torch.cat([si_trunk,si_input],dim=-1)
        si=self.linear_s(self.layer_norm_s(si))
        n=0.25*torch.log(t/self.sigma_data)
        n=self.fourier_emb(n.unsqueeze(-1))
        si=si+self.linear_n(self.layer_norm_n(n)).unsqueeze(-2)
        for layer in self.transition_s:
            si=si+layer(si,mask=batch['token_mask'],chunk_size=chunk_size)
        output=(si,saved['pair'])
        if not saved['checked']:
            reference=original_condition(self,batch,t,si_input,si_trunk,zij_trunk,use_conditioning,chunk_size)
            if not all(torch.equal(a,b) for a,b in zip(output,reference)):
                raise RuntimeError('Conditioning cache differs from original')
            saved['checked']=True;record['exact_checks']+=1
        record['conditioning_hits']+=1
        return output
    DiffusionConditioning.forward=condition

    original_bias=DiffusionAttentionPairBias._prep_bias
    @functools.wraps(original_bias)
    def bias(self,a,z,mask):
        cache=state.get()
        if cache is None or self.training or torch.is_grad_enabled():return original_bias(self,a,z,mask)
        key=(id(self),tuple(a.shape[:-2]),tensor_key(z),tensor_key(mask))
        saved=cache['bias'].get(id(self))
        if saved is None or saved['key']!=key:
            output=original_bias(self,a,z,mask)
            cache['bias'][id(self)]={'key':key,'inputs':(z,mask),'output':output,'checked':False}
            record['max_bias_entries']=max(record['max_bias_entries'],len(cache['bias']))
            return output
        if not saved['checked']:
            reference=original_bias(self,a,z,mask)
            if not all(torch.equal(a,b) for a,b in zip(saved['output'],reference)):
                raise RuntimeError('Pair-bias cache differs from original')
            saved['checked']=True;record['exact_checks']+=1
        record['bias_hits']+=1
        return saved['output']
    DiffusionAttentionPairBias._prep_bias=bias

    def save():
        if record_path is not None:Path(record_path).write_text(json.dumps(record,indent=2)+'\n')
    return save
