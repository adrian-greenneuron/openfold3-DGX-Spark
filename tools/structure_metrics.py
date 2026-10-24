"""C-alpha comparisons for sequence-matched structural analysis.

lDDT uses a 15-A inclusion radius and 0.5/1/2/4-A thresholds. Comparisons
against another prediction measure agreement, not experimental accuracy.
"""
from pathlib import Path
import shlex
import numpy as np


def read_ca(path):
    lines=Path(path).read_text().splitlines()
    i=next(i for i,line in enumerate(lines) if line.startswith('_atom_site.'))
    columns=[]
    while lines[i].startswith('_atom_site.'):
        columns.append(lines[i].strip());i+=1
    result={}
    for line in lines[i:]:
        if line.strip() in ('#','loop_') or line.startswith('_'):break
        if not line.strip():continue
        values=shlex.split(line)
        if len(values)!=len(columns):raise ValueError('Unsupported wrapped atom-site row')
        row=dict(zip(columns,values))
        if row['_atom_site.label_atom_id']!='CA' or row['_atom_site.label_seq_id'] in ('.','?'):continue
        if row.get('_atom_site.label_alt_id','.') not in ('.','A'):continue
        key=(row['_atom_site.label_asym_id'],int(row['_atom_site.label_seq_id']))
        if key in result:raise ValueError('Multiple models or duplicate C-alpha atoms')
        result[key]={'residue':row['_atom_site.label_comp_id'],
                     'xyz':np.array([float(row['_atom_site.Cartn_'+axis]) for axis in 'xyz'])}
    if not result:raise ValueError('No C-alpha coordinates')
    return result


def distances(x):
    return np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=-1))


def lddt_ca(model,reference,allowed_pairs=None):
    """Per-residue and mean directional lDDT-C-alpha on fully matched atoms."""
    model,reference=np.asarray(model),np.asarray(reference)
    if model.shape!=reference.shape:raise ValueError('Coordinates must match')
    dr=distances(reference);dm=distances(model)
    include=(dr<15)&~np.eye(len(dr),dtype=bool)
    if allowed_pairs is not None:include &= allowed_pairs
    errors=np.abs(dm-dr)
    fractions=sum((errors<threshold).astype(float) for threshold in (0.5,1.,2.,4.))/4
    counts=include.sum(axis=1)
    per=np.divide((fractions*include).sum(axis=1),counts,
                  out=np.full(len(dr),np.nan),where=counts>0)
    return {'mean':float(np.nanmean(per)),'per_residue':per.tolist(),
            'supported_residues':int(np.sum(counts>0)),
            'reference_distance_pairs':int(include.sum()//2)}


def superpose(model,reference,fit_indices=None):
    model,reference=np.asarray(model),np.asarray(reference)
    chosen=np.arange(len(model)) if fit_indices is None else np.asarray(fit_indices)
    x0=model[chosen].mean(axis=0);y0=reference[chosen].mean(axis=0)
    u,_,v=np.linalg.svd((model[chosen]-x0).T@(reference[chosen]-y0))
    correction=np.diag([1.,1.,np.linalg.det(u@v)])
    aligned=(model-x0)@(u@correction@v)+y0
    rmsd=float(np.sqrt(np.mean(np.sum((aligned[chosen]-reference[chosen])**2,axis=1))))
    return aligned,rmsd
