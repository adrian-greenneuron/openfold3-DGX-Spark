# MCL1: local fold agreement and assembly variation

24 October 2026

The retained BF16 fast-path and FP32 verification predictions have **99.1–100% intrachain lDDT-Cα agreement** across the five matched MCL1 samples. Their individual protein chains are therefore very similar locally, even when the whole-complex RMSD is large. Interchain contacts can still differ substantially; those differences concern the assembly and should not be dismissed as a metric artefact.

## Comparison

Both paths used the original `of3_ft3_v1.pt` checkpoint, the same saved-MSA query and seed 42. The FP32 verification path also enabled deterministic PyTorch operations and a deterministic cuBLAS workspace. This is a retrospective comparison of two implementation modes, **not a precision-only ablation or an experimental accuracy test**.

All 632 protein Cα atoms were matched. The four equivalent MCL1 chains were assigned by the lowest whole-complex Cα RMSD over all 24 permutations. Each sample was compared with the same sample index; samples were not chosen to maximise agreement.

| Sample | BF16 vs FP32: local agreement | BF16 vs FP32: interchain contact F1 | BF16 repeat: local agreement | BF16 repeat: interchain contact F1 |
|---|---:|---:|---:|---:|
| 1 | 99.11% | 6.25% | 99.99% | 84.73% |
| 2 | 100.00% | 100.00% | 100.00% | 100.00% |
| 3 | 100.00% | 97.03% | 99.76% | 16.93% |
| 4 | 99.82% | 52.23% | 100.00% | 99.67% |
| 5 | 99.28% | 84.56% | 99.62% | 91.26% |

Higher denotes greater agreement, not greater accuracy. Local agreement is the symmetric mean of directional **intrachain lDDT-Cα**: distances to Cα neighbours within 15 Å are compared at tolerance thresholds of 0.5, 1, 2 and 4 Å, and scores are averaged over residues. Both directions are averaged because neither prediction is an experimental reference. See the [original lDDT method](https://doi.org/10.1093/bioinformatics/btt473).

Interchain contact F1 compares sets of Cα pairs closer than 8 Å, excluding pairs within the same chain. It is a simple contact-agreement measure, **not DockQ**, and does not assess side chains or ligand poses. It distinguishes stable individual folds from altered chain packing.

Whole-complex Cα RMSD ranged from **0.03–8.96 Å** for the two precision/verification paths and **0.02–10.06 Å** between two BF16 fast-path repeats. After aligning each chain separately, BF16-versus-FP32 RMSDs were only **0.02–0.47 Å**. The lower repeatability of some fast-path assemblies is consistent with the earlier numerical investigation; it does not identify the experimentally correct assembly.

These are five paired samples from one artificial example query, not a population-level result. No experimental structure has been validated for the complete supplied combination of four MCL1 copies, three ATP molecules and the specified other ligand. The [raw local metrics](data/mcl1_local_structure_comparison_241026.json), [RMSD measurements](data/mcl1_precision_structure_comparison_241026.json) and [metric implementation](../tools/structure_metrics.py) retain the definitions and individual values. The [runtime study](official_model_performance_241026.md) retains the official precision policy and instead uses an output-preserving cache.
