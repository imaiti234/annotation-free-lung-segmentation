# Training-free lung segmentation in volumetric CT

Code for the paper *"Training-free lung segmentation in volumetric CT through adaptive
thresholding, anatomy-constrained recovery, and 3D continuity"* (I. Maiti,
M. Mahadevappa, N. Ghosh; submitted to *Scientific Reports*).

The method segments the right and left lung in a chest CT volume without training data,
reference masks or pretrained weights. Every step is a deterministic function of the CT
being segmented, and it runs on a CPU.

| Stage | File | What it does |
|---|---|---|
| A1 | `lungseg/alg1_coarse.py` | intensity normalisation, body mask, coarse lung at −320 HU |
| A2 | `lungseg/alg2_elbow.py` | per-slice threshold at the elbow of the area-versus-threshold curve (piecewise-linear fit, order chosen by BIC) |
| A3 | `lungseg/alg3_separation.py` | left–right lung separation (threshold descent, bridge test) |
| A4 | `lungseg/alg4_merging.py` | adaptive merging of boundary regions that resemble the lesions enclosed by the lung |
| A4b | `lungseg/alg4b_cavity_recovery.py` | recovery of diseased lung constrained to the thoracic cavity |
| A4d | `lungseg/alg4d_lesion_model.py` | patient-specific lesion likelihood and geodesic growth |
| A5 | `lungseg/alg5_connectivity3d.py` | 3D inter-slice connectivity correction |
| A5b | `lungseg/alg5b_airway.py` | removal of the trachea and main bronchi |

`lungseg/pipeline.py` runs them in order and saves every stage. All parameters are in
`lungseg/config.py` (configuration version `2026-10-15c_lesion_model`, used for every
result in the paper). `lungseg/alg4c_costal_envelope.py` and the "dense pleural recovery"
option of A4b are experimental, switched off in the configuration and not part of the
method; they are kept for transparency.

## Installation

Python 3.11 was used.

```bash
conda create -n lungseg python=3.11 -y
conda activate lungseg
pip install -r requirements.txt            # the method only
pip install -r requirements-paper.txt      # + comparators and LCTSC loading (exact versions of the paper)
```

## Quick test (synthetic phantom, about a minute)

```bash
python scripts/00_make_phantom.py --out data/phantom
python scripts/01_run_segmentation.py --images data/phantom/rp_im --out outputs/phantom
```

## Segment your own CT

```bash
python scripts/01_run_segmentation.py --images <folder with .nii/.nii.gz> --out <output folder>
```

Output per case: `A5_final.nii.gz` (1 = right lung, 2 = left lung), the intermediate
stages, `slice_diagnostics.csv` (per-slice threshold and number of linear pieces),
`config_used.json` and `run_meta.json` (run time per stage).

## Reproducing the paper

The CT data are not included. Download them from the original sources:

* **COVID-19-CT-Seg** (20 scans; Ma et al.): https://doi.org/10.5281/zenodo.3757476
  → `data/COVID19_CT_Seg/all` (CT), `.../lung_mask`, `.../Infection_Mask`
* **LCTSC** (60 scans; Yang et al., The Cancer Imaging Archive):
  https://doi.org/10.7937/K9/TCIA.2017.3r3fvz08 → `data/LCTSC`

Then run, in Spyder (open the file, press F5) or with `python <file>`:

1. `RUN_IN_SPYDER.py` – proposed method on COVID-19-CT-Seg, evaluation, figures. It writes
   the frozen configuration that every later run reuses.
2. `RUN_COMPARISON_IN_SPYDER.py` – comparators on COVID-19-CT-Seg: lungmask U-Net R231,
   R231CovidWeb and LTRCLobes (CPU), TotalSegmentator v2 (CPU, lung lobes merged),
   Otsu + 3D components and a re-implementation of Hu et al. (2001).
3. `RUN_SEVERITY_IN_SPYDER.py` – analysis by infection burden.
4. `RUN_LCTSC_IN_SPYDER.py` – the same frozen method and comparators on LCTSC.
5. `python analysis/paper_statistics.py` – recomputes the statistics of Tables 1–4 from
   the per-scan results (expected output: `analysis/paper_statistics.txt`).

Development cases, kept out of all statistics: `coronacases_001`, `coronacases_002`,
`radiopaedia_14_85914_0` and `radiopaedia_27_86410_0` (used to set the parameters), and
`LCTSC-Test-S3-104` (used to develop the original elbow method).

## Results included

`results/covid19_ct_seg` and `results/lctsc` contain the per-scan metrics of every method
(`comparison_metrics_all_cases.csv`), the stage-wise metrics (COVID-19-CT-Seg), the run
times, the infection-burden tables, the frozen configuration and, for every scan, the
configuration used, per-stage run times and per-slice threshold diagnostics. Predicted
masks are not included (size); they can be regenerated with the steps above.

## Hardware used for the run times

All run times in the paper were measured on one desktop computer: Intel Core i5-4570
(4 cores, 4 threads, 3.2 GHz), 16 GB RAM, Windows 10 Pro, Python 3.11.7, no GPU used by
any method. See `ENVIRONMENT.md`. `tools/check_system_info.py` collects the same
information on your computer.

## Citation

If you use this code, please cite the paper and the archived code (see `CITATION.cff`).

## Licence

MIT (see `LICENSE`).
