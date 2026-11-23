# Computing environment of the paper's runs

Collected with `tools/check_system_info.py` on 23 Nov 2026 on the computer that produced
all run times reported in the paper.

| Item | Value |
|---|---|
| CPU | Intel Core i5-4570 @ 3.20 GHz, 4 physical cores, 4 logical processors |
| RAM | 16 GB (15.9 GB usable) |
| GPU | NVIDIA GeForce GT 730 present but not used (`torch.cuda.is_available()` = False; lungmask run with `force_cpu=True`, TotalSegmentator with `device="cpu"`) |
| Operating system | Windows 10 Pro, 64-bit (build 19045) |
| Python | 3.11.7 (Anaconda) |
| Parallel workers | `n_jobs = -1` (all 4 cores) for the per-slice threshold sweep (A2); PyTorch used 4 CPU threads |

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.11.4 |
| scikit-image | 0.26.0 |
| nibabel | 5.2.1 |
| pwlf | 2.2.1 |
| joblib | 1.2.0 |
| pandas | 2.1.4 |
| matplotlib | 3.8.0 |
| SimpleITK | 2.3.1 |
| pydicom | 3.0.2 |
| torch | 2.7.0 |
| lungmask | 0.2.21 |
| TotalSegmentator | 2.18.0 |

## What the run times measure

* **Proposed method** – `timing_s.total_s` in each `run_meta.json`: wall-clock time of the
  whole pipeline on a volume already in memory (file reading and writing excluded). It also
  includes the computation of the ablation configuration without A4b and A4d (one extra A5
  and A5b run, about 3% of the total).
* **lungmask** – wall-clock time of `LMInferer.apply` on an image already in memory.
* **TotalSegmentator** – wall-clock time of the Python API call
  (`roi_subset` = the five lung lobes, `fast=False`, `device="cpu"`), which includes its own
  reading of the input file and its internal pre- and post-processing.
* **Classical methods** – wall-clock time of the segmentation function on a volume in memory.
