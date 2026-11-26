# WeaveSight

Annotation-free fabric defect localisation. Three anomaly-detection families are compared, all trained on defect-free images only: Dinomaly (Transformer reconstruction), a convolutional autoencoder (CAE, course-based baseline) and PatchCore (feature-based benchmark). Course project, Generative AI, FAST-NUCES, Fall 2026.

Data: MVTec AD texture classes (carpet, grid, leather, tile, wood) and AITEX real fabric (7 fabrics). Output: heatmap, mask and PASS/FAIL verdict in a Streamlit demo.

## Repository layout

- `splits/` image-level splits made before any patch extraction (`mvtec_seed42.csv`, `aitex_seed42.csv`)
- `src/` training, evaluation, calibration and analysis scripts
- `app/` Streamlit interface (`app.py`), `predict.py` (Dinomaly, PatchCore), `predict_cae.py` (CAE), thresholds (`thresholds.json`, `thresholds_cae.json`)
- `results/` result files, failure figures and the merged table `manahil_merged_table.csv`
- `checkpoints/cae/` the 12 CAE checkpoints with `SHA256SUMS.txt`

Result file prefixes: `ayesha_*` (Dinomaly, PatchCore) and `manahil_*` (CAE).

## Environment

Python 3.13, torch 2.15.0+cu132, `anomalib==2.7.1` (pinned in `requirements.txt`). Do not use a different anomalib version: the AUPRO metric and some results can change. anomalib's automatic MVTec download is broken (issue #3892), so download MVTec manually. MVTec AD is non-commercial (CC BY-NC-SA 4.0).

Scripts use lab-PC paths: `~/datasets/MVTec_split/<class>/{train_good,val_good,test_good,test_defect,test_mask}` and `~/datasets/AITEX_tiles/fabric_XX/...` (symlinked as `MVTec_split/aitex_XX`). AITEX strips (4096x256) were split first and then tiled to 256x256.

## Evaluation rules (all three models)

- Train on `train_good` only. `val_good` is used only to set thresholds.
- Raw scores only. anomalib's own rescaling and its F1 numbers are not used.
- Image threshold: 99th percentile (P99) of raw `val_good` image scores (main result). Sensitivity rules, labelled as such: validation maximum, and mean + 3 standard deviations.
- Metrics: image AUROC, pixel AUROC, AUPRO (fpr limit 0.3), and the share of normal test images flagged and defective images caught at P99.
- Masks get the same resize as the images (nearest-neighbour).
- Single seed (42), one run per class. Small test sets and tiles from one strip are correlated, so treat the numbers as rough.

## CAE baseline (Manahil)

Fixed before any test result was seen, identical for all 12 categories:

- Input resized to 256x256, random 128x128 training patches
- Encoder with five stride-2 convolutions, dense latent of 100, mirrored decoder
- Loss L1 + (1 - SSIM), Adam, learning rate 2e-4, batch 64, **20,000 steps**, seed 42
- Inference: 128x128 window at stride 32, overlapping reconstructions averaged, per-pixel L1 residual, Gaussian smoothing (sigma 4)
- Image score: **maximum of the map**. Dinomaly uses the mean of the top 1% of its map, so image scores are not directly comparable.

Two things to know:

- A first run used 5,000 steps. The training loss was still falling, so the step count was raised to 20,000 before any test result was seen. The 5,000-step checkpoint was never evaluated.
- The first AITEX batch stopped during fabric 03 for an unknown reason. Fabric 03 was retrained from scratch for the full 20,000 steps.

Run from `src/`:

```
python train_cae.py --category carpet --steps 20000
python eval_cae.py --category carpet
python failure_cases_cae.py --category carpet
```

Pass `--steps 20000` explicitly: the default in `train_cae.py` is still 5000. For AITEX use `--category aitex_03` and so on. `eval_cae.py` writes `results/manahil_{rawthreshold,pixel,tilescores}_cae_<category>.json`.

## Checkpoints

`checkpoints/cae/cae_<category>.ckpt` are the files `cae_<category>.pt` renamed, and `SHA256SUMS.txt` holds their checksums (`sha256sum -c SHA256SUMS.txt`). `app/predict_cae.py` loads `~/results/cae_<category>/cae_<category>.pt`, so copy a checkpoint there under that name to use it elsewhere.

## Interface

```
streamlit run app/app.py
```

`predict(image, model, category)` returns a dict with `heatmap`, `mask`, `score`, `threshold`, `pixel_threshold` and `verdict`. The image threshold is P99 of raw `val_good` scores and the pixel threshold is the 99.9th percentile of all raw `val_good` pixel values.

## Results and limits

All numbers are in `results/manahil_merged_table.csv` (12 categories x 3 models). Dinomaly was trained for 1,000 steps (the paper uses about 5,000), batch size 8. The CAE is clearly behind on MVTec and close to the other models on only a few AITEX fabrics.

- AITEX fabric 05 has one defect tile, so its numbers are not meaningful.
- Fabrics 00 and 06 have only 9 and 7 defect tiles.
- Failure-case figures show worst cases (highest-scoring false alarms, lowest-scoring misses), not typical behaviour.
- AITEX breakdowns (strip type, tile position, defect size) are post-hoc and single seed. Their causes are unproven.
- Latency numbers are provisional: the GPU was shared with another user's job.
- Unverified: PatchCore's image-score definition, and whether defective AITEX strips were captured differently from defect-free ones, which could inflate AITEX image AUROC.
