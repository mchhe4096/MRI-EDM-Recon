# Comparison

This folder is an isolated comparison workspace for **metric-protocol alignment only**.
It does not modify the original training/sampling pipeline under `scripts/` and `src/`.

## Files

- `sample_param_search_author.py`
  - copied from `scripts/sample_param_search.py` and simplified.
  - removed non-essential metrics (SSIM/NMSE/LPIPS/HFEN).
  - added author-aligned PSNR protocol options:
    - complex normalize + z-score (`--author_norm`)
    - volume slice crop (`--author_crop_head`, `--author_crop_tail`, default `4:-1`)
    - data_range mode (`--author_range volume|slice`)

## Run

From `my-project` root:

```bash
python -m Comparison.sample_param_search_author \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --num_cases 12 \
  --num_samples 1 \
  --steps_list 30,40 \
  --dc_start_list 0.7,0.8 \
  --dc_every_list 2,3 \
  --dc_lam_list 0.08,0.12,0.15 \
  --dc_ramp_list 0,1 \
  --include_no_dc_baseline \
  --init_mode_list zf,blend \
  --init_blend_list 0.5,0.7 \
  --metric_protocol both \
  --rank_metric author_psnr \
  --author_norm \
  --author_crop_head 4 \
  --author_crop_tail 1 \
  --author_range volume \
  --device cuda \
  --outdir ../outputs/param_search_comparison
```

## Outputs

- `summary.csv`: per-config aggregate (`current_psnr_mean`, `author_psnr_mean`)
- `per_case.csv`: per-slice current PSNR
- `per_volume.csv`: per-volume author PSNR
- `baseline_zf.json`: ZF baseline summary (both metrics)
- `baseline_zf_per_case.csv`, `baseline_zf_per_volume.csv`
- `topk.json`, `meta.json`
