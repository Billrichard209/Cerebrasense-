# FreeSurfer Structural Metrics Integration

This backend can parse externally generated FreeSurfer statistics files, but it
does not run FreeSurfer and does not fabricate missing anatomical measurements.

## Expected Inputs

After running FreeSurfer outside this backend, a subject directory usually
contains files like:

```text
SUBJECTS_DIR/<subject_id>/stats/aseg.stats
SUBJECTS_DIR/<subject_id>/stats/lh.aparc.stats
SUBJECTS_DIR/<subject_id>/stats/rh.aparc.stats
```

The backend parser currently uses:

- `aseg.stats` for brain-region volumes and hippocampal volumes.
- `lh.aparc.stats` and `rh.aparc.stats` for cortical thickness summaries.
- Derived left/right hippocampal asymmetry only when both left and right
  hippocampal rows are present.

## Build A Report

From `archive (1)`:

```powershell
cmd /c build_structural_metrics_report.cmd --subject-id OAS1_0002 --session-id OAS1_0002_MR1 --aseg-stats-path "C:\path\to\aseg.stats" --lh-aparc-stats-path "C:\path\to\lh.aparc.stats" --rh-aparc-stats-path "C:\path\to\rh.aparc.stats"
```

Reports are saved under:

```text
alz_backend/outputs/reports/structural_metrics/
```

## Safety Notes

- Hippocampal volume and cortical thickness are accepted only from external
  segmentation/statistics outputs in this version.
- The existing foreground-proxy volumetrics remain useful for backend
  prototyping, but they are not a replacement for validated segmentation.
- Structural metrics are feature inputs for research and decision-support
  workflows, not diagnostic conclusions.

