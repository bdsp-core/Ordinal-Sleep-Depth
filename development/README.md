# Development

This folder contains the older research and paper-reproduction workflow.

Contents include:
- preprocessing scripts
- training scripts
- prediction scripts used during development
- statistics and figure-generation scripts
- optional ORP-related scripts
- exploratory notebook(s)

The public inference entrypoint for new users is not here.

Use instead:

```bash
python -m OSD.demo
python OSD/inference.py /path/to/file.h5 --plot --print-summary
```

Some scripts in this folder assume project-specific paths and historical environments. They are preserved for reproducibility, not polished as product-facing entrypoints.
