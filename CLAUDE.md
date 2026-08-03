# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Prototype pipeline for processing NISAR SAR data to generate pre-training datasets for geoscience foundation models. Uses ISCE3 (InSAR Scientific Computing Environment v3) Python API as the core SAR processing backend, supplemented by custom Python code.

## Environment

The project uses a conda environment named `fmnisar` (defined in `environment.yml`). Activate it before running anything:

```bash
conda activate fmnisar
```

## Key Domain Concepts

- **NISAR**: NASA-ISRO SAR Mission — L-band and S-band dual-frequency SAR satellite. Data products include RSLC (Range-compressed Single-Look Complex), GSLC, GCOV, GUNW, etc.
- **Foundation model pre-training**: The output of this pipeline is patches/chips of SAR amplitude, coherence, or phase data intended as self-supervised pre-training inputs.
