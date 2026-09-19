# Work Summary and Handover

## Overview

I created the `2d_slice_crop` branch and developed an end-to-end workflow for
sampling and analysing representative 2D fetal-brain sections from NIfTI label
volumes and surface meshes. The workflow finds the maximum cross-sectional
area, selects a band of nearby slices, calculates area/perimeter and LGI-related
measurements, produces images and data summaries, optionally overlays pial
surfaces, and crops consistent regions of interest for further analysis.

## Detailed list

| Component | What it does | Location |
| --- | --- | --- |
| NIfTI area-band sampler | Loads labelled NIfTI volumes, selects explicit tissue labels, calculates the slice-area profile, locates the maximum-area region and sampling band, and writes slice images, profiles, measurements, CSV/Excel files, and JSON summaries. | `functions/nifti_area_sampler.py` |
| ROI cropper | Crops the generated band slices using global, axis-specific, or automatic normalised bounding boxes. It supports batch processing, validation, manifests, and physically scaled bars placed outside the tissue image. | `functions/crop_band_roi.py` |
| Pial overlay | Loads left/right pial surfaces, maps scanner or FreeSurfer coordinates into voxel space, intersects mesh triangles with the selected slice, and draws the pial contour on output images. | `functions/pial_overlay_tri.py` |
| Surface-mesh sampler | Supports STL, VTK, PLY, and OBJ input. It voxelises and solidifies imperfect meshes, reuses the NIfTI sampling pipeline, preserves detailed mesh-section outlines, and reports mesh and raster perimeter/LGI measurements. | `functions/stl_area_sampler.py` |

Contents of `scripts/`:

- `scripts/area_band_cli.py` — command-line entry point for single or batch
  NIfTI runs, including label validation, legends, all-axis processing, and
  optional pial overlays.
- `scripts/stl_area_cli.py` — command-line entry point for single or batch mesh
  runs.
- `scripts/rescale_atlas_to_true_size.py` — creates clean, true-size dHCP atlas
  parcellations by combining the clean shape from `parcellations_scaled` with
  the measured brain size from `parcellations_orig_size`.

Committed example configurations are under `configs/`. They document label
selection and recommended settings for single cases, batches, the dHCP atlas,
cortex-inclusive measurements, true-size atlas data, ROI cropping, and mesh
sampling.

## Where to find detailed information

- `AREA_BAND_CLI_RUN.md` — full NIfTI workflow, label handling, single and batch
  commands, output structure, scale bars, dHCP atlas guidance, and the true-size
  rescaling method.
- `CROP_BAND_ROI.md` — ROI configuration, normalised bounding boxes, scale-bar
  behaviour, output locations, and troubleshooting.
- `STL_AREA_CLI_RUN.md` — mesh solidification and sampling method, parameter
  choices, known measurement bias, outputs, and run examples.
- `configs/*.example.json` — ready-to-copy configurations with notes explaining
  the dataset-specific choices.
- The dataclasses at the start of each sampler/crop module contain the complete
  programmatic configuration options.

## Typical outputs

Each subject/axis output contains rendered band slices, the area profile,
per-slice CSV/Excel measurements, and JSON run summaries. Cropped images are
written under `brain_slices_cropped/` with a `crop_manifest.json`. Mesh runs
also provide detailed outline images and mesh-specific measurement summaries.

The exact directory layout and meaning of each output field are documented in
the three guides above.
