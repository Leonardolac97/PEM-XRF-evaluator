# PEM XRF Analysis Software

A PyQt6-based graphical user interface for processing, visualizing, segmenting, and quantifying X-ray fluorescence (XRF) maps of PEM water electrolysis catalyst-coated membrane (CCM) cross-sections.

This software was developed for PEMWE CCM XRF datasets, with a focus on Pt, Ir, Fe, Ti, absorption-contrast maps, membrane segmentation, catalyst-layer separation, and quantitative post-processing of elemental distributions.

---

## Main features

- Load XRF TIFF stacks and individual TIFF images.
- Display Pt and Ir emission maps.
- Extract and display TIFF stack channel names.
- Apply thresholding methods to emission maps, including:
  - manual thresholding,
  - minimum thresholding,
  - triangle thresholding,
  - Otsu thresholding.
- Generate binary masks for Nafion/excess regions and catalyst-containing regions.
- Segment Nafion, Pt catalyst layer, and Ir catalyst layer regions.
- Extract approximate catalyst and membrane contours.
- Fine-tune membrane segmentation.
- Filter high-intensity dust or artefact pixels from Nafion regions.
- Apply multi-Otsu segmentation to Nafion membrane intensity maps.
- Estimate membrane thickness from absorption-contrast images.
- Quantify elemental distributions using thickness-corrected maps.
- Generate publication-style maps, histograms, 3D intensity plots, and summary figures.
- Export intermediate and final plots for reporting, thesis work, or publication preparation.

---

## Status

This repository contains research/prototype code developed for XRF analysis of PEMWE catalyst-coated membrane cross-sections.

The current version prioritizes preservation of the working analysis workflow and reproducibility of the data treatment. The internal code structure is still being refactored and may later be split into separate modules for GUI handling, image processing, plotting, and quantification.

---

## Input data

The expected input consists of processed XRF map files, typically provided as TIFF stacks or individual TIFF images.

The workflow assumes that the TIFF stack contains element-specific map channels, for example:

```text
Pt
Ir
Fe
Ti
Absorption contrast
```

The exact order and names of the channels depend on the preprocessing and export procedure used before loading the data into the software.

---

## Typical workflow

1. Select or create a project folder.
2. Load the processed XRF TIFF stack.
3. Inspect Pt and Ir emission maps.
4. Select thresholding parameters.
5. Generate Nafion and catalyst binary masks.
6. Segment catalyst layers and membrane regions.
7. Fine-tune the Nafion membrane mask if needed.
8. Filter dust or artefact pixels.
9. Segment the Nafion region using multi-Otsu thresholding.
10. Load the absorption-contrast image.
11. Convert absorption contrast to membrane thickness.
12. Quantify elemental distributions using thickness-corrected maps.
13. Export figures and intermediate CSV outputs.

---

## Installation

### Option 1 — Conda / Mamba

```bash
conda env create -f environment.yml
conda activate xrf-pemwe-analysis
python XRF_PEMWE_software.py
```

### Option 2 — pip + virtual environment

Create a virtual environment:

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python XRF_PEMWE_software.py
```

#### macOS / Linux

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python XRF_PEMWE_software.py
```

---

## Requirements

Recommended Python version:

```text
Python 3.11
```

Main Python packages:

```text
PyQt6
matplotlib
numpy
pandas
scikit-image
tifffile
```

---

## Running the software

After installing the dependencies, run:

```bash
python XRF_PEMWE_software.py
```

The software opens a PyQt6 graphical interface from which the user can load XRF maps, set segmentation parameters, generate figures, and export analysis outputs.

---

## Output files

Depending on the selected analysis steps, the software can generate folders and files such as:

```text
project_folder/
├── Final_Plots/
│   └── Final_Plots_paper_<element>_det_<detector>/
│       ├── Pt and Ir slices.png
│       ├── Cat and excess.png
│       ├── Sample segmented_epoxy.png
│       ├── Sample segmented.png
│       ├── Main2.png
│       └── Thickness_map.png
├── Plot_of_stacks/
│   └── Plot_of_stacks_<element>/
│       ├── gray_values_nafion_DF.csv
│       ├── gray_values_Pt_DF.csv
│       ├── gray_values_Ir_DF.csv
│       └── gray_values_real_DF.csv
└── variables/
```

Generated folders, TIFF files, CSV outputs, and figures are ignored by the provided `.gitignore` file unless explicitly added by the user.

---

## Known limitations

- The current version is a research/prototype GUI rather than a fully packaged Python library.
- The workflow assumes that the input XRF maps have already been processed and exported in a compatible TIFF format.
- Some default variables and plotting settings were developed for PEMWE CCM case-study data and may need adjustment for other datasets.
- Segmentation quality depends strongly on image quality, threshold selection, and the contrast between membrane, catalyst layers, epoxy, and artefacts.
- Quantification depends on calibration factors, thickness correction, and assumptions defined by the user.
- Large TIFF files and generated outputs should not be committed directly to the repository.

---

## Citation

Please cite the software using the metadata provided in `CITATION.cff`.

The DOI and GitHub repository fields are intentionally left open in the current citation file and should be filled after repository creation and archiving.

---

## License

This repository is distributed under the MIT License.
