# Machine Learning in Image Analysis

Coursework and experiments for a Machine Learning in Image Analysis class. This repository contains Python and Jupyter Notebook solutions, supporting data, figures, and assignment write-ups.

## Repository structure

```text
.
├── PS1/
│   ├── ps1.py               # Problem Set 1 implementation
│   ├── image/               # Input images and generated figures
│   └── *.pdf                # Assignment and submission documents
└── PS2/
    ├── PS2.pdf              # Problem Set 2 handout
    └── PS2_Q2/
        ├── ReadMe.rtf       # Question-specific notes
        ├── PS2_Q2_3_sup_materials.pdf
        └── code+data/
            ├── PS2_Q2.ipynb
            ├── source.pt
            └── v0.pt
```

## Getting started

Clone the repository and create a virtual environment:

```bash
git clone git@github.com:yueeez-c/Machine-Learning---Image-Analysis.git
cd Machine-Learning---Image-Analysis
python3 -m venv .venv
source .venv/bin/activate
```

Install the libraries required by the assignment you want to run. The current work uses common image-analysis and scientific Python packages such as NumPy, Pillow, Matplotlib, OpenCV, PyTorch, and Jupyter:

```bash
python -m pip install numpy pillow matplotlib opencv-python torch jupyter
```

Run the Problem Set 1 script from the repository root:

```bash
python PS1/ps1.py
```

Or launch Jupyter for Problem Set 2:

```bash
jupyter notebook PS2/PS2_Q2/code+data/PS2_Q2.ipynb
```
