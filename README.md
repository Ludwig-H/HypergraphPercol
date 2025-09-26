# HypergraphPercol

HypergraphPercol is a research-grade clustering algorithm based on hypergraph percolation. This repository turns the original research notebook into a reusable Python package with a reproducible project layout, documentation, and examples.

## Features

- Weighted Delaunay / order-k Delaunay support via CGAL helper binaries
- Integration with HDBSCAN for hierarchical cluster extraction
- Optional PCA / UMAP dimensionality reduction pipeline
- Parallel radius computation using [`miniball`](https://pypi.org/project/MiniballCpp/)
- Example script showing end-to-end usage on synthetic data

## Project layout

```
CGALDelaunay/           # Expected location of CGAL helper projects (see below)
examples/demo.py        # Minimal demonstration of HypergraphPercol
scripts/setup_cgal.py   # Utility to clone the CGAL helper repositories
src/hypergraphpercol/   # Python package
└── core.py             # Main HypergraphPercol implementation
```

## Installation

1. Clone this repository and the CGAL helper executables:

   ```bash
   git clone https://github.com/your-user/HypergraphPercol.git
   cd HypergraphPercol
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install -e .
   ./scripts/setup_cgal.py
   ```

   The helper script clones the six repositories containing the prebuilt CGAL executables into `CGALDelaunay/`. If you already have them, place them in that directory or set the `CGALDELAUNAY_ROOT` environment variable to point to their location.

   > **Note**
   > The editable install step compiles a small Cython extension that accelerates key geometric routines. Ensure a C/C++ toolchain and Python headers are available on your platform.

2. Install optional extras (if you need UMAP-based dimensionality reduction):

   ```bash
   pip install umap-learn
   ```

## Usage

```python
import numpy as np
from hypergraphpercol import HypergraphPercol

X = np.random.random((200, 3))
labels = HypergraphPercol(
    X,
    K=2,
    min_cluster_size=30,
    min_samples=5,
    metric="euclidean",
    complex_chosen="auto",
    expZ=2,
    precision="safe",
    verbeux=True,
    cgal_root="./CGALDelaunay",
)
```

The `cgal_root` argument is optional; when omitted, the package searches for the helper binaries inside `CGALDelaunay/` relative to the repository root or uses the `CGALDELAUNAY_ROOT` environment variable.

Set `return_multi_clusters=True` to obtain, for each point, the list of candidate clusters with membership probability.

## Example

Run the demo script to build a synthetic dataset and cluster it:

```bash
python examples/demo.py
```

## Testing

You can verify the installation with the included unit tests:

```bash
pytest
```

## License

The Python code in this repository is released under the MIT License. The CGAL helper projects keep their original licenses.
