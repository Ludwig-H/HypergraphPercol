# HypergraphPercol

HypergraphPercol is a research-grade clustering algorithm based on hypergraph percolation.

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

The commands below describe the exact sequence used to validate the repository in the development container. Adjust the package
manager commands for your platform as needed.

1. **Clone the repository and install the Python package in editable mode.**

   ```bash
   git clone https://github.com/Ludwig-H/HypergraphPercol.git
   cd HypergraphPercol
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```

2. **Download the CGAL helper repositories.**

   ```bash
   ./scripts/setup_cgal.py
   ```

   This script clones six helper projects into `CGALDelaunay/`. If you already have them locally, place them in that directory
   or point the `CGALDELAUNAY_ROOT` environment variable to their location.

3. **Install the system packages required to compile the helpers.**

   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libcgal-dev libtbb-dev libtbbmalloc2 libgmp-dev libmpfr-dev libeigen3-dev
   ```

4. **Build each helper project.**

   ```bash
   for project in CGALDelaunay/*; do
       [ -d "$project" ] || continue
       cmake -S "$project" -B "$project/build" -DCMAKE_BUILD_TYPE=Release
       cmake --build "$project/build" -j
   done
   ```

   The loop above is equivalent to running the `cmake` configure/build commands in each helper directory (e.g. `EdgesCGAL1D`,
   `EdgesCGAL2D`, …). You can also execute the commands manually if you prefer.

5. **Install optional extras** (only required for the UMAP dimensionality reduction pipeline):

   ```bash
   pip install umap-learn
   ```

> **Note**
> The editable install step compiles a small Cython extension that accelerates key geometric routines. Ensure a C/C++ toolchain
> and Python headers are available on your platform.

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
