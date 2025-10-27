# HypergraphPercol

HypergraphPercol is a research-grade clustering algorithm based on hypergraph percolation.

## Features

- Weighted Delaunay / order-k Delaunay support via CGAL helper binaries
- Integration with HDBSCAN for hierarchical cluster extraction
- Optional PCA / UMAP dimensionality reduction pipeline
- Parallel radius computation using [`cyminiball`](https://pypi.org/project/cyminiball/)
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

1. **Clone the repository and prepare a virtual environment.**

   ```bash
   git clone https://github.com/Ludwig-H/HypergraphPercol.git
   cd HypergraphPercol
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. **Build and install `cyminiball` from source.**

   The PyPI sdist of `cyminiball` ships generated C++ bindings that no longer
   compile on modern Python/NumPy combinations. The helper script below clones
   the upstream repository, cythonises the bindings and installs the fresh
   build into the current environment:

   ```bash
   ./scripts/build_cyminiball.sh
   ```

   Re-run the script to update to a newer commit or point the
   `CYMINIBALL_REF`/`CYMINIBALL_REPO` environment variables at an alternative
   revision.

3. **Install HypergraphPercol in editable mode.**

   ```bash
   pip install -e .
   ```

4. **Download the CGAL helper repositories.**

   ```bash
   ./scripts/setup_cgal.py
   ```

   This script clones six helper projects into `CGALDelaunay/`. If you already have them locally, place them in that directory
   or point the `CGALDELAUNAY_ROOT` environment variable to their location.

5. **Install the system packages required to compile the helpers.**

   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libcgal-dev libtbb-dev libtbbmalloc2 libgmp-dev libmpfr-dev libeigen3-dev
   ```

6. **Build each helper project.**

   ```bash
   for project in CGALDelaunay/*; do
       [ -d "$project" ] || continue
       cmake -S "$project" -B "$project/build" -DCMAKE_BUILD_TYPE=Release
       cmake --build "$project/build" -j
   done
   ```

   The loop above is equivalent to running the `cmake` configure/build commands in each helper directory (e.g. `EdgesCGAL1D`,
   `EdgesCGAL2D`, …). You can also execute the commands manually if you prefer.

7. **Install optional extras** (only required for the UMAP dimensionality reduction pipeline):

   ```bash
   pip install umap-learn
   ```

> **Note**
> The editable install step compiles a small Cython extension that accelerates key geometric routines. Ensure a C/C++ toolchain
> and Python headers are available on your platform.
>
> `scripts/build_cyminiball.sh` installs the upstream `cyminiball` sources with
> a local Cython build so that Python 3.10+ and NumPy 2.x environments remain
> compatible. The project metadata now tracks those relaxed constraints.

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

## Google Colab build notebook

If you prefer building the full stack inside a managed environment, open [`hgp-compil.ipynb`](./hgp-compil.ipynb) in Google Colab (``File → Open notebook → GitHub`` and paste the repository URL, or use the Colab badge below). The notebook mirrors the Dockerfile you provided:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ludwig-H/HypergraphPercol/blob/main/hgp-compil.ipynb)

1. Installs the required system packages (`libcgal-dev`, `libtbb-dev`, `libboost-all-dev`, …).
2. Clones both `HypergraphPercol` and the `cyminiball` dependency at runtime.
3. Builds `cyminiball`, patches the CGAL helper projects for pthread linkage, and compiles every helper binary.
4. Installs the freshly built wheel so that calling `HypergraphPercol(...)` works inside the notebook session.

After the build cells complete, the final validation cell performs a quick clustering run to confirm that `HypergraphPercol` is usable from the notebook.

### Verifying the build locally

The hosted Colab runtime currently ships with Python 3.10, but newer desktop distributions (including the automated checks run for this repository) already use Python 3.12. When reproducing the pipeline locally with Python 3.12 you must ensure that `pip` reuses the `cyminiball` wheel built earlier in the process. The easiest way to do so is to install the runtime dependencies first and then install HypergraphPercol with `--no-deps`:

```bash
# build the CGAL helpers following the notebook instructions, then run
python3 -m pip install --upgrade scikit-learn hdbscan gudhi joblib threadpoolctl
python3 -m pip install --no-deps --force-reinstall .
```

This sequence avoids re-invoking the `cyminiball` build step, which fails under Python 3.12 due to upstream API changes, while still producing a working installation of HypergraphPercol.

## Testing

You can verify the installation with the included unit tests:

```bash
pytest
```

## License

The Python code in this repository is released under the MIT License. The CGAL helper projects keep their original licenses.
