# POTR
Pose Transformer


## Standard POTR

## Deformable POTR

The implementation is based on [this implementation](https://github.com/fundamentalvision/Deformable-DETR) of Deformable DETR.

### Building

Because the original implementation structures the `MSDeformAttention` module as a separate Python package with its own Python, CUDA, and C++ scripts, one needs to build this before running the training.

First create a dedicated **Conda environment** with Python 3.7:

```bash
conda create -n deformable_potr python=3.7 pip
conda activate deformable_potr
```

Then install all the dependencies using:

```bash
pip install -r requirements.txt
conda install cudatoolkit-dev -c conda-forge
```

And finally, build the `MSDeformAttention` module (and all other custom CUDA operators) using:

```bash
cd ./deformable_potr/models/ops
sh ./make.sh
```

You can then test the CUDA operators by running `python test.py`, where all tests should pass.

To get notebooks install juyptext
```bash
pip install jupytext
```

## How to get Jupyter notebooks

To facilitate nice Git workflow we use [Jupytext](https://github.com/mwouts/jupytext).
See their manual for installation instruction and when creating your own notebooks.

To get Jupyter Notebook `.ipynb` out of a `.py` file using Jupyter Lab:
1. Have Jupytext installed.
2. Run Jupyter Lab.
3. In navigation sidebar right-click on a `.py` file, select Open With -> Notebook.
4. Save the just opened notebook. New file with `.ipynb` extension should appear.
5. Enjoy the new notebook, don't forget to close `.py` file opened as notebook ;-)