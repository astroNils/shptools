# shptools_BOULDERING

(last updated the 20th of July 2023).

Shptools is a package that gathers a collection of tools to manipulate shapefiles. A large portion of this package was written for  the pre- and post-processing of planetary images so that it can be easily ingested in deep learning algorithms. Because of that, some of the functions are a bit, sometimes, too specific and repetitive (sorry for that!). I will try over time to improve this GitHub repository. Please contribute if you are interested. 

This GitHub repository is written following the functional programming paradigm. 

## To do

- [x] Add multi-ring function.

## Installation

Create a new environment if wanted. Then you can install the rastertools by writing the following in your terminal. 

```bash
git clone https://github.com/astroNils/shptools.git
cd shptools
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING
pip install -r requirements.txt
```

You should now have access to this module in Python.

```bash
python
```

```python
from shptools_BOULDERING import shp, geometry as shp_geom, geomorph as shp_geomorph, annotations as shp_anno
```

## Getting Started

A jupyter notebook is provided as a tutorial ([GET STARTED HERE](./resources/nb/GETTING_STARTED.ipynb)).





