module purge
module load miniconda/3
conda init
conda create --name APCEMM-caleb python=3.8
conda activate APCEMM-caleb
python -m pip install python3-dev
python -m pip install netcdf4
python -m pip install chaospy
python -m pip install xarray
python -m pip install matplotlib
python -m pip install importlib-metadata==4.13.0  
