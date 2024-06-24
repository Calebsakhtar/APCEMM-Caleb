module load miniconda/3
conda init
conda create --name APCEMM-caleb python=3.8
conda activate APCEMM-caleb
python -m pip3 install python3-dev
python -m pip3 install netcdf4
python -m pip3 install chaospy
python -m pip3 install xarray
python -m pip3 install matplotlib
python -m pip3 install importlib-metadata==4.13.0  
