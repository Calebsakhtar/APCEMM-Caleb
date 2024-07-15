import os
import shutil
import os.path
import pickle
import time
import shutil
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def extract_APCEMM_value(file_path, quantity_name):
    """
    Valid quantities:
        - "intOD"
        - "width"
        - "depth"
        - "Ice Mass"
        - "Number Ice Particles"
    """
    ds = xr.open_dataset(file_path, engine = "netcdf4", decode_times = False)

    # Save the variables that do not lie on a grid
    return ds[quantity_name].values[0]

def compare_APCEMM_out_runs(time_hrs, time_mins, run_number_a, run_number_b, quantity_to_compare):
    """
    Valid quantities to compare:
        - "intOD"
        - "width"
        - "depth"
        - "Ice Mass"
        - "Number Ice Particles"
    """
    run_number_a = str(run_number_a)
    run_number_b = str(run_number_b)

    directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    directory = os.path.join(directory, "APCEMM_out/")
    directory_a = os.path.join(directory, f"Run{run_number_a}/")
    directory_b = os.path.join(directory, f"Run{run_number_b}/")

    str_hrs = str(time_hrs)
    str_mins = str(time_mins)

    filename = "ts_aerosol_case0_"
    if len(str_hrs) < 2:
        filename += '0' + str_hrs
    if len(str_mins) < 2:
        filename += '0' + str_mins
    filename += ".nc"

    filepath_a = os.path.join(directory_a, filename)
    filepath_b = os.path.join(directory_b, filename)

    value_a = extract_APCEMM_value(filepath_a, quantity_to_compare)
    value_b = extract_APCEMM_value(filepath_b, quantity_to_compare)

    print(f"\nValues for {quantity_to_compare} at {str_hrs} hrs and {str_mins} mins\n")
    print(f"At run {run_number_a}: {value_a}\n")
    print(f"At run {run_number_b}: {value_b}\n")

	
if __name__ == "__main__":
	compare_APCEMM_out_runs(time_hrs=4,
                            time_mins=0,
                            run_number_a=1,
                            run_number_b=3,
                            quantity_to_compare="intOD")