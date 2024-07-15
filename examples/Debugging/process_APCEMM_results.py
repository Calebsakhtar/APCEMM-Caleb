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

def process_and_save_outputs(filepath = "outputs/APCEMM-test-outputs.csv"):
    directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    op_filepath = os.path.join(directory, filepath)
    directory = os.path.join(directory, "APCEMM_out/")

    # Initialise empty lists for the outputs of interest
    t_hrs = []
    width_m = []
    depth_m = []
    tau = []
    tau_int = []
    I = []
    N = []

    for file in sorted(os.listdir(directory)):
        if(file.startswith('ts_aerosol') and file.endswith('.nc')):
            file_path = os.path.join(directory,file)
            ds = xr.open_dataset(file_path, engine = "netcdf4", decode_times = False)

            # Calculate the time since formation from the filename
            tokens = file_path.split('.')
            mins = int(tokens[-2][-2:])
            hrs = int(tokens[-2][-4:-2])
            t_hrs.append(hrs + mins/60)

            # Save the variables that do not lie on a grid
            width_m.append(ds["width"].values[0])
            depth_m.append(ds["depth"].values[0])
            I.append(ds["Ice Mass"].values[0])
            N.append(ds["Number Ice Particles"].values[0])
            tau_int.append(ds["intOD"].values[0])

            # Calculate the average grid cell dimensions, and the number of grid cells
            x = ds["x"].values
            y = ds["y"].values
            dx = abs(x[-1] - x[0]) / len(x)
            dy = abs(y[-1] - y[0]) / len(y)

            # Extract the vertically integrated optical depth
            tau_vert = ds["Vertical optical depth"].values # It's a function of x

            # Populate the optical depth grid by undoing the integration
            tau_grid = np.zeros( (len(y), len(x) ) )
            for i in range(len(x)):
                tau_avg = tau_vert[i] / dy / len(y)
                tau_grid[:,i] = tau_avg

            # Integrate the optical depth over both x and y to calculate the average value
            tau_integ = 0
            tau_squared_integ = 0

            for i in range(len(x)):
                for j in range(len(y)):
                    tau_integ += tau_grid[j, i] * dx * dy
                    tau_squared_integ += tau_grid[j, i] ** 2 * dx * dy

            if tau_integ == 0:
                tau.append(0)
            else:
                tau.append(tau_squared_integ / tau_integ)

    t_hrs = np.array(t_hrs)
    width_m = np.array(width_m)
    depth_m = np.array(depth_m)
    tau = np.array(tau)
    tau_int = np.array(tau_int)
    I = np.array(I)
    N = np.array(N)

    data = {
        "Time Since Formation, h": t_hrs,
        "N, # / m": N,
        "Optical Depth, ---": tau,
        "Integrated Optical Depth, m^2": tau_int,
        "I, kg of ice / m": I,
        "Extinction defined width, m": width_m,
        "Extinction defined depth, m": depth_m,
    }

    DF = pd.DataFrame.from_dict(data)
    DF.to_csv(op_filepath)

    plt.plot(t_hrs, tau_int)
    plt.xlabel("Time, hrs")
    plt.ylabel("intOD, m^2")
    plt.savefig("outputs/intOD" + ".png", format = "png", dpi = 300, bbox_inches = "tight")
    plt.close()


    plt.plot(t_hrs, width_m)
    plt.xlabel("Time, hrs")
    plt.ylabel("Width, m")
    plt.savefig("outputs/width" + ".png", format = "png", dpi = 300, bbox_inches = "tight")
    plt.close()


    plt.plot(t_hrs, depth_m)
    plt.xlabel("Time, hrs")
    plt.ylabel("Depth, m")
    plt.savefig("outputs/depth" + ".png", format = "png", dpi = 300, bbox_inches = "tight")
    plt.close()

    return data
	
if __name__ == "__main__":
	process_and_save_outputs()