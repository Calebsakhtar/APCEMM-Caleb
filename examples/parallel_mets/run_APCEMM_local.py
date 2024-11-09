import sys
import os

if __name__ == "__main__":
    ipdir = "inputs/"
    for file in sorted(os.listdir(ipdir)):
        if os.fsdecode(file).endswith('.nc'):
            casename = file.split("-met")[0]

            os.system(f"cp -r sample_rundir/ {casename}")
            os.system(f"cp {ipdir} {casename}/met.nc")
            os.system(f"cd {casename} && ./../../../build/APCEMM input-local.yaml")
