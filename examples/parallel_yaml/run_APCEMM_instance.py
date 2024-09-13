import sys
import os

if __name__ == "__main__":
    inputpath = sys.argv[1]
    casename = inputpath.split('/')[-1]
    casename = casename.split(".")[0]

    os.system(f"cp -r sample_rundir {casename}")
    os.system(f"cp inputpath {casename}/input.yaml")
    os.system(f"cd {casename} && ./../../../build/APCEMM input.yaml")
