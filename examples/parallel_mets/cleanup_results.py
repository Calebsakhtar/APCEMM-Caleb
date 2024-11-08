import sys
import os

if __name__ == "__main__":
    date = "11-08"

    os.system(f"mkdir -p {date}/APCEMM/raw/")

    for file in sorted(os.listdir()):
        if(file.startswith('sweep')):
            casename = file
            os.system(f"mv {file}/APCEMM_out {file}/{casename}")
            os.system(f"cp -r {file}/{casename}/ {date}/APCEMM/raw/")

