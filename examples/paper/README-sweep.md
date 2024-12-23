## Overview
This example allows for the running of APCEMM (`v1.2.0`). There have been some modifications to the repo. Please see `MODIFICATIONS.pdf`

## Getting Started
### Dependencies

For the dependencies, see `.devcontainer/Dockerfile.apcemm` and the APCEMM repo `README.md`. The utilization of VS Code with Docker is recommended.

### Running the code

1.  Run Docker Desktop
2.  Activate the environment in VS Code
3.  Open the root directory `APCEMM` in VS Code
4.  Click on the two blue arrows on the bottom left corner, then select `Reopen in Container`
5.  Compile APCEMM according to the instructions on `README.md` in the root directory
6.  In the VS Code container terminal, navigate to `examples/metsweep` 
7.  Change the date in `cleanup_results.py`
8.  Put your meteorology into `examples/metsweep/inputs/` 
9.  Change the APCEMM config in `examples/metsweep/sample_rundir/input.yaml`. This will be the same for all runs
10. Run APCEMM from the VS Code container terminal in `examples/metsweep`
```
python3 run_APCEMM_local.py && python3 cleanup_results.py
```
11. Collect your results from the new folder `examples/metsweep/<chosen_date>`