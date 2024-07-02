# DO NOT module purge
# Run these commands in hpc-work
git clone https://github.com/Calebsakhtar/APCEMM.git
cd APCEMM
git pull
git checkout ca525/perform-comparison
git submodule update --init --recursive 