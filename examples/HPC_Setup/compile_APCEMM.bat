module purge
module load git-2.20.1-gcc-5.4.0-p7ladnf
module load gcc/11
module load python/3.8
module load cmake/latest
git clone https://github.com/Calebsakhtar/APCEMM.git
cd APCEMM
git pull
git checkout ca525/add-debug-example
git submodule update --init --recursive 
mkdir build
cd build
cmake ../Code.v05-00/
