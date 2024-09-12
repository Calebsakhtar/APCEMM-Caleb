# RUN THIS FROM THE APCEMM directory
git clone https://github.com/Calebsakhtar/APCEMM.git
cd APCEMM
git pull
git checkout ca525/perform-comparison
git submodule update --init --recursive 
mkdir build
cd build
cmake ../Code.v05-00/

module purge
module load gcc/11
module load python/3.8
module load cmake/latest
cmake ../Code.v05-00/
cmake --build .
