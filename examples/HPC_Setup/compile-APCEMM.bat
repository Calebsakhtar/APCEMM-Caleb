# RUN THIS FROM THE APCEMM directory
module purge
module load gcc/11
module load python/3.8
module load cmake/latest
mkdir build
cd build
cmake ../Code.v05-00/
cmake --build .
