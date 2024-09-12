git submodule update --init --recursive
sudo apt install build-essential
sudo apt-get install curl zip unzip tar
mkdir build

# START A NEW TERMINAL IF IT DOES NOT WORK
cd build
cmake ../Code.v05-00/
cmake --build .