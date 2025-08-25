# Source files that need to be compiled
set(SRCS
    FreeCoordBoxGrid.cpp
    RemappingFunctions.cpp
    )

# This command ensures the static library gets built
add_library(LAGRID STATIC ${SRCS})

target_link_libraries(Core PRIVATE netCDF::netcdf netCDF::netcdf-cxx4)

# This command defines the dependencies of libFDM_ANDS.a
target_link_libraries(LAGRID OpenMP::OpenMP_CXX)
