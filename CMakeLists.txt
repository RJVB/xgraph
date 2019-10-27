cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(xgraph LANGUAGES C CXX)

option(BUILD_XGTEST "build a test curves generator and the example dymod" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules ${CMAKE_MODULE_PATH} )

include(FeatureSummary)
include(FindPkgConfig)
include(GetGitRevisionDescription)
include(AddDymods)

find_package(X11)
pkg_check_modules(PKG_FFTW3 fftw3)
mark_as_advanced(PKG_FFTW3_INCLUDEDIR)

set_package_properties(PKG_FFTW3 PROPERTIES
                      DESCRIPTION "FFTW3 Fourier Transform library"
                      TYPE OPTIONAL
                      URL "http://www.fftw.org/")


include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR})

add_definitions("-DXGRAPH")
if (X11_Xinerama_FOUND)
    add_definitions("-D_HAVE_XINERAMA_")
endif()
add_definitions("\"-DC_COMMAND=${CMAKE_C_COMPILER} ${CMAKE_C_FLAGS}\" \"-DCXX_COMMAND=${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS}\"")

add_subdirectory(ux11)
add_subdirectory(xtb)

execute_process(COMMAND
        uname
        -nmrs
    WORKING_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE
        RES
    OUTPUT_VARIABLE
        UNAME
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT RES EQUAL 0)
    message(STATUS "uname -nmrs returned ${RES}, \"${UNAME}\"")
    set(UNAME "(unknown platform)")
endif()
configure_file(buildplatform.h.in ${CMAKE_CURRENT_BINARY_DIR}/buildplatform.h)
execute_process(COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_cycles_per_second
    OUTPUT_FILE
        ${CMAKE_CURRENT_BINARY_DIR}/cpu_cycles_per_second.h)

set(LIBXGRAPH_SRCS
    ascanfc-table.c
    ascanfc3.c
    ascanfcSS.c
    ascanfc.c
    ascanfc2.c
    ascanfcMap2.c
    vscanf/asscanf.c

    ascanfcMap.cpp
    arrayvops.cpp

    xgX.c
    hard_devices.c
    dialog.c
    dialog_s.c
    new_ps.c
    matherr.c
    SS.c
    dymod.c
    xgPen.c
    ReadData.c
    xgInput.c
    LegendsNLabels.c
    alloca.c
    fascanf.c
    regex.c
    lowlevel_timer.c
)

add_library(libxgraph STATIC ${LIBXGRAPH_SRCS})

add_executable(xgraph main.c xgraph.c xgsupport.c)
target_link_libraries(xgraph
    libxgraph
    libux11
    libxtb
    X11::Xinerama
    X11::Xext
    X11::X11
    ${CMAKE_DL_LIBS})
if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Linux")
    target_link_libraries(xgraph
        -lm -lrt)
endif()

install(TARGETS xgraph
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if (BUILD_XGTEST)
    add_executable(xgtest xgtest.c)
    if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Linux")
        target_link_libraries(xgtest
            -lm)
    endif()
endif()

### DyMods plugins:

if (BUILD_XGTEST)
    add_dymod(dm_example.c)
endif()

add_dymods(constants.c strings.c CMaps.c)

add_dymods(contrib/splines.c
    contrib/simanneal.c
    contrib/integrators.c
    contrib/ddeltaNEC.c
    contrib/fig_dist.c)
add_dymod(contrib/pearson_correl.c DYMOD_NAME pearson_correlation)
add_dymod(contrib/fourconv3.c arrayvops.cpp DYMOD_NAME fourconv3)
add_dymod(contrib/fourconv3.c arrayvops.cpp DYMOD_NAME fourconv3f DEFINITIONS FFTW_SINGLE)

add_dymods(Import/GSRio.c Import/IEFio.c Import/CSVio.c)

add_subdirectory(Python)

FEATURE_SUMMARY(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)
