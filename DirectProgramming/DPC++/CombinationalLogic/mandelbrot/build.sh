#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPC++ - Combinational Logic Mandelbrot sample
mkdir build
cd build
cmake ..
make
