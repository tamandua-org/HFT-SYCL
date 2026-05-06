In order to compile for each different setup, (from root project directory)

### CPU-only unoptimized version
```bash
mkdir build-gcc && cd build-gcc
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/gcc.cmake
make
```
### SYCL-Host Device version
```bash
mkdir build-sycl && cd build-sycl
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/icpx-sycl.cmake \
  -DUSE_SYCL=ON
make
```

### SYCL-FPGA version
```bash
mkdir build-fpga && cd build-fpga
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/icpx-fpga.cmake \
  -DUSE_FPGA=ON
make
```

**Aún no se ha testeado nada** 

Se han analizado los repos de trisycl (https://github.com/trisycl/trisycl) y oneAPI DPC++ (https://github.com/intel/llvm) compiler de Intel

En principio si sería posible utilizar oneAPI FPGA para la PYNQ-Z2 siguiendo los siguientes pasos:

Habría que usar Intel oneAPI DPC++ compiler combiando con un Xilinx/AMD FPGA Support Package (mirar https://github.com/codeplaysoftware/deploy-oneapi). Esto último se trata de unos plugins que nos permitiría correr onAPI en NVIDA (CUDA) y hardware AMD.

Para la implementación, el ARM de la pynq no sería capaz de compilarlo, por lo que nos haría falta configurar un cross-compiler.


Nota: trisycl lleva sin recibir actualizaciones significativas desde finales de 2023.
