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

### SYCL-FPGA Emulator version
```bash
mkdir build-fpga-emu && cd build-fpga-emu
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/icpx-fpga.cmake \
  -DUSE_FPGA=ON \
  -DFPGA_MODE=emulator
make
```
### SYCL-FPGA Simulator version
```bash
mkdir build-fpga-sim && cd build-fpga-sim
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/icpx-fpga.cmake \
  -DUSE_FPGA=ON \
  -DFPGA_MODE=simulation
make
```

### SYCL-FPGA HW Report version
```bash
mkdir build-fpga-report && cd build-fpga-report
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/icpx-fpga.cmake \
  -DUSE_FPGA=ON \
  -DFPGA_MODE=report
make
```

**Aún no se ha testeado nada** 


Repositorios y links a tener en cuenta para usar SYCL en el ecosistema Xilinx:
- trisycl (https://github.com/trisycl/trisycl) 
- Intel oneAPI DPC++ (https://github.com/intel/llvm)
- XRT (https://xilinx.github.io/XRT/2024.2/html/platforms.html)

Al final no parace posible que se pueda utilizar SYCL para la PYNQ-Z2, ya que XRT no le da soporte (solo se lo da a ZYNQ-7000, ZYNQ Ultrascale+ MPSoC y  Versal ACAP)

Para la implementación, el chip de la FPGA no sería suficiente para compilar, por lo que nos haría falta configurar un cross-compiler desde otra maquina. Luego cargar el bitstream y el ejecutable en un sd a la FPGA.

Alternativa de alto nivel para desarrollo en FPGA en el ecosistema Xilinx sería utilizar Vitis HLS.



Nota: trisycl lleva sin recibir actualizaciones significativas desde finales de 2023.
