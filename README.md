** Aún no se ha testeado nada ** 

Se han analizado los repos de trisycl (https://github.com/trisycl/trisycl) y oneAPI DPC++ (https://github.com/intel/llvm) compiler de Intel

En principio si sería posible utilizar oneAPI FPGA para la PYNQ-Z2 siguiendo los siguientes pasos:

Habría que usar Intel oneAPI DPC++ compiler combiando con un Xilinx/AMD FPGA Support Package (mirar https://github.com/codeplaysoftware/deploy-oneapi). Esto último se trata de unos plugins que nos permitiría correr onAPI en NVIDA (CUDA) y hardware AMD.

Para la implementación, el ARM de la pynq no sería capaz de compilarlo, por lo que nos haría falta configurar un cross-compiler.

TESTEO




Nota: trisycl lleva sin recibir actualizaciones significativas desde finales de 2023.
