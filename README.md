# gpuFoam - Tools and extensions for GPU-accelerated OpenFOAM simulations

## Overview
* gpu_chemistry - Gpu accelerated detailed chemistry model

For more details check the README section in each of the subfolders listed above.

## Requirements
* C++17 supporting host compiler
* Either Nvidia [cuda compiler](https://developer.nvidia.com/hpc-sdk) (nvcc) version 10+
* Or AMD [hip compiler](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/install.html) (hipcc) version 5.6+
* Latest [OpenFOAM foundation development release](https://openfoam.org/version/dev/)

## Compilation
Everything in this folder should compile with the Allwmake script. Before running it, ensure that you have a valid OpenFOAM installation and compilers available.

* #### For Nvidia

    To check that you have valid compilers for the Nvidia backend
    ```
    foamRun -help
    g++ --version
    nvcc --version
    ```

    Then check the name of your graphics card (for example with nvidia-smi) on and go to [this page](https://developer.nvidia.com/cuda-gpus) to see which compute capability it has. Modify the file ```./nvcpp``` accordingly if necessary.



    Finally, run the commands
    ```
    export GPUFOAM_NVIDIA_BACKEND=1
    ./Allwmake
    ```

* #### For AMD

    To check that you have valid compilers for the AMD backend
    ```
    foamRun -help
    g++ --version
    hipcc --version
    ```
    Then check the name of your graphics card and go to [this page](https://llvm.org/docs/AMDGPUUsage.html#processors) to see the matching architecture keyword. Modify the file ```./hipcc``` if necessary.

    Finally, run the commands
    ```
    export GPUFOAM_AMD_BACKEND=1
    ./Allwmake
    ```



## Credits
This project contains source code taken from the following projects

* [mdspan](https://github.com/kokkos/mdspan)
* [variant](https://github.com/bryancatanzaro/variant)

The licenses of the external projects can be found from their respective folders.