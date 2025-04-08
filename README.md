This repository contains the code for Luca Strässle's master's thesis [Tensor Program Optimization with Auto-Batching](https://www.research-collection.ethz.ch/handle/20.500.11850/730387)

# Getting Started
```
conda create -n AutoPEFT python=3.12
conda activate AutoPEFT
pip install -r requirements.txt
```

## PEFT installation
```
cd
git clone -b v0.15.1 https://github.com/huggingface/peft.git
cd peft
pip install -e .
```

## TVM Installation (Nvidia GPU)
```
conda install -c conda-forge -c anaconda "llvmdev==19.1.4" "cmake==3.31.1" git libxml2
cd
git clone --recursive -b v0.18.0 https://github.com/apache/tvm tvm
export LD_LIBRARY_PATH=$(conda info --base)/envs/AutoPEFT/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$(conda info --base)/envs/AutoPEFT/lib:$LIBRARY_PATH
export CPATH=$(conda info --base)/envs/AutoPEFT/include:$CPATH
cd tvm
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
```
Some lines in `config.cmake` have to be checked and potentially modified. Open it with `vim config.cmake` and make sure the following are set correctly:
```
set(USE_LLVM "llvm-config --ignore-libllvm --link-static")
set(USE_CUDA ON)
set(USE_METAL OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)
set(USE_CUBLAS ON)
set(USE_CUDNN ON)
set(USE_CUTLASS ON)
```
Now continue with the build:
```
cmake -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc -DCMAKE_PREFIX_PATH=$(conda info --base)/envs/AutoPEFT -DLIBXML2_LIBRARIES=$(conda info --base)/envs/AutoPEFT/lib/libxml2.so .. && cmake --build . --parallel $(nproc)
export TVM_LIBRARY_PATH=/home/<user>/tvm/build
cd ../python
pip install -e .
```

# Repository Structure
The repository contains the following folders:
- **huggingface**: contains the implementations using HuggingFace's `transformers` and `peft` libraries
- **init_peft_weights**: contains code to randomly generate LoRA weights
- **prompts**: contains the default prompt (512 tokens) that we used for our experiments
- **tvm**: contains the implementations using TVM

The **huggingface** and **tvm** folders contain READMEs with detailed execution instructions.
