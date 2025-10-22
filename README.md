# llama.cu

A pure CUDA implementation of the LLaMA model for high-performance inference and educational purposes. Supports LLaMA 1, 2, and 3 architectures.

This repository demonstrates how to run LLaMA inference using CUDA C++, making it ideal for learning GPU acceleration techniques and understanding transformer internals with minimal dependencies.

## Features

- **Pure CUDA Implementation** – Direct CUDA kernels for maximum performance without heavy ML frameworks
- **Optimized Matrix Operations** – Custom CUDA kernels for matrix multiplication and attention mechanisms
- **Memory Efficient** – Optimized memory access patterns for GPU cache efficiency
- **Minimal Dependencies** – Standalone CUDA implementation without PyTorch or TensorFlow
- **Educational** – Clean, readable CUDA code with inline documentation for learning GPU programming

## Usage

```sh
make
./llama stories15M.bin
```

The examples use small models trained by [Andrej Karpathy](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models) for demonstration.

## Building

Requires NVIDIA CUDA Toolkit (11.0 or later):

```sh
make
```

Or using CMake:

```sh
mkdir build && cd build
cmake ..
make
```

## Related Work

If you're interested in LLaMA implementations in other languages:

- **[llama.go](https://github.com/gitctrlx/llama.go)** – Pure Go implementation
- **[llama.np](https://github.com/gitctrlx/llama.np)** – NumPy-based implementation

## Acknowledgments

Inspired by [llama2.c](https://github.com/karpathy/llama2.c), [llama3.cuda](https://github.com/likejazz/llama3.cuda) and the broader LLaMA community. This project aims to provide a GPU-accelerated alternative for educational purposes.

## License

MIT
