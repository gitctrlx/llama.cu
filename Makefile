NVCC = nvcc

all: llama

.PHONY: llama
llama:
	$(NVCC) -DUSE_CUBLAS=1 -g -o llama llama.cu -lm -lcublas

.PHONY: clean
clean:
	rm -f llama
