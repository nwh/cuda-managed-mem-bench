NVCC := nvcc
NVCCFLAGS := -gencode arch=compute_35,code=sm_35
NVCCLIBS := -I./cub-1.5.1 -lcurand

.PHONY: all
all: mybench mycub mybench_managed

mycub: mycub.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLIBS)

mybench: mybench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLIBS)

mybench_managed: mybench_managed.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLIBS)

.PHONY: get-cub
get-cub:
	wget https://github.com/NVlabs/cub/archive/1.5.1.zip
	unzip 1.5.1.zip

.PHONY: clean
clean:
	$(RM) mycub
	$(RM) mybench
	$(RM) mybench_managed
