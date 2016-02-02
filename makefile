NVCC := nvcc
NVCCFLAGS := -gencode arch=compute_30,code=sm_30
NVCCLIBS := -I./cub-1.5.1 -lcurand

mycub: mycub.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLIBS)

mybench: mybench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLIBS)

.PHONY: run_mycub
run_mycub: mycub
	./mycub

.PHONY: clean
clean:
	$(RM) mycub
	$(RM) mythrust
	$(RM) mybench
