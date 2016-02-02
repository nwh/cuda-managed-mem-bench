mycub: mycub.cu
	nvcc -gencode arch=compute_30,code=sm_30 $^ -o $@ -lcurand \
		-I./cub-1.5.1

mybench: mybench.cu
	nvcc -gencode arch=compute_30,code=sm_30 $^ -o $@ -lcurand \
		-I./cub-1.5.1

mygtod: mygtod.cu
	nvcc -gencode arch=compute_30,code=sm_30 $^ -o $@

mythrust: mythrust.cu
	nvcc -gencode arch=compute_30,code=sm_30 $^ -o $@

.PHONY: run_mycub
run_mycub: mycub
	./mycub

.PHONY: run_thrust
run_mythrust: mythrust
	./mythrust

.PHONY: clean
clean:
	$(RM) mycub
	$(RM) mythrust
	$(RM) mybench
