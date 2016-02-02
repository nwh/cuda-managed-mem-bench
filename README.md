# G4CU Sorting Benchmark

The purpose of this benchmark is to determine how long it will take to perform a
sort and run length encode using CUB and Thrust.  If this time is small enough,
we may be able to use this technique to mitigate thread divergence issues on the
GPU.

## Overall process

1. generate random integers in a range using CURAND and Thrust
2. compute a uniform sequence in another array [0, 1, 2, 3, ...]
3. sort pairs from (1) and (2)
4. perform run length encode on sorted keys from (1)
5. move number of segments and segment counts back to host

Note: we need to separately benchmark the RNG, because we are only interested in
the cost of the sort and run length encode.

## Input parameters

* rng seed
* length of array
* number of samples
* number of segments

## 2016-02-02 Experiments (workstation)

Compile time:

```
$ time make mybench
nvcc -gencode arch=compute_35,code=sm_35 -o mybench mybench.cu -I./cub-1.5.1 -lcurand

real	0m34.425s
user	0m33.612s
sys	0m0.776s
$ time make mybench_managed
nvcc -gencode arch=compute_35,code=sm_35 -o mybench_managed mybench_managed.cu -I./cub-1.5.1 -lcurand

real	0m34.849s
user	0m34.032s
sys	0m0.792s
```

Run without any called to `cudaMallocManaged`:

```
$ ./mybench
Thrust v1.8
seed: 1235
num_items: 131072
num_samples: 5000
num_segments: 9
rng_time: 2.51756e-05
cub_time: 0.000513044
thrust_time: 0.00138256
--- end of my bench ---
```

Run with calls to `cudaMallocManaged`:

```
$ ./mybench_managed 
Thrust v1.8
seed: 1235
num_items: 131072
num_samples: 5000
num_segments: 9
rng_time: 2.5168e-05
rng_managed_time: 0.000119185
cub_time: 0.000919785
cub_managed_time: 0.000920232
thrust_time: 0.0130265
--- end of my bench ---
```

## 2016-02-02 Experiments (icme-gpu1)

```
$ ./mybench
Thrust v1.8
seed: 1235
num_items: 131072
num_samples: 5000
num_segments: 9
rng_time: 2.4925e-05
cub_time: 0.000443189
thrust_time: 0.00161056
--- end of my bench ---
$ make mybench_managed
nvcc -gencode arch=compute_35,code=sm_35 -o mybench_managed mybench_managed.cu -I./cub-1.5.1 -lcurand
$ ./mybench_managed 
Thrust v1.8
seed: 1235
num_items: 131072
num_samples: 5000
num_segments: 9
rng_time: 2.4205e-05
rng_managed_time: 0.000207982
cub_time: 0.00118513
cub_managed_time: 0.00118606
thrust_time: 0.0165671
--- end of my bench ---
$ 
```

## 2016-02-02 Update

All code seems to run slower after using `cudaMallocManaged`!

```
$ ./mybench_managed 
Thrust v1.8
seed: 1235
num_items: 131072
num_samples: 5000
num_segments: 9
rng_time(1): 2.51796e-05
rng_managed_time: 0.000119629
rng_time(2): 0.000120445
cub_time: 0.000986438
cub_managed_time: 0.000923436
thrust_time: 0.0151882
--- end of my bench ---
```

Here: `rng_time(1)` and `rng_time(2)` are the same function run before and after
`rng_managed_time` respectively.
