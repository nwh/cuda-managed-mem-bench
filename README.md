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

