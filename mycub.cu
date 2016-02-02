#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include <cuda.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <cub/cub.cuh>

void setup_rng(curandGenerator_t& gen, uint64_t seed) {
  // create the rng
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // set the rng seed
  curandSetPseudoRandomGeneratorSeed(gen, seed);
}

struct range_op: public thrust::unary_function<int,int> {
  const unsigned int range;
  const int min_val;
  __host__ __device__
  range_op(unsigned int _range, int _min_val):
      range(_range), min_val(_min_val) {}
  __host__ __device__
  int operator()(int x) {
    unsigned int y = (unsigned int)x;
    return ((int) (y % range)) + min_val;
  }
};

void map_to_range(int min_val, int max_val, int* a, size_t n) {
  // functor to perform the map
  int range = max_val-min_val;
  thrust::device_ptr<int> a_ptr = thrust::device_pointer_cast(a);
  thrust::transform(a_ptr,a_ptr+n,a_ptr,range_op(range,min_val));
}

int main() {
  using std::cout;
  using std::endl;

  // settings
  size_t num_items = 20;
  
  // allocate memory on device
  int* keys = NULL;
  int* vals = NULL;
  cudaMalloc(&keys,num_items*sizeof(*keys));
  cudaMalloc(&vals,num_items*sizeof(*vals));

  // setup random number generator
  curandGenerator_t gen;
  setup_rng(gen,1010ULL);

  // generate random numbers & map to a range
  curandGenerate(gen, (unsigned int*)keys, num_items);
  map_to_range(0,5,keys,num_items);

  // sequence the values
  thrust::device_ptr<int> vals_ptr = thrust::device_pointer_cast(vals);
  thrust::sequence(vals_ptr,vals_ptr+num_items);
  
  // prepare for cub
  /// allocate output buffers
  int* keys_alt = NULL;
  int* vals_alt = NULL;
  cudaMalloc(&keys_alt,num_items*sizeof(*keys_alt));
  cudaMalloc(&vals_alt,num_items*sizeof(*vals_alt));
  /// create double buffers
  cub::DoubleBuffer<int> keys_dbuf(keys,keys_alt);
  cub::DoubleBuffer<int> vals_dbuf(vals,vals_alt);
  
  /// allocate temporary storage
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_dbuf,
                                  vals_dbuf, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // sort with cub
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_dbuf,
                                  vals_dbuf, num_items);
  
  //thrust::sort(thrust::device,a,a+n);
  
  // create device ptr and inspect
  thrust::device_ptr<int> keys_ptr = thrust::device_pointer_cast(keys_dbuf.Current());
  vals_ptr = thrust::device_pointer_cast(vals_dbuf.Current());
  for (size_t i=0; i<num_items; i++) {
    cout << i << ": key = " << keys_ptr[i] << ", val = " << vals_ptr[i] << endl;
  }

  // run length encode
  /// compacted keys
  int* keys_cmp = NULL; cudaMalloc(&keys_cmp,num_items*sizeof(*keys_cmp));
  /// count of keys
  int* keys_cnt = NULL; cudaMalloc(&keys_cnt,num_items*sizeof(*keys_cnt));
  /// number of computed segments
  int* num_segments = NULL; cudaMalloc(&num_segments,sizeof(int));
  /// temporary storage
  void* rle_temp_storage = NULL;
  size_t rle_storage_bytes = 0;
  cub::DeviceReduce::RunLengthEncode(rle_temp_storage, rle_storage_bytes, keys_dbuf.Current(),
                                     keys_cmp, keys_cnt, num_segments, num_items);
  cudaMalloc(&rle_temp_storage,rle_storage_bytes);
  /// perform rle
  cub::DeviceReduce::RunLengthEncode(rle_temp_storage, rle_storage_bytes, keys_dbuf.Current(),
                                     keys_cmp, keys_cnt, num_segments, num_items);

  // look at results
  cout << "run length encode..." << endl;
  thrust::device_ptr<int> cmp_ptr = thrust::device_pointer_cast(keys_cmp);
  thrust::device_ptr<int> cnt_ptr = thrust::device_pointer_cast(keys_cnt);
  thrust::device_ptr<int> seg_ptr = thrust::device_pointer_cast(num_segments);
  for (size_t i = 0; i < seg_ptr[0]; i++) {
    cout << i << ": key = " << cmp_ptr[i] << ", cnt = " << cnt_ptr[i] << endl;
  }
  
  return 0;
}
