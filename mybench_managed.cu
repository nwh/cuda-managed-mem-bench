// mybench.cu  benchmarks for sorting related to G4CU

// std includes
#include <iostream>

// system includes
#include <sys/time.h>

// cuda includes
#include <cuda.h>
#include <curand.h>

// thrust includes
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

// cub includes
#include <cub/cub.cuh>

// typedef for timeval struct
typedef struct timeval timeval_t;

// return elapsed time between two timeval_ts
double elapsed_time(timeval_t start, timeval_t finish) {
  double start_s = (double)start.tv_sec +
    1.0e-6*(double)start.tv_usec;
  double finish_s = (double)finish.tv_sec +
    1.0e-6*(double)finish.tv_usec;
  return finish_s-start_s;
}

// set up the curand rng
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

void generate_in_range(curandGenerator_t& gen, int min_val, int max_val, int* a, size_t n) {
  // generate random integers
  curandGenerate(gen, (unsigned int*)a, n);
  // map to range
  int range = max_val-min_val;
  thrust::device_ptr<int> a_ptr = thrust::device_pointer_cast(a);
  thrust::transform(a_ptr,a_ptr+n,a_ptr,range_op(range,min_val));
}

double bench_rng(const unsigned long long seed,
                 const size_t num_items,
                 const int num_samples,
                 const int num_segments) {
  // set up the rng
  curandGenerator_t gen;
  setup_rng(gen,seed);
  // allocate buffer
  int* keys = NULL;
  cudaMalloc(&keys,num_items*sizeof(*keys));
  timeval_t start;
  // start timing the loop
  gettimeofday(&start,NULL);
  for (int samp = 0; samp < num_samples; samp++) {
    generate_in_range(gen,0,num_segments,keys,num_items);
  }
  timeval_t finish;
  gettimeofday(&finish,NULL);
  // cleanup the rng
  curandDestroyGenerator(gen);
  // free memory
  cudaFree(keys);
  // compute elapsed time and return the average time
  double eltm = elapsed_time(start,finish);
  return eltm / num_samples;
}

double bench_rng_managed(const unsigned long long seed,
                         const size_t num_items,
                         const int num_samples,
                         const int num_segments) {
  // set up the rng
  curandGenerator_t gen;
  setup_rng(gen,seed);
  // allocate buffer
  int* keys = NULL;
  cudaMallocManaged(&keys,num_items*sizeof(*keys));
  timeval_t start;
  // start timing the loop
  gettimeofday(&start,NULL);
  for (int samp = 0; samp < num_samples; samp++) {
    generate_in_range(gen,0,num_segments,keys,num_items);
  }
  timeval_t finish;
  gettimeofday(&finish,NULL);
  // cleanup the rng
  curandDestroyGenerator(gen);
  // free memory
  cudaFree(keys);
  // compute elapsed time and return the average time
  double eltm = elapsed_time(start,finish);
  return eltm / num_samples;
}

double bench_cub(const unsigned long long seed,
                 const size_t num_items,
                 const int num_samples,
                 const int num_segments) {
  // set up the rng
  curandGenerator_t gen;
  setup_rng(gen,seed);

  // allocate buffers
  int* keys = NULL; // process index in context of G4CU
  int* vals = NULL; // thread id, [0, 1, 2, 3, ...] before sort by pairs
  int* keys_alt = NULL; // alternative keys buffer
  int* vals_alt = NULL; // alternative vals buffer
  int* keys_compact = NULL; // array of compacted keys after RLE
  int* keys_counts = NULL; // array of key counts after RLE
  int* keys_num = NULL; // total number of keys
  cudaMalloc(&keys,num_items*sizeof(*keys));
  cudaMalloc(&vals,num_items*sizeof(*vals));
  cudaMalloc(&keys_alt,num_items*sizeof(*keys_alt));
  cudaMalloc(&vals_alt,num_items*sizeof(*vals_alt));
  cudaMalloc(&keys_compact,num_items*sizeof(*keys_compact));
  cudaMalloc(&keys_counts,num_items*sizeof(*keys_counts));
  cudaMalloc(&keys_num,sizeof(*keys_num));
  
  // create double buffers
  cub::DoubleBuffer<int> keys_dbuf(keys,keys_alt);
  cub::DoubleBuffer<int> vals_dbuf(vals,vals_alt);

  // allocate temp storage for sort
  void* temp_sort = NULL;
  size_t temp_sort_bytes = 0;
  cub::DeviceRadixSort::SortPairs(temp_sort, temp_sort_bytes, keys_dbuf,
                                  vals_dbuf, num_items);
  cudaMalloc(&temp_sort, temp_sort_bytes);

  // allocate temp storage for run length encode
  void* temp_rle = NULL;
  size_t temp_rle_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(temp_rle, temp_rle_bytes, keys_dbuf.Current(),
                                     keys_compact, keys_counts, keys_num, num_items);
  cudaMalloc(&temp_rle,temp_rle_bytes);
  
  // thrust device pointers
  thrust::device_ptr<int> vals_ptr = thrust::device_pointer_cast(vals);
  thrust::device_ptr<int> keys_compact_ptr = thrust::device_pointer_cast(keys_compact);
  thrust::device_ptr<int> keys_counts_ptr = thrust::device_pointer_cast(keys_counts);
  thrust::device_ptr<int> keys_num_ptr = thrust::device_pointer_cast(keys_num);

  // host vectors for keys and counts
  thrust::host_vector<int> keys_compact_host(num_items);
  thrust::host_vector<int> keys_counts_host(num_items);

  // number of segments (host-side)
  int keys_num_host = 0;
  
  // start timing the loop
  timeval_t start;
  gettimeofday(&start,NULL);
  for (int samp = 0; samp < num_samples; samp++) {
    // generate random integers for keys
    generate_in_range(gen,0,num_segments,keys,num_items);
    // produce vals sequence
    thrust::sequence(vals_ptr,vals_ptr+num_items);
    // set current in double buffers
    keys_dbuf.selector = 0;
    vals_dbuf.selector = 0;
    // sort
    cub::DeviceRadixSort::SortPairs(temp_sort, temp_sort_bytes, keys_dbuf,
                                    vals_dbuf, num_items);
    // reduce
    cub::DeviceRunLengthEncode::Encode(temp_rle, temp_rle_bytes, keys_dbuf.Current(),
                                       keys_compact, keys_counts, keys_num, num_items);
    // read number of segments
    keys_num_host = keys_num_ptr[0];

    // copy data to host
    thrust::copy(keys_compact_ptr,keys_compact_ptr+keys_num_host,keys_compact_host.begin());
    thrust::copy(keys_counts_ptr,keys_counts_ptr+keys_num_host,keys_counts_host.begin());
  }
  timeval_t finish;
  gettimeofday(&finish,NULL);

  // cleanup the rng
  curandDestroyGenerator(gen);
  // free memory
  cudaFree(keys);
  cudaFree(vals);
  cudaFree(keys_alt);
  cudaFree(vals_alt);
  cudaFree(keys_compact);
  cudaFree(keys_counts);
  cudaFree(keys_num);
  cudaFree(temp_sort);
  cudaFree(temp_rle);
  
  // compute elapsed time and return the average time
  double eltm = elapsed_time(start,finish);
  return eltm / num_samples;
}

double bench_cub_managed(const unsigned long long seed,
                         const size_t num_items,
                         const int num_samples,
                         const int num_segments) {
  // set up the rng
  curandGenerator_t gen;
  setup_rng(gen,seed);

  // allocate buffers
  int* keys = NULL; // process index in context of G4CU
  int* vals = NULL; // thread id, [0, 1, 2, 3, ...] before sort by pairs
  int* keys_alt = NULL; // alternative keys buffer
  int* vals_alt = NULL; // alternative vals buffer
  int* keys_compact = NULL; // array of compacted keys after RLE
  int* keys_counts = NULL; // array of key counts after RLE
  int* keys_num = NULL; // total number of keys
  cudaMallocManaged(&keys,num_items*sizeof(*keys));
  cudaMallocManaged(&vals,num_items*sizeof(*vals));
  cudaMallocManaged(&keys_alt,num_items*sizeof(*keys_alt));
  cudaMallocManaged(&vals_alt,num_items*sizeof(*vals_alt));
  cudaMallocManaged(&keys_compact,num_items*sizeof(*keys_compact));
  cudaMallocManaged(&keys_counts,num_items*sizeof(*keys_counts));
  cudaMallocManaged(&keys_num,sizeof(*keys_num));
  
  // create double buffers
  cub::DoubleBuffer<int> keys_dbuf(keys,keys_alt);
  cub::DoubleBuffer<int> vals_dbuf(vals,vals_alt);

  // allocate temp storage for sort
  void* temp_sort = NULL;
  size_t temp_sort_bytes = 0;
  cub::DeviceRadixSort::SortPairs(temp_sort, temp_sort_bytes, keys_dbuf,
                                  vals_dbuf, num_items);
  cudaMallocManaged(&temp_sort, temp_sort_bytes);

  // allocate temp storage for run length encode
  void* temp_rle = NULL;
  size_t temp_rle_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(temp_rle, temp_rle_bytes, keys_dbuf.Current(),
                                     keys_compact, keys_counts, keys_num, num_items);
  cudaMallocManaged(&temp_rle,temp_rle_bytes);
  
  // thrust device pointers
  thrust::device_ptr<int> vals_ptr = thrust::device_pointer_cast(vals);
  thrust::device_ptr<int> keys_compact_ptr = thrust::device_pointer_cast(keys_compact);
  thrust::device_ptr<int> keys_counts_ptr = thrust::device_pointer_cast(keys_counts);
  thrust::device_ptr<int> keys_num_ptr = thrust::device_pointer_cast(keys_num);

  // host vectors for keys and counts
  thrust::host_vector<int> keys_compact_host(num_items);
  thrust::host_vector<int> keys_counts_host(num_items);

  // number of segments (host-side)
  int keys_num_host = 0;
  
  // start timing the loop
  timeval_t start;
  gettimeofday(&start,NULL);
  for (int samp = 0; samp < num_samples; samp++) {
    // generate random integers for keys
    generate_in_range(gen,0,num_segments,keys,num_items);
    // produce vals sequence
    thrust::sequence(vals_ptr,vals_ptr+num_items);
    // set current in double buffers
    keys_dbuf.selector = 0;
    vals_dbuf.selector = 0;
    // sort
    cub::DeviceRadixSort::SortPairs(temp_sort, temp_sort_bytes, keys_dbuf,
                                    vals_dbuf, num_items);
    // reduce
    cub::DeviceRunLengthEncode::Encode(temp_rle, temp_rle_bytes, keys_dbuf.Current(),
                                       keys_compact, keys_counts, keys_num, num_items);
    // read number of segments
    keys_num_host = keys_num_ptr[0];

    // copy data to host
    thrust::copy(keys_compact_ptr,keys_compact_ptr+keys_num_host,keys_compact_host.begin());
    thrust::copy(keys_counts_ptr,keys_counts_ptr+keys_num_host,keys_counts_host.begin());
  }
  timeval_t finish;
  gettimeofday(&finish,NULL);

  // cleanup the rng
  curandDestroyGenerator(gen);
  // free memory
  cudaFree(keys);
  cudaFree(vals);
  cudaFree(keys_alt);
  cudaFree(vals_alt);
  cudaFree(keys_compact);
  cudaFree(keys_counts);
  cudaFree(keys_num);
  cudaFree(temp_sort);
  cudaFree(temp_rle);
  
  // compute elapsed time and return the average time
  double eltm = elapsed_time(start,finish);
  return eltm / num_samples;
}

double bench_thrust(const unsigned long long seed,
                    const size_t num_items,
                    const int num_samples,
                    const int num_segments) {
  // set up the rng
  curandGenerator_t gen;
  setup_rng(gen,seed);

  // device vectors
  thrust::device_vector<int> keys(num_items);
  thrust::device_vector<int> vals(num_items);
  thrust::device_vector<int> keys_compact(num_items);
  thrust::device_vector<int> keys_counts(num_items);

  // host vectors
  thrust::host_vector<int> keys_compact_host(num_items);
  thrust::host_vector<int> keys_counts_host(num_items);
  
  // raw pointers
  int* keys_raw = thrust::raw_pointer_cast(keys.data());

  // number of compacted keys
  size_t keys_num_host = 0;
  
  // start timing the loop
  timeval_t start;
  gettimeofday(&start,NULL);
  for (int samp = 0; samp < num_samples; samp++) {
    // generate random integers for keys
    generate_in_range(gen,0,num_segments,keys_raw,num_items);
    // produce vals sequence
    thrust::sequence(vals.begin(),vals.end());
    // sort
    thrust::sort_by_key(keys.begin(),keys.end(),vals.begin());
    // reduce
    keys_num_host = thrust::reduce_by_key(
        keys.begin(),
        keys.end(),
        thrust::make_constant_iterator(1),
        keys_compact.begin(),
        keys_counts.begin()).first - keys_compact.begin();

    // copy data to host
    thrust::copy(keys_compact.begin(),keys_compact.begin() + keys_num_host, keys_compact_host.begin());
    thrust::copy(keys_counts.begin(),keys_counts.begin() + keys_num_host, keys_counts_host.begin());
  }
  timeval_t finish;
  gettimeofday(&finish,NULL);

  // cleanup the rng
  curandDestroyGenerator(gen);
  
  // compute elapsed time and return the average time
  double eltm = elapsed_time(start,finish);
  return eltm / num_samples;
}

int main() {
  using std::cout;
  using std::endl;

  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;

  cout << "Thrust v" << major << "." << minor << endl;

  unsigned long long seed = 1235;
  int num_items = 1024*128;
  int num_samples = 5000;
  int num_segments = 9;

  cout << "seed: " << seed << endl;
  cout << "num_items: " << num_items << endl;
  cout << "num_samples: " << num_samples << endl;
  cout << "num_segments: " << num_segments << endl;
  
  double rng_time = bench_rng(seed,num_items,num_samples,num_segments);
  cout << "rng_time: " << rng_time << endl;

  double rng_managed_time = bench_rng_managed(seed,num_items,num_samples,num_segments);
  cout << "rng_managed_time: " << rng_managed_time << endl;
  
  double cub_time = bench_cub(seed,num_items,num_samples,num_segments);
  cout << "cub_time: " << cub_time << endl;

  double cub_managed_time = bench_cub_managed(seed,num_items,num_samples,num_segments);
  cout << "cub_managed_time: " << cub_managed_time << endl;
  
  double thrust_time = bench_thrust(seed,num_items,num_samples,num_segments);
  cout << "thrust_time: " << thrust_time << endl;
  
  cout << "--- end of my bench ---" << endl;
  cudaDeviceReset();
}
