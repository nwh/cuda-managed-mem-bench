#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include <cuda.h>
#include <curand.h>

// error checking
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);\
      return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);\
      return EXIT_FAILURE;}} while(0)

void setup_rng(curandGenerator_t& gen, uint64_t seed) {
  // create the rng
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  // set the rng seed
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
}

void sample_uniform_int(curandGenerator_t& gen, int min_val, int max_val, int* vec, size_t len) {
  // check min_val and max_val
  if (min_val >= max_val) {
    throw std::runtime_error("sample_uniform_int: min_val >= max_val");
  }
  // sample uniform
  CURAND_CALL(curandGenerateUniform(gen, (unsigned int*)vec, n));
  // restrict to range
  int r = max_val - min_val + 1;
  // create functor to modify random value
  struct myrange {
    const unsigned int range;
    const int min_val;
    myrange(unsigned int _range, int _min_val):
        range(_range), min_val(_min_val){}
    __host__ __device__
    int operator()(const ) const {

    }
  }
}

template<typename T>
void print_host_vector(const thrust::host_vector<T>& h_vec) {
  size_t len = h_vec.size();
  std::cout << "[";
  for (size_t i=0; i<len; i++) {
    std::cout << h_vec[i] << ", ";
  }
  std::cout << "]" << std::endl;
}

int mymod(int x) {
  return x % 10;
}

int main() {
  using std::cout;
  using std::endl;

  // generate vector of random integers on device
  int n = 10;
  thrust::host_vector<int> h_vec(n);
  std::generate(h_vec.begin(), h_vec.end(), rand);
  std::transform(h_vec.begin(), h_vec.end(), h_vec.begin(), mymod);
  print_host_vector(h_vec);
  
  // use cub to sort integers
  
  cout << "--- end of mycub ---" << endl;
  
  return 0;
}
