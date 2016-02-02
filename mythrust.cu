#include <iostream>

#include <cuda.h>

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

int main() {
  using std::cout;
  using std::endl;
  
  int* a = NULL;
  size_t n = 10;
  cudaMalloc(&a, n*sizeof(*a));

  thrust::device_ptr<int> a_ptr = thrust::device_pointer_cast(a);

  thrust::sequence(a_ptr,a_ptr+n);

  for (size_t i=0; i<n; i++) {
    cout << "a[" << i << "] = " << a_ptr[i] << endl;
  }

  cout << "--- end of mythrust ---" << endl;
  return 0;
}
