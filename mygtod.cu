#include <iostream>
#include <unistd.h>
#include <sys/time.h>

typedef struct timeval timeval_t;
double elapsed_time(timeval_t start, timeval_t finish) {
  double start_s = (double)start.tv_sec +
    1.0e-6*(double)start.tv_usec;
  double finish_s = (double)finish.tv_sec +
    1.0e-6*(double)finish.tv_usec;
  return finish_s-start_s;
}

/*
// run all tests
timeval_t start;
timeval_t finish;
for (size_t i=0; i != num_run; ++i) {
  gettimeofday(&start,NULL);
  vec_exp(v1,v2);
  gettimeofday(&finish,NULL);
  time_vec->a[i] = elapsed_time(start,finish);
}
*/

int main() {
  using std::cout;
  using std::endl;

  timeval_t start;
  timeval_t finish;

  gettimeofday(&start,NULL);
  sleep(1);
  gettimeofday(&finish,NULL);

  double eltm = elapsed_time(start,finish);

  cout << "eltm: " << eltm << endl;
  
  cout << "--- end of my gtod ---" << endl;
}
