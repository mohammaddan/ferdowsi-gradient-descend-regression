#include <cstdio>
#include <omp.h>
using namespace std;
#define NUM_THREADS 8

int main(int argc, char **argv) {
  int p = 40;
  double temp_error2[NUM_THREADS];
  double delta_temp[p][NUM_THREADS];
  omp_set_num_threads(NUM_THREADS);
  int j;
#pragma omp parallel for private (j)
  for (int i = 0; i < p; i++) {
    for (j = 0; j < 8; j++) {
      printf("%d : %d : %d\n", i,j, omp_get_thread_num());
    }
  }
}