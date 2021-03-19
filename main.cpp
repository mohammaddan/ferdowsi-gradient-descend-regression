#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <chrono>
#include "public-lib.h"

using namespace std;

const int n = 2000;
const int p = 10;
const double epsilon = 0.0001;
double params[] = {100, 2, 4, 0.5, 1, 4, 3, 2, 1, 2};
int attrbite_range[] = {10, 100, 50000, 5, 10, 3, 50, 20, 50, 100};
double norm[p];
double train_x[n][p] = {0}, train_y[n] = {0};
double theta[p];

void set_train_set();
void tink(double *theta);
void tink0(double *theta);
void tink2(double *theta,int thread_num);
double cal_y(double *x);
double predict_y(double *x);

int main(int argc, const char **argv) {

  // omp_set_num_threads(NUM_THREADS);
  srand(time(NULL));
  set_train_set();

  setcolor(1);

  system_clock::time_point start, end;

  start = system_clock::now();
  tink0(theta);
  end = system_clock::now();
  cout << "Sequense (tink function) single thread time : " << diffclock(end, start)/1000.0
       << " s\n";

  for(int i=1;i<=8;i++){
    start = system_clock::now();
    tink2(theta,i);
    end = system_clock::now();
    cout << "parallel (tink2 function) thread_num : "<< i <<" time : " << diffclock(end, start)/1000.0
        << " s\n";

  }

  setcolor(7);
  print_vector("theta", theta, p, 2, 0);
  cout << "\n";
  // double x[p];
  // x[0] = 1;
  // for (int i = 1; i < p; i++)
  //   x[i] = attrbite_range[i] / (1 + rand() % 5);
  // print_vector("x", x, p, 7, 4);
  // double cy = cal_y(x), py = predict_y(x);
  // setcolor(2, 0);
  // cout << "  y=" << cy << "  predict_y=" << py;
  // setcolor(1);
  // cout << " diff=" << fabs(cy - py) << "\n\n";
  setcolor(7);
  return 0;
}

double cal_y(double *x) {
  double result = theta[0];
  for (int i = 1; i < p; i++)
    result += x[i] * theta[i];
  return result;
}

double predict_y(double *x) {
  double result = params[0];
  for (int i = 1; i < p; i++)
    result += x[i] * params[i];
  return result;
}

void set_train_set() {
  double t = 0;
  for (int i = 0; i < n; i++) {
    train_x[i][0] = 1;
    train_y[i] = train_x[i][0] * params[0];
    for (int j = 1; j < p; j++) {
      float noise = (rand() % 100 - 50) / 10000.0;
      train_x[i][j] = rand() % attrbite_range[j];
      train_x[i][j] /= attrbite_range[j];
      train_y[i] += params[j] * train_x[i][j] + noise;
    }
  }
}

void tink(double *theta) {
  double alpha = 0.01;
  for (int i = 0; i < p; i++)
    theta[i] = (rand() % 10) / 1000.0;
  double delta[p] = {0};
  double y_pred[n] = {0};
  int cnt = 2000000, it = 0;
  double last_error, error = 100;
  do {
    last_error = error;
    for (int i = 0; i < p; i++)
      delta[i] = 0;
    for (int i = 0; i < n; i++) {
      y_pred[i] = 0;
      for (int j = 0; j < p; j++) {
        y_pred[i] += theta[j] * train_x[i][j];
      }
    }
    double error2 = 0, err;
    for (int i = 0; i < n; i++) {
      err = y_pred[i] - train_y[i];
      error2 += err * err;
      for (int j = 0; j < p; j++) {
        delta[j] += train_x[i][j] * err;
      }
    }
    error = sqrt(error2);
    if (error > last_error)
      alpha /= 1.2;
    for (int i = 0; i < p; i++)
      theta[i] -= alpha * delta[i] / n;
    // if (it % 10000 == 0) {
    //   cout << "iterate : " << it << "\n";
    //   print_vector("theta", theta, p, 2, 0);
    //   cout << "\nalpha=" << alpha << " error : " << error << endl;
    // }

  } while (it++ < cnt && fabs(error - last_error) > epsilon);
  // setcolor(1);
  // cout << "\ncalulate on ";
  // setcolor(4);
  // cout << it;
  // setcolor(15);
  // cout << " iteratation \n";
  // cout << endl;
}

void tink0(double *theta) {
  double alpha = 0.01;
  double err_0[n];

  for (int i = 0; i < p; i++)
    theta[i] = (rand() % 10) / 1000.0;

  double delta[p] = {0};
  double y_pred[n] = {0};
  int cnt = 2000000, it = 0;
  double last_error, error = 100;
  do {
    last_error = error;
    int j;
    for (int i = 0; i < n; i++) {
      y_pred[i] = 0;
      for (j = 0; j < p; j++) {
        y_pred[i] += theta[j] * train_x[i][j];
      }
    }

    double error2 = 0;

    for (int i = 0; i < p; i++) {
      delta[i] = 0;
    }

    for (int i = 0; i < n; i++) {
      err_0[i] = y_pred[i] - train_y[i];
      error2 += err_0[i] * err_0[i];
    }

    for (int j = 0; j < p; j++){
      for(int i=0;i<n;i++)
        delta[j] += train_x[i][j] * err_0[i];
    }

    error = sqrt(error2);
    if (error > last_error)
      alpha /= 1.2;

    for (int i = 0; i < p; i++)
      theta[i] -= alpha * delta[i] / n;

  } while (it++ < cnt && fabs(error - last_error) > epsilon);
}

void tink2(double *theta,int thread_num) {
  double alpha = 0.01;
  double err_0[n];

  for (int i = 0; i < p; i++)
    theta[i] = (rand() % 10) / 1000.0;

  double delta[p] = {0};
  double y_pred[n] = {0};
  int cnt = 2000000, it = 0;
  double last_error, error = 100;
  do {
    last_error = error;
    int j;
    #pragma omp parallel for private(j) num_threads(thread_num) schedule(static,8)
    for (int i = 0; i < n; i++) {
      y_pred[i] = 0;
      for (j = 0; j < p; j++) {
        y_pred[i] += theta[j] * train_x[i][j];
      }
    }

    double error2 = 0;

    #pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < p; i++) {
      delta[i] = 0;
    }

    #pragma omp parallel for reduction(+:error2) num_threads(thread_num)  schedule(static,8)
    for (int i = 0; i < n; i++) {
      err_0[i] = y_pred[i] - train_y[i];
      error2 += err_0[i] * err_0[i];
    }

    for (int j = 0; j < p; j++){
      double temp=0;
      #pragma omp parallel for shared(delta) num_threads(thread_num) reduction(+:temp)
      for(int i=0;i<n;i++)
        temp += train_x[i][j] * err_0[i];
      delta[j]=temp;
    }

    error = sqrt(error2);
    if (error > last_error)
      alpha /= 1.2;

    #pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < p; i++)
      theta[i] -= alpha * delta[i] / n;

  } while (it++ < cnt && fabs(error - last_error) > epsilon);

}