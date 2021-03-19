#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// double diffclock( high_resolution_clock::time_point t2, high_resolution_clock::time_point t1)
// {
//     duration<double, std::milli> time_span = t2 - t1;
//     return time_span.count();
// }

long diffclock(system_clock::time_point c1,system_clock::time_point c2){
  milliseconds ms = duration_cast< milliseconds >(c1-c2);
  return ms.count();
}

void setcolor(int foreground, int background=0) {
  foreground < 8 ? foreground += 30 : foreground += 82;
  background < 8 ? background += 40 : background += 92;

  if (background == 40) {
    cout << "\033[" << foreground << ";0m";
    cout << "\033[" << foreground << "m";
  } else
    cout << "\033[" << foreground << ";" << background << "m";
}

void print_vector(string name, double *t, int n, int foreground = 7,
                  int background = 0) {
  setcolor(foreground, background);
  cout << name << "={ ";
  for (int i = 0; i < n; i++) {
    cout << t[i];
    if (i < n - 1)
      cout << ",";
  }
  cout << " }";
  setcolor(15);
}


/*
Colors:
0 : black
1 : red
2 : green
3 : yellow
4 : blue
5 : purple
6 : cyan
7 : white
*/