#include <iostream>
#include <random>

#include <RK4_heat_equation.h>
#include <omp.h>

int main() {
    double T = 1.0, L = M_PI;
    int n = 128 * 32;
    int m = 1024 * 32;
    heat_task task{ T, L, n, m };
    double *v = new double[n + 1];

    int num_threads[3] = { 1, 2, 4 };
    double start_time, end_time;
    double time[3];

    for (int i = 0; i < 3; ++i) {
        omp_set_num_threads(num_threads[i]);


        start_time = omp_get_wtime();
        heat_equation_runge_kutta(task, v);
        end_time = omp_get_wtime();
        time[i] = end_time - start_time;

        std::cout << "Number of threads: " << num_threads[i]
                  << ", Time: " << time[i]
                  << ", Boost: " << time[0] / time[i] << "\n";

    }

    delete [] v;

    return 0;
}
