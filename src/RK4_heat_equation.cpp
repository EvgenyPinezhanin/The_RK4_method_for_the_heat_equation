#include <RK4_heat_equation.h>

#include <cmath>
#include <algorithm>
#include <iostream>

#include <omp.h>

double heat_task::initial_condition(double x) {
    return 0.0;
}

double heat_task::left_condition(double t) {
    return 0.0;
}

double heat_task::right_condition(double t) {
    return 1.0;
}

double heat_task::f(double x, double t) {
    return 0.0;
}

void heat_equation_runge_kutta(heat_task task, double *v) {
    double h = task.L / task.n;
    double tau = task.T / task.m;

    int n = task.n;
    int m = task.m;

    double *v_curr = new double[n + 1];
    double *v_next = new double[n + 1];
    double *v_tmp = new double[n + 1];
    double *k1 = new double[n + 1];
    double *k2 = new double[n + 1];
    double *k3 = new double[n + 1];
    double *k4 = new double[n + 1];

#pragma omp parallel for schedule(static)
    for (int i = 1; i < n; ++i) {
        v_curr[i] = task.initial_condition(i * h);
    }
    v_curr[0] = task.left_condition(0.0);
    v_curr[n] = task.right_condition(0.0);

    for (int iter = 0; iter < m; ++iter) {
    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            k1[i] = task.f(i * h, iter * tau) + (v_curr[i - 1] - 2.0 * v_curr[i] + v_curr[i + 1]) / (h * h);
            k1[i] *= tau;
        }

    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            v_tmp[i] = v_curr[i] + 0.5 * k1[i];
        }
        v_tmp[0] = task.left_condition((iter + 0.5) * tau);
        v_tmp[n] = task.right_condition((iter + 0.5) * tau);

    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            k2[i] = task.f(i * h, (iter + 0.5) * tau) + (v_tmp[i - 1] - 2.0 * v_tmp[i] + v_tmp[i + 1]) / (h * h);
            k2[i] *= tau;
        }

    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            v_tmp[i] = v_curr[i] + 0.5 * k2[i];
        }
        v_tmp[0] = task.left_condition((iter + 0.5) * tau);
        v_tmp[n] = task.right_condition((iter + 0.5) * tau);

    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            k3[i] = task.f(i * h, (iter + 0.5) * tau) + (v_tmp[i - 1] - 2.0 * v_tmp[i] + v_tmp[i + 1]) / (h * h);
            k3[i] *= tau;
        }

    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            v_tmp[i] = v_curr[i] + k3[i];
        }
        v_tmp[0] = task.left_condition((iter + 1) * tau);
        v_tmp[n] = task.right_condition((iter + 1) * tau);

    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            k4[i] = task.f(i * h, (iter + 1) * tau) + (v_tmp[i - 1] - 2.0 * v_tmp[i] + v_tmp[i + 1]) / (h * h);
            k4[i] *= tau;
        }

    #pragma omp parallel for schedule(static)
        for (int i = 1; i < n; ++i) {
            v_next[i] = v_curr[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        v_next[0] = task.left_condition((iter + 1) * tau);
        v_next[n] = task.right_condition((iter + 1) * tau);

        std::swap(v_curr, v_next);
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; ++i) {
        v[i] = v_curr[i];
    }

    delete [] v_curr;
    delete [] v_next;
    delete [] v_tmp;
    delete [] k1;
    delete [] k2;
    delete [] k3;
    delete [] k4;
}
