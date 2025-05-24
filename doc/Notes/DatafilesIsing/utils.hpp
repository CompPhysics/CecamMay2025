#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>
#include <random> // for std::mt19937
#include <omp.h>
#include <armadillo>
#include <vector>
#include <algorithm> // for openmp ?
#include <stdlib.h>  /* srand, rand */


arma::mat init_random_config(int L, double &exp_E, double &exp_M, int align);
arma::mat metropolis(arma::mat config, double beta, double &exp_E, double &exp_M, arma::vec p_vec);
int mt_random_int(int low, int high);
double mt_random_float(int low, int high);
void monte_carlo(int L, int mc_cycles, int burn_pct, double T, int align, int thread_id, std::string output, double lower_temp, double upper_temp);


#endif
