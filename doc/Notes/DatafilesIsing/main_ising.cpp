#include <string>
#include "./include/utils.hpp"

// define your physical system's variables here:

int main(int argc, char *argv[])
{   
    
    // main will evolve the system and output the results to a file in matrix form
    int L = atoi(argv[1]);
    
    int mc_cycles = atoi(argv[2]); // 1 = L^2 runs, 2 = 2*L^2 runs, etc
    
    int burn_pct = atoi(argv[3]);  // 10 = 10% burn in, 20 = 20% burn in, etc
    std::cout << burn_pct << std::endl;
    double lower_temp = atof(argv[4]);
    double upper_temp = atof(argv[5]);
    double temp_step = atof(argv[6]);
    int align = atoi(argv[7]); // 0 = random, 1 = aligned up (1), 2 = aligned down (-1)
    std::string output = argv[8];

    
    int temp_n = int((upper_temp - lower_temp) / temp_step);

#pragma omp parallel
    {
        int iter_;
        double T;
        double temp_step = atof(argv[6]);
#pragma omp for
        for (iter_ = 0; iter_ < temp_n + 1; iter_++)
        {
            T = lower_temp + iter_ * temp_step;
            monte_carlo(L, mc_cycles, burn_pct, T, align, omp_get_thread_num(), output, lower_temp, upper_temp);
            std::cout << "Thread " << omp_get_thread_num() << " finished T = " << T << std::endl;
        }
    }
    return 0;
}
