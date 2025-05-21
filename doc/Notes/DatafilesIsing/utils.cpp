#include "./include/utils.hpp"

std::mt19937 gen(time(NULL));
double kb = 1.;
arma::mat init_random_config(int L, double &E, double &M, int align)
{
    int N = L * L;
    arma::mat config = arma::zeros<arma::mat>(L, L);

    for (int i = 0; i < N; i++)
    {
        int k = i / L;
        int l = i % L;
        if (align == 0)
        {
            config(k, l) = mt_random_int(0, 1) * 2 - 1; // makes -1 or 1 with equal probability
        }
        else if (align == 1)
        {
            config(k, l) = 1;
        }
        else if (align == 2)
        {
            config(k, l) = -1;
        }
        M += config(k, l);
    }
    // notice that to find the initial energy, we need another loop
    for (int i = 0; i < N; i++)
    {
        int k = i / L;
        int l = i % L;

        E += -config(k, l) * (config((k + 1) % L, l) + config((k - 1 + L) % L, l) + config(k, (l + 1) % L) + config(k, (l - 1 + L) % L));
    }
    E = E / 2; // since we double count each interaction

    return config;
}

arma::mat metropolis(arma::mat config, double beta, double &E, double &M, arma::vec p_vec)
{
    int L = config.n_rows;
    for (int iter = 0; iter < L * L; iter++)
    {
        double neigbours_energy = 0; // energy of the neighbours of the randomly chosen spin that is why it has to be reset
        double delta_E = 0;

        int i = mt_random_int(0, L * L - 1); // choose a random spin index to flip

        // figure out k and l from i
        int k = i / L;
        int l = i % L;

        neigbours_energy += -config(k, l) * (config((k + 1) % L, l) + config((k - 1 + L) % L, l) + config(k, (l + 1) % L) + config(k, (l - 1 + L) % L));

        delta_E = -2 * neigbours_energy; // because because delta_E = E_new - E_old = -((-a)*b) - (-(a*b)) = 2*(a*b) = -2*neigbours_energy

        if (delta_E <= 0)
        {
            config(k, l) *= -1;
            E += delta_E;
            M += 2 * config(k, l); // factor of 2 because we are only changing one spin
        }
        else
        {
            double p = p_vec[(delta_E + 8) / 4]; // this did not make it faster

            if (mt_random_float(0, 1) < p)
            {
                config(k, l) *= -1;
                E += delta_E;
                M += 2 * config(k, l); // factor of 2 because we are only changing one spin
            }
        }
    }
    return config;
}

void monte_carlo(int L, int mc_cycles, int burn_pct, double T, int align, int thread_id, std::string output, double lower_temp, double upper_temp)
{
    double beta = 1. / (kb * T);
    arma::vec p_vec = arma::zeros<arma::vec>(5);
    arma::vec delta_e_vec = arma::vec(std::vector<double>{-8, -4, 0, 4, 8});
    for (int i = 0; i < 5; i++)
    {
        p_vec(i) = std::exp(-beta * delta_e_vec(i));
    }

    std::ofstream file1;
    std::ofstream file2;
    std::ofstream file3;

    std::string rounded_T = std::to_string(T).substr(0, std::to_string(T).find(".") + 4);
    std::string lower_T = std::to_string(lower_temp).substr(0, std::to_string(lower_temp).find(".") + 4);
    std::string upper_T = std::to_string(upper_temp).substr(0, std::to_string(upper_temp).find(".") + 4);
    std::string str_L = std::to_string(L);
    std::string str_mc_cycles = std::to_string(mc_cycles);
    std::string str_burn_pct = std::to_string(burn_pct);
    std::string str_align = std::to_string(align);

    std::cout << "T: " << T << std::endl;
    std::string filename1;
    std::string filename2;
    std::string filename3;
    if (output == "all_qt" || output == "last_qt")
    {
        std::cout << "upper_temp" << upper_temp << "lower_temp" << lower_temp << std::endl;
        filename1 = "data/" + str_L + "/" + output + "_L" + str_L + "_A" + str_align + "_mc" + str_mc_cycles + "_burn" + str_burn_pct + "_lt" + lower_T + "_ut" + upper_T + ".csv";
    }
    else if (output == "grid")
    {

        filename1 = "data/" + str_L + "/s1_cfg_L" + str_L + "_A" + str_align + "_mc" + str_mc_cycles + "_burn" + str_burn_pct + "_tl" + lower_T + "_tu" + upper_T + ".csv";
        filename2 = "data/" + str_L + "/s2_cfg_L" + str_L + "_A" + str_align + "_mc" + str_mc_cycles + "_burn" + str_burn_pct + "_tl" + lower_T + "_tu" + upper_T + ".csv";
        filename3 = "data/" + str_L + "/s3_cfg_L" + str_L + "_A" + str_align + "_mc" + str_mc_cycles + "_burn" + str_burn_pct + "_tl" + lower_T + "_tu" + upper_T + ".csv";
    }
    else if (output == "epsilons")
    {
        filename1 = "data/" + str_L + "/epsilons_L" + str_L + "_A" + str_align + "_mc" + str_mc_cycles + "_burn" + str_burn_pct + "_lt" + lower_T + "_ut" + upper_T + ".csv";
    }
    else
    {
        std::cout << "Invalid output type" << std::endl;
        return;
    }

    // avg will always be wrt to the number of mc cycles
    double E = 0;
    double M = 0;

    double cumul_E = 0;

    double cumul_E2 = 0;
    double avg_e = 0;
    double avg_E = 0;
    double avg_E2 = 0;

    double var_E = 0;

    double var_M = 0;
    double avg_M2 = 0;

    double avg_mabs = 0;

    double cumul_M2 = 0;
    double avg_Mabs = 0;

    double avg_Mabs2 = 0;
    double cumul_Mabs = 0;

    double cumul_mabs = 0;

    double Cv = 0;
    double chi = 0;
    arma::mat config = init_random_config(L, E, M, align);

    // create vector filled with the same tempeara as the one we are currently simulating L times
    arma::vec T_vec = arma::zeros<arma::vec>(L);
    T_vec.fill(T);

    file1.open(filename1, std::ofstream::app);
    if (output == "grid")
    {
        file2.open(filename2, std::ofstream::app);
        file3.open(filename3, std::ofstream::app);
        config.save(file1, arma::csv_ascii);
        file1.close();
    }

    double N = L * L;
    int burn_in = int((burn_pct / 100.) * mc_cycles);
    int mc_counter = 0;

    arma::mat output_qt_all = arma::zeros<arma::mat>(mc_cycles - burn_in, 5);
    arma::mat output_last_qt = arma::zeros<arma::mat>(1, 5); // for the last row of the qt file to avoid loading a huge file if that is now asked
    arma::mat epsilons = arma::zeros<arma::mat>(mc_cycles - burn_in, 1);

    for (int i = 0; i < mc_cycles; i++)
    {
        config = metropolis(config, beta, E, M, p_vec);

        cumul_E += E;
        cumul_E2 += E * E;
        cumul_Mabs += std::abs(M);
        cumul_M2 += M * M;

        if ((output == "grid") && i == int(mc_cycles / 2))
        {
            config.save(file2, arma::csv_ascii);
            file2.close();
        }

        if (i >= burn_in)
        {

            avg_E2 = cumul_E2 / (i + 1);
            avg_E = cumul_E / (i + 1);
            var_E = avg_E2 - avg_E * avg_E;

            avg_Mabs = cumul_Mabs / (i + 1);

            avg_M2 = cumul_M2 / (i + 1);
            avg_Mabs2 = avg_Mabs * avg_Mabs;

            var_M = avg_M2 - avg_Mabs2;

            avg_e = avg_E / N;
            avg_mabs = avg_Mabs / N;

            Cv = (var_E * beta) / (N * T);
            chi = (var_M * beta) / (N);
            if (output == "all_qt")
            {
                output_qt_all(i - burn_in, 0) = avg_e;
                output_qt_all(i - burn_in, 1) = avg_mabs;
                output_qt_all(i - burn_in, 2) = Cv;
                output_qt_all(i - burn_in, 3) = chi;
                output_qt_all(i - burn_in, 4) = T;
            }
            epsilons(i - burn_in) = E / N;
        }
    }

    // notice this is outside the loop
    if (output == "all_qt")
    {
        output_qt_all.save(file1, arma::csv_ascii);
        file1.close();
    }
    else if (output == "last_qt")
    {
        output_last_qt(0, 0) = avg_e;
        output_last_qt(0, 1) = avg_mabs;
        output_last_qt(0, 2) = Cv;
        output_last_qt(0, 3) = chi;
        output_last_qt(0, 4) = T;
        output_last_qt.save(file1, arma::csv_ascii);
        file1.close();
    }
    else if (output == "epsilons")
    {
        epsilons.save(file1, arma::csv_ascii);
        file1.close();
    }
    else if (output == "grid")
    {
        // concatenate config and T_vec
        arma::mat config_T = arma::join_rows(config, T_vec);
        config_T.save(file3, arma::csv_ascii);
        file3.close();
    }
}

int mt_random_int(int low, int high)
{
    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

double mt_random_float(int low, int high)
{
    std::uniform_real_distribution<double> dist2(low, high);
    return dist2(gen);
}