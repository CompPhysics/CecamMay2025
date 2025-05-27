compile_mac: 
	CPATH=/opt/homebrew/include LIBRARY_PATH=/opt/homebrew/lib g++-12 main_ising.cpp utils.cpp ./include/utils.hpp -std=c++11 -larmadillo  -fopenmp -o main -O3 

compile_linux:
	 g++ main_ising.cpp utils.cpp ./include/utils.hpp  -fopenmp -o main -O3 

run:
	./main $(L) $(mc_cycles) $(burn_pct) $(lower_temp) $(upper_temp) $(temp_step) $(align) $(output)

run_loop_L:
	for L in 40 60 80 100; do \
		./main $$L $(mc_cycles) $(burn_pct) $(lower_temp) $(upper_temp) $(temp_step) $(align) $(output); \
	done
