## Identifying Ising model phase transitions with neural networks

We simulated the Ising model using the Metropolis method, a Markov Chain Monte Carlo (MCMC) method. 

We implemented the 50x50 2D lattice version of the Ising model and trained a convolutional neural network (CNN) to perform regession and classification.
The regression task was of predicting the lattice temperature while the classification was identifying configurations above and below critical temperature of phase transition. We then predict the critical temperature as the one which exhibits largest uncertainty in classification.

Here are 3 of the 1265 elements of the data set
<p align="center">
<img src="https://github.com/Daniel-Haas-B/FYS-STK4155/blob/main/project3/figs/L50_configs.png?raw=true"/>
<p align="center">

Here is the architecture of the best CNN implemented:

<p align="center">
<img src="https://github.com/Daniel-Haas-B/FYS-STK4155/blob/main/project3/figs/TF_CNN_arch.png?raw=true"  width="500"/>


Here is our best regression predictions over test set

<p align="center">
<img src="https://github.com/Daniel-Haas-B/FYS-STK4155/blob/main/project3/figs/TF_CNN_l2reg001_eta00001_epoch1000-1.png?raw=true" width="500"/>
<p align="center">



Here is the confusion matrix from the critical temperature classification
<p align="center">
<img src="https://github.com/Daniel-Haas-B/FYS-STK4155/blob/main/project3/figs/TF_CNN_confusion_matrix-1.png?raw=true" width="500"/>
<p align="center">


Here is the critical temperature infered from classification


<p align="center">
<img src="https://github.com/Daniel-Haas-B/FYS-STK4155/blob/main/project3/figs/TF_CNN_probabilities-1.png?raw=true" width="500"/>
<p align="center">


## Generating the datasets yourself with our C++ code
Compile for mac users: 
```
make compile_mac
```
OBS: notice that the cpath in the make file is user-specific and might require changing  

Compile for linux users: 
```
make compile_linux
```

Run example:
```
make run L=2 mc_cycles=100000 burn_pct=1 lower_temp=1 upper_temp=2 temp_step=0.5 align=0 output=last_qt
```
- Parameters 
  - L: the size of the grid of a total of N = LXL electrons;
  - mc_cycles: number of N attempted randomly selected spin flips;
  - burn_pct: percentage of mc_cycles to burn;
  - lower_temp: temperature to begin the loop of temperatures (each temperature is saved in a different file);
  - upper_temp: temperature to stop the loop of temperatures;
  - temp_step: temperature step;
  - align: 0 initializes spin configuration randomly, 1 initializes all spins up and 2 initializes all spins down;
  - output: possibilities are "last_qt", "all_qt", "epsilons" and "grid".


Understanding possible output parameters:
  - output = "epsilons" outputs the Energies per spin at the end of each Monte Carlo cycle (notice this is not the average);
  - output = "qt_all" outputs all the quantities, avg_e avg_mabs Cv Chi and T (notice that all besides the temperature are normalized per spin) at the end of every Monte Carlo cycle;
  - output = "qt_last" outputs all the quantities, avg_e avg_mabs Cv Chi and T (notice that all besides the temperature are normalized per spin) at the end of the last Monte Carlo cyclem depening of which value of `mc_cycles` is passed as input;
  - output = "grid" outputs three configurations of the lattice grid: the initialized one, the one at half the Monte Carlo cycles, and the final configuration.
  
  
 **Important information about generated files:**
  - The C++ code requires the user to have a folder called "data" with the subfolders "20", "40", "60", "80" and "100" in order to put the datafiles. Github does not comport empty folders and since we are not supposed to load the files to the repository, there is no "data" folder visible.
