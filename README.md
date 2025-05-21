# CEACM Flagship School: Machine Learning in Physical Sciences: Theory and Applications
May 26, 2025 - May 30, 2025 see https://www.cecam.org/workshop-details/machine-learning-in-physical-sciences-theory-and-applications-1449


## Using Machine Learning to Classify Phase Transitions

These notes, with pertinent exercises cover the following topics 
- Phase Transitions & Critical Phenomena: Definitions and key concepts (order parameters, critical points, first vs second order).
- Spin Models: 2D Ising model and the q-state Potts model (examples of phase transitions).
- Data Generation: Monte Carlo simulations for sampling spin configurations across temperatures.
- Unsupervised Learning (PCA): Principal Component Analysis to visualize phase separation without labels.
- Supervised Learning (CNN): Convolutional Neural Networks for classifying phases from raw configurations.
- Generative Models (VAE): Variational Autoencoders for latent representation learning and critical anomaly detection.
- Comparisons: Interpretability and performance trade-offs between PCA, CNN, and VAE.






### Phase Transitions: Overview


- Definition: A phase transition is characterized by an abrupt, non-analytic change in a macroscopic property of a system as some external parameter (e.g. temperature) is varied . In simpler terms, the system’s state or phase changes dramatically at a critical point.
- Order Parameter: Associated with each phase transition is an order parameter – a quantity that is zero in one phase and non-zero in the other. For example, magnetization plays the role of an order parameter in magnetic systems, distinguishing ordered (magnetized) from disordered (unmagnetized) phases.
- Critical Point: At the critical temperature (or pressure, etc.), the order parameter changes (continuous or discontinuous) and the system exhibits critical phenomena: large fluctuations, divergence of correlation length, and the onset of scale invariance. Critical points of second-order transitions feature continuous change of the order parameter with characteristic critical exponents and universal behavior across different systems.



### Phase Transitions: First vs Second Order


- First-Order vs Second-Order: In a first-order transition, the order parameter changes discontinuously at the transition (often with latent heat), whereas in a second-order (continuous) transition, the order parameter goes to zero continuously at T_c, accompanied by diverging susceptibility and correlation length. For example, the liquid–gas transition (at sub-critical pressures) is first-order, while the ferromagnetic transition in the 2D Ising model is second-order (continuous).
- Example – Potts Model Transitions: The q-state Potts model generalizes Ising (which is q=2). In 2D, the Potts model undergoes a continuous transition for q ≤ 4 and a discontinuous (first-order) transition for q > 4 . This highlights how the nature of the phase transition can change with system parameters.
- Critical Phenomena: Near second-order transitions, critical phenomena include power-law divergences (e.g. specific heat, susceptibility), critical opalescence (fluctuations at all scales), and universality (different systems share the same critical exponents if they have the same symmetry and dimensionality). These concepts set the stage for identifying phase transitions through data features (e.g. large fluctuations near T_c might be detectable by learning algorithms).

### 2D Ising Model (Ferromagnet)





Two-dimensional Ising model with spins up (pink arrows) and down (blue arrows) on a square lattice. Each spin can be $s_i = \pm 1$ and interacts with its nearest neighbors; aligned neighbors (same color) contribute a lower energy ($-J$ per bond) while anti-aligned neighbors contribute higher energy ($+J$). The Hamiltonian (with no external field) is $H = -J \sum_{\langle i,j\rangle} s_i s_j$, favoring parallel alignment of spins. The competition between interaction energy and thermal agitation leads to a ferromagnetic phase at low temperatures (most spins align, yielding non-zero net magnetization) and a paramagnetic phase at high temperatures (spins are disordered, zero net magnetization). The 2D square-lattice Ising model is the simplest model that exhibits a phase transition at a finite critical temperature T_c (approximately $T_c \approx 2.269,J/k_B$ for the infinite lattice). Below $T_c$ the system spontaneously magnetizes (symmetry breaking), while above $T_c$ it remains disordered.






### q-State Potts Model


- Definition: The q-state Potts model is a generalization of the Ising model where each spin can take $q$ discrete values (e.g. $q$ “colors” instead of just up/down). The Hamiltonian for the ferromagnetic Potts model can be written $H = -J \sum_{\langle i,j\rangle} \delta_{\sigma_i,\sigma_j}$, where $\sigma_i \in {1,\dots,q}$ and $\delta$ is the Kronecker delta (neighbor interaction is $-J$ if spins are in the same state). For $q=2$, this reduces to the Ising model.
- Phases: Like the Ising case, at low temperature a Potts model magnetically orders (most spins in the same state), and at high temperature it is disordered (spins randomly distributed among the $q$ states). The nature of the transition, however, depends on $q$. In 2D, all $q\ge 1$ have a phase transition at some critical temperature given by $k_B T_c/J = 1/\ln(1+\sqrt{q},)$ . For $q \le 4$ the transition is continuous (second-order), in the same universality class as the Ising model for $q=2$, whereas for $q > 4$ the transition becomes first-order (discontinuous jump in order parameter) .
- Significance: The $q$-state Potts model exemplifies how increasing internal symmetry (more spin states) can change transition order. It provides a wider test-bed for machine learning methods: e.g. can an algorithm trained on one type of transition detect another? We will focus primarily on the Ising case (as $q=2$) for data generation and learning, with the understanding that methods can be extended to Potts models and beyond.






### Data Generation via Monte Carlo Simulations


- Purpose of Simulation: To apply machine learning, we first need data – in this context, spin configuration snapshots at various temperatures. We generate these using Monte Carlo (MC) simulations of the spin models. MC methods (like the Metropolis algorithm or cluster algorithms such as Wolff) allow us to sample representative spin configurations from the equilibrium distribution at a given temperature $T$ (according to the Boltzmann weight $P({s}) \propto e^{-H/k_BT}$).
- Metropolis Algorithm: Starting from a random or ordered spin configuration, we randomly flip spins and accept or reject flips based on the energy change $\Delta E$, according to the Metropolis criterion: always accept if $\Delta E \le 0$ (favorable) and accept with probability $e^{-\Delta E/k_BT}$ if $\Delta E > 0$. This procedure ergodically samples the configuration space. We perform many sweeps (updates of all spins) to ensure the system equilibrates at the target temperature before collecting data.
- Sampling Configurations: We simulate a range of temperatures spanning below and above the expected $T_c$. For each temperature, we collect many uncorrelated configurations (taking samples sufficiently far apart in MC steps to reduce autocorrelation). These configurations can be represented as 2D images (with spin up vs down as two colors, or values ±1). In practice, researchers generate gigabytes of spin configuration “images” across phases – e.g. hundreds or thousands of configurations at each temperature – to use as the training dataset for machine learning. The labels (phase or temperature) may be attached to each sample if doing supervised learning, or left unlabelled for unsupervised methods.
- Example: Carrasquilla and Melko (2017) generated a large collection of Ising model states at various temperatures using simulation, effectively creating a database of “magnet snapshots” on which they trained neural networks . The availability of large simulation datasets is a key enabler for applying modern machine learning to phase transition problems.






### Unsupervised Learning: PCA for Phase Separation



- Principal Component Analysis: PCA is a classical unsupervised dimensionality reduction technique. It identifies directions (principal components) in feature space along which the data variance is maximal. By projecting high-dimensional data (here, spin configurations with $N$ spins as $N$-dimensional vectors) onto the first few principal components, we obtain a low-dimensional representation that captures the most significant variations in the data.
- Applying PCA to Ising Data: We treat each spin configuration as a vector of length $N=L\times L$ (with entries  ±1 for spin-down/up). PCA can be performed on a set of configurations across different temperatures. Interestingly, PCA often finds that the leading principal component corresponds to the order parameter (magnetization per spin) in the Ising model . In other words, the largest variance in the dataset comes from whether the configuration is mostly +1 or mostly –1 (ordered vs disordered spins). This is because below $T_c$ configurations have a bias toward all-up or all-down (high $|M|$), whereas above $T_c$ configurations have $M\approx 0$ on average; thus, when mixed together, the dominant distinguishing feature is the magnetization.
- Visualizing Phases: Plotting configurations in the space of the first one or two principal components reveals clusters corresponding to phases. For example, one can observe two clusters of points: low-temperature configurations cluster at extreme values of PC1 (positive or negative, reflecting the two possible magnetization orientations), and high-temperature configurations cluster near PC1 = 0 (no magnetization). This unsupervised clustering means PCA distinguishes the ferromagnetic and paramagnetic phases without any labels, effectively using variance in spin alignment to separate phases . Moreover, by scanning through temperature, one can identify where the data variance (or the separation along PC1) rapidly changes – giving an estimate of the critical temperature. Studies have shown that PCA not only identifies phases but can also locate the transition point and even differentiate types of transitions (e.g. by analyzing how many principal components carry significant variance, one can sometimes tell continuous vs first-order transitions ).
- Physical Interpretation: PCA provides an interpretable result: the principal axes (eigenvectors) can often be interpreted in physical terms. For the Ising model, the first principal component’s weight vector is essentially uniform (all spins weighted equally), corresponding to the collective magnetization mode . This aligns with the notion that magnetization is the key feature distinguishing phases. Higher principal components might capture more subtle patterns (e.g. domain wall structures or staggered magnetization if present). The clear mapping of a principal component to an order parameter is a big advantage in interpretability – it tells us the algorithm “learned” a known physical quantity. Of course, PCA is linear and may fail to capture non-linear correlations or more complex orders, but it serves as a powerful baseline.


### Supervised Learning: CNN for Phase Classification


- Goal: In supervised learning, we provide labeled examples to train a model to classify phases. For the Ising model, a straightforward labeling is to tag each configuration as “ordered” (if $T < T_c$) or “disordered” ($T > T_c$). Alternatively, one can label by temperature value or even try to predict the temperature from the configuration (a regression task). Here we focus on classification into phases using a Convolutional Neural Network (CNN) – a powerful architecture for image recognition tasks.
- Why CNN: Spin configurations can be viewed as 2D images (with each site like a pixel of value ±1). A CNN is well-suited to capture local spatial patterns (e.g. clusters of aligned spins, domain walls) via convolution filters. It also respects translational invariance (a domain of up-spins is detected regardless of where it is on the lattice). Earlier work showed that even a simple feed-forward neural network could be trained to identify phases from raw configurations . By using a CNN, we can even capture more complex features, including those relevant for topological phases (which lack a local order parameter) .
- Training the CNN: We supply the CNN with many labeled examples of configurations at known temperatures. The network learns to output one class for low-T (ferromagnet) and another for high-T (paramagnet). Remarkably, once trained, the CNN can accurately distinguish an ordered phase from a disordered phase from the raw spin snapshot . When presented with an unseen configuration at an intermediate temperature, the network’s output can indicate the probability of it being in the ordered phase. As one scans temperature, the output probability drops from ~1 to 0 around the critical region, allowing an estimate of $T_c$ (the point of maximal confusion). In Carrasquilla and Melko’s pioneering study, the CNN not only distinguished phases with high accuracy but also identified the phase transition boundary without being told the physics explicitly .
- Interpretability: Although CNNs are complex models, we can attempt to interpret what features they learn. In the Ising case, it was found that the trained neural network essentially learned to measure the magnetization of the input configurations . The network’s output was strongly correlated with the magnetization (which is the theoretical order parameter) – indicating the CNN autonomously discovered this key feature to discriminate phases. This is reassuring: the “black box” arrived at a physically meaningful strategy. For more complex phases (e.g. topological phases with no obvious local order parameter), CNNs have been shown to detect transitions as well , though interpreting what they rely on can be more challenging.
- Beyond Binary Classification: One can extend this supervised approach to classify multiple phases or phases of the q-state Potts model (with q > 2, there may be more than two phase labels if considering symmetry-broken states as distinct). The network architecture or output layer can be adjusted accordingly (softmax outputs with $n$ classes). Supervised ML has also been used to recognize phases in other models (XY model Kosterlitz-Thouless transition, etc.), sometimes requiring more sophisticated techniques when there is no clear label (there is a method called “learning by confusion” which involves training on hypothetical labels and finding when the network is most confused, pinpointing the transition). Overall, CNNs provide a high-performance tool: with sufficient training data, they can pinpoint phase transitions even in cases where traditional order parameters are not obvious .





### Generative Learning: VAE for Latent Representations

- Autoencoders: Another unsupervised approach uses autoencoders, neural networks trained to compress and then reconstruct data. A Variational Autoencoder (VAE) is a probabilistic generative model that learns a latent variable description of the data. It consists of an encoder network $E_\phi$ that maps an input (spin configuration) to a set of latent variables (mean & variance for each latent dimension), and a decoder network $D_\theta$ that maps a sample from this latent distribution back to the data space, attempting to reconstruct the original input . The VAE is trained by maximizing a lower bound to the data likelihood, which includes a reconstruction error term and a regularization term pushing the latent distribution toward a prior (usually an isotropic Gaussian).
VAEs for Phase Transitions: The idea is that the VAE’s latent space will capture the essential features of the configurations. If the model is well-trained, configurations from different phases might occupy different regions in latent space. Indeed, studies have found that a VAE with a 1- or 2-dimensional latent space can learn to encode spin configurations in such a way that the latent variable correlates with the order parameter (magnetization) . For example, a single latent dimension might be mapped to the magnetization of the configuration. As a result, the latent representation $z$ effectively classifies the phase: $z$ near +1 might correspond to mostly spin-up, $z$ near –1 to spin-down (ordered phases), and $z \approx 0$ to disordered configurations. This means the VAE autonomously discovers the phase structure: the latent variables cluster configurations by phase without being told about phases . This was demonstrated by Wetzel (2017), who showed that PCA and VAE both yield latent parameters corresponding to known order parameters, and that latent encodings form distinct clusters for different phases .
Detecting Criticality: How do we detect the phase transition using a VAE? One way is to look at the distribution of latent variables as a function of temperature. In the ordered phase, the latent encodings might split into two separated modes (for up vs down magnetization domains), whereas near $T_c$ the encodings spread out or the distribution changes character, and in the disordered phase they cluster around a single mode (zero magnetization) . Another way is to use the VAE’s reconstruction loss: it has been observed that the reconstruction error often peaks or changes behavior at the phase transition . Intuitively, at criticality the configurations are most “surprising” or hardest to compress, which can lead to a bump in reconstruction error – making it a possible unsupervised indicator of $T_c$. In summary, VAEs provide both a dimensionality reduction and a generative model: they not only identify phases (via latent clustering) but can also generate new configurations by sampling latent variables and decoding. Generated samples from a trained VAE qualitatively resemble the real ones and carry features of the learned distribution . For instance, decoding random latent vectors yields spin configurations whose energy–magnetization distribution looks similar to the training data’s distribution across phases (though one must be cautious: small latent spaces may fail to capture all correlations, producing unphysical samples at times ).
- Anomaly Detection: Another interesting use of VAEs is treating the VAE as an anomaly detector for criticality. If you train the VAE on data mostly away from $T_c$, the critical configurations might reconstruct poorly (higher error) because they don’t fit well into either phase’s learned features. By scanning temperature, a spike in VAE reconstruction error or a significant shift in latent space usage can signal a transition . This approach doesn’t require prior labeling of phases, thus serving as a fully unsupervised phase transition detector.
Summary: VAEs, like PCA, perform unsupervised feature extraction, but with non-linear deep learning power. They bridge a gap: more expressive than linear PCA, and they provide generative capability (sampling and interpolation between configurations). Their latent dimensions can be interpreted (e.g. as order parameters) but the mapping is learned, not predefined. This makes them a promising tool for discovering unknown order parameters or subtle phase transitions in many-body systems .




### Identifying Ising model phase transitions with neural networks

The Ising model has been simulated using the Metropolis method, a Markov Chain Monte Carlo (MCMC) method. 

Here results here have been obtained by  implementing the 50x50 2D lattice version of the Ising model and
training a convolutional neural network (CNN) to perform regession and
classification.  The regression task deals with the  prediction of  the lattice
temperature while the classification part  identifies configurations
above and below critical temperature of phase transition. We then
predict the critical temperature as the one which exhibits largest
uncertainty in classification.

Here are 3 of the 1265 elements of the data set
<p align="center">
<img src="https://github.com/Daniel-Haas-B/FYS-STK4155/blob/main/project3/figs/L50_configs.png?raw=true"/>
<p align="center">

Here is the architecture of the best CNN implemented:

<p align="center">
<img src="https://github.com/CompPhysics/CecamMay2025/tree/main/doc/Notes/DatafilesIsing/figs/TF_CNN_arch.png?raw=true"  width="500"/>


Here is the best regression predictions over test set

<p align="center">
<img src="https://github.com/CompPhysics/CecamMay2025/tree/main/doc/Notes/DatafilesIsing/figs/TF_CNN_l2reg001_eta00001_epoch1000-1.png?raw=true" width="500"/>
<p align="center">



Here is the confusion matrix from the critical temperature classification
<p align="center">
<img src="https://github.com/CompPhysics/CecamMay2025/tree/main/doc/Notes/DatafilesIsing/figs/TF_CNN_confusion_matrix-1.png?raw=true" width="500"/>
<p align="center">


Here is the critical temperature infered from classification


<p align="center">
<img src="https://github.com/CompPhysics/CecamMay2025/tree/main/doc/Notes/DatafilesIsing/figs/figs/TF_CNN_probabilities-1.png?raw=true" width="500"/>
<p align="center">


## Generating the datasets yourself with the C++ code
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
  - The C++ code requires the user to have a folder called "data" with the subfolders "20", "40", "60", "80" and "100" in order to put the datafiles. Github does not comport empty folders and since we are not supposed to load the files to the repository, there is no "data" folder visible.  These data need to be generated by you. Feel free to change the c++ code to say one in Python or other programming languages.
