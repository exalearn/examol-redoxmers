# ExaMol for Redoxmers

This project contains the scripts used to design new redoxmers with [ExaMol]([examol](https://exalearn.github.io/ExaMol/)https://exalearn.github.io/ExaMol/).
The scripts go from the intiial validation of the quantum chemistry methods used in ExaMol 
up through the analysis of the run after completion.

## Installation

First clone this repository to the computer you'll use to design molecules:

```commandline
git clone https://github.com/exalearn/examol-redoxmers.git
```

You can then build the entire environment with Anaconda:

```commandline
cd examol-redoxmers
conda env create --file envs/environment-cpu.yml --force
```

The above command builds the environment for a commodity CPU. 
We will provide environments compatible with supercomputers over time.
