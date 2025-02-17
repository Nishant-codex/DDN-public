# dist-delay-net

This repository contains the code, data and analysis discussed in the distance-based delay networks
paper [Exploiting Signal Propagation Delays to Match Task Memory Requirements in Reservoir Computing](https://doi.org/10.3390/biomimetics9060355).
For related papers, check branches:

- [[ddn-ipc]](https://github.com/StefanIacob/DDN-public/tree/ipc-paper) branch for the [ddn ipc paper](https://www.mdpi.com/2313-7673/9/12/755)

## Dependencies

pip install requirements.txt

## Hyperparameter Optimisation

Code for CMA-ES hyperparameter optimisation is included for two benchmark tasks

### NARMA

Optimizing DDNs or baseline ESNs for NARMA tasks can be done by running [NARMA_experiment.py](NARMA_experiment.py) with
the appropriate parameters. Results from previous optimization runs can be found
in [NARMA-10_results_23](NARMA-10_results_23) and [NARMA-30_results_23](NARMA-30_results_23), also used in
the [analysis notebook](analysis.ipynb) to generate paper figures.

### Mackey-Glass

Optimizing DDNs, ADDNs, adaptive ESNs or baseline ESNs for Mackey-Glass tasks can be done by
running [mg_experiment.py](mg_experiment.py) with the appropriate parameters. Results from previous optimization runs
can be found in [ADDN_further_experiments](ADDN_further_experiments), also used in
the [analysis notebook](analysis.ipynb) to generate paper figures.

## Paper Figures and data analysis

All paper figures can be reproduced using the included [jupyter notebook](analysis.ipynb).

## Visual Example

See [visual_example.py]([examples/visual_example.py]) for a DDN or ESN (either random or optimized) simulated with a
GUI, visually showing differences in network responses to various inputs. 