# Redefining Epidemiological Waves: Structural Stability and Global Empirical Validation in Sobolev Spaces

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official Python implementation for the paper:
**"Redefining Epidemiological Waves: Structural Stability and Global Empirical Validation in Sobolev Spaces"**

**Author:** Santi García-Cremades (Miguel Hernandez University of Elche)

## Overview
Epidemiological data (such as COVID-19 incidence) is often plagued by institutional noise, reporting delays, and weekend artifacts. Differentiating these raw signals to find epidemic peaks creates extreme mathematical instability. 

This repository provides a structurally stable algorithm that:
1. Projects noisy incidence data into the **Sobolev space $H^3$** using sparse Tikhonov regularization.
2. Identifies the exact geometric boundaries (start, peak, end) of epidemic waves using **kinematic derivatives** (velocity and acceleration).
3. Evaluates the global *Topological Condemnation* (total days spent in structural instability) across 170+ nations using JHU and World Bank data.

## Files in this Repository
* `redefining_epidemiological_waves.ipynb`: The main reproducible Jupyter Notebook containing all the mathematical logic and the global execution loop.
* `Country_Summary_H3.csv`: The final dataset summarizing the total epidemic days and wave counts for over 170 countries.
* `Detailed_Waves_H3.csv`: A detailed log of every single wave detected globally, including exact phase transition dates.
* `figures/`: A folder containing the generated topo-kinematic graphs for all analyzed countries.

## How to Run
To reproduce the global results from the paper, simply install the required dependencies and run the notebook or script.

```bash
pip install numpy pandas scipy matplotlib requests
python main.py
