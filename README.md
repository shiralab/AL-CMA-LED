# Augmented Lagrangian CMA-ES-LED

## bbob-constrained
This folder contains the code for experiments using the BBOB-constrained benchmark suite.

- `bbob_main.py`: the main script implementing the core algorithms and optimization methods
- `bbob_run_test.py`: the script for running the benchmarking experiments
  - experimental data is saved in the `exdata/` directory
- `bbob_plot_bet_m.py`: after running the post-processing script (`bbob_plot_bet_m.py`), the analyzed results and plots are grouped by the number of constraints and topology (unimodal/multimodal)
   - experimental data is saved in the `ppdata_m_{m_label}_{topology}/` directory

## sphere_demo
This folder contains the code for verification experiments using the Sphere function with linear constraints.

- `main.py`: the core script implementing the algorithms for each method
- `monitor_test.py`: the script for running the verification experiments
  - detailed logs for each trial (including metrics like $N_{\mathrm{eff}}$, condition numbers, $\sigma$, etc.) are saved as CSV files in the `Results_BBOB_dim.../` directory
  - a summary of all trials (e.g., total evaluations to target) is generated as `summary_BBOB_dim...csv`
- `benchmark_problems.py`: defines the objective functions and constraints (the number of constraints can be modified within this file)
- `analyze_metrics.py`: A script to plot the transition of the estimated number of effective dimensions ($N_{eff}$)
  - the plot is saved as PNG images in the `Images/` directory

## Reference
H.Nakagawa, K.Uchida and S.Shirakawa, "Evaluation of Element-wise Effectiveness Estimation
for Augmented Lagrangian CMA-ES," 2026 Genetic and Evolutionary Computation Conference (GECCO), San José, Costa Rica, 2026
