# EOL-Power-Battery-Recycling-Industry
# Reducing the Environmental Impacts of Battery Recycling through Spatial and Technological Alignment

## Overview
This repository accompanies the paper **"Reducing the environmental impacts of battery recycling through spatial and technological alignment"**. It contains the full modelling stack required to:

- Forecast the end-of-life (EOL) volume of power batteries from passenger electric vehicles (PEV) and commercial electric vehicles (CEV).
- Simulate provincial recycling technology portfolios under multiple scenarios (baseline, technology-oriented, supply-oriented, etc.).
- Quantify life-cycle environmental impacts and cross-regional transport flows that arise from alternative spatial alignments of recycling infrastructure.

The repository is organised into three functional modules that can be executed independently or as a complete workflow:

1. **EOL power batteries prediction** – demand-side projections of battery retirements by city and scenario.
2. **Scenario simulation** – technology portfolio mixes, environmental impacts, and metal recovery under different policy assumptions.
3. **Environment assessment** – transport network modelling and energy-mix sensitivity analysis.

## Repository Structure
```
EOL-Power-Battery-Recycling-Industry/
├── EOL power batteries prediction/        # EOL forecasting models and outputs
│   ├── input data/                        # Raw vehicle stock and socio-economic drivers (Excel)
│   ├── output data/                       # Model artefacts, plots, and forecast tables
│   ├── step1_prediction_CEV.py            # Clustered forecasting workflow for CEV
│   ├── step1_Prediction_PEV.py            # Clustered forecasting workflow for PEV
│   ├── step2_Prediction ... BS.py         # Scenario alignment – baseline
│   ├── step2_Prediction ... ED.py         # Scenario alignment – Economic development
│   ├── step2_Prediction ... LE.py         # Scenario alignment – Low emissions
│   └── step2_Prediction ... TP.py         # Scenario alignment – Technology push
│
├── scenario simulation/                   # Recycling technology portfolio simulations
│   ├── input data/                        # LCA coefficients, scenario-specific adjustments
│   ├── output data/                       # Scenario level environmental and metal recovery summaries
│   ├── Simulation_BS_environment.py       # Baseline scenario calculator
│   ├── Simulation_TO_environment.py       # Technology-oriented scenario calculator
│   ├── Simulation_ES_environment.py       # Energy-structure scenarios (ES1–ES3)
│   ├── Simulation_SU_environment.py       # Supply-oriented scenario calculator
│   ├── Simulation_AR_environment.py       # Alignment & relocation scenario calculator
│   └── sum_comparison total environment impact and metal.ipynb
│                                         # Notebook for aggregating and visualising scenario results
│
└── environment assessment/
    ├── Cross regions transportation/
    │   ├── input data/                    # Transport network, facility capacity, and adjacency matrices
    │   ├── output data/                   # Route-level flow tables for each transport policy
    │   └── CrossTransportation.py         # Transport optimisation and allocation script
    └── output data/                       # Energy-mix sensitivity factors (baseline & TO1–TO3)
```

> **Note:** The current repository follows the layout used for the published experiments. Future refactoring will group shared utilities (e.g., common loaders and plotting helpers) into dedicated modules; contributions that improve modularity are very welcome.

## Prerequisites
- Python **3.9 – 3.11** (tested on 3.10).
- System packages: `git`, `python3-venv`, and a working C/C++ toolchain (required by `xgboost`).
- Recommended hardware: at least 16 GB RAM for the forecasting stage because RandomizedSearchCV performs repeated training.

## Python Dependencies
Install the core scientific stack before running the scripts:

```bash
pip install -U pip
pip install numpy pandas scipy scikit-learn xgboost seaborn matplotlib openpyxl tqdm geopandas networkx ortools
```

The transport optimisation module additionally uses `geopandas`, `networkx`, and `ortools`. If these are unnecessary for your workflow you may omit them, but the full suite is required to reproduce the manuscript results.

## Data Availability
All input Excel and CSV files referenced by the scripts are provided under the corresponding `input data/` folders.

| Module | Key inputs | Description |
| --- | --- | --- |
| `EOL power batteries prediction/input data/` | `2016-2030_PEV and CEV_month+data.xlsx` | Monthly vehicle registrations with socio-economic drivers. |
|  | `annual_PEV_data.xlsx`, `annual_CEV_data.xlsx` | Annual aggregates used for calibration and validation. |
|  | `battery proportion of 24 in TP.xlsx` | Scenario-specific technology shares for TP scenario. |
| `scenario simulation/input data/` | `EOL LFP and NCM battery.xlsx` | Scenario-aligned EOL capacity by province, battery chemistry, and year. |
|  | `Process type proportion_BS.xlsx` and analogous files | Shares of hydrometallurgical / pyrometallurgical processes under each scenario. |
|  | `LCA data*.xlsx` series | Life-cycle impact factors for each treatment pathway. |
| `environment assessment/Cross regions transportation/input data/` | `eol_city_year.csv`, `dist_city_city.csv`, `province_adjacency.csv`, etc. | City-level EOL supply, inter-city distances, and province-level constraints for transport modelling. |

All outputs are written to the matching `output data/` directories. Delete or rename existing files if you want to regenerate results from scratch.

## Quick Start
### 1. Clone the Repository
```bash
git clone https://github.com/<user>/EOL-Power-Battery-Recycling-Industry.git
cd EOL-Power-Battery-Recycling-Industry
```

### 2. Create a Virtual Environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or use the command listed above
```

### 3. Run an End-to-End Workflow
The workflow consists of three stages. You may execute them sequentially or run a single module as needed.

1. **Forecast EOL volumes**
   ```bash
   cd "EOL power batteries prediction"
   python step1_Prediction_PEV.py   # Generates PEV clusters, model diagnostics, and forecasts
   python step1_prediction_CEV.py   # Generates CEV counterparts
   python "step2_Prediction EOL power battery from CEV and PEV in BS.py"
   python "step2_Prediction EOL power battery from CEV and PEV in TP.py"
   python "step2_Prediction EOL power battery from CEV and PEV in ED.py"
   python "step2_Prediction EOL power battery from CEV and PEV in LE.py"
   ```
   The step-2 scripts merge PEV & CEV forecasts with scenario-specific parameters and save harmonised provincial outputs (Excel workbooks) for downstream modules.

2. **Simulate recycling scenarios**
   ```bash
   cd "../scenario simulation"
   python Simulation_BS_environment.py
   python Simulation_TO_environment.py
   python Simulation_SU_environment.py
   python Simulation_ES_environment.py
   python Simulation_AR_environment.py
   ```
   Each script reads the harmonised EOL datasets, applies technology proportions, computes environmental impacts, and writes Excel summaries to `output data/`. The optional notebook `sum_comparison total environment impact and metal.ipynb` consolidates scenario outputs into comparison tables and figures.

3. **Assess transport and energy-mix impacts**
   ```bash
   cd "../environment assessment/Cross regions transportation"
   python CrossTransportation.py
   ```
   This module allocates EOL flows across provinces under multiple routing policies (baseline intra-province, neighbour provinces, 300 km radius, etc.), and exports route-level and province-level flow tables. Energy-mix sensitivity factors for TO1–TO3 scenarios are computed separately and stored in `environment assessment/output data/`.

Return to the repository root once the workflow is complete:
```bash
cd ../../
```

## Example: Inspecting Outputs
- `EOL power batteries prediction/output data/results_electric vehicle/PEV_ClusterBest_Forecast.xlsx` – contains time-series forecasts by cluster and city for PEV fleets.
- `scenario simulation/output data/Environmental impact and metal recovery results under BS scenario.xlsx` – summarises life-cycle impact scores and recovered metals for the baseline scenario.
- `environment assessment/Cross regions transportation/output data/flows_baseline_inprov.csv` – annual in-province flow allocations used in the transport analysis.

## Customising the Experiments
- **Update socio-economic drivers:** replace the Excel files in `EOL power batteries prediction/input data/`. Ensure column names align with those expected by the scripts (see constants near the top of each file).
- **Change scenario proportions:** edit the process proportion workbooks in `scenario simulation/input data/` and rerun the corresponding scenario script.
- **Modify transport constraints:** adjust facility capacities (`formal_capacity_by_province.csv`) or adjacency relationships (`province_adjacency.csv`) before re-running `CrossTransportation.py`.

## Troubleshooting
- **Missing dependency errors:** ensure the Python environment matches the versions listed above. Some packages (e.g., `xgboost`, `ortools`) require compilation; install system build tools if pip wheels are unavailable for your platform.
- **Excel write permissions:** scripts overwrite existing Excel sheets. Close any open workbooks before running the code to avoid `PermissionError`.
- **Memory errors during model fitting:** reduce `N_TRIALS` in the step-1 prediction scripts to limit RandomizedSearchCV iterations or run clustering/model fitting per city cluster.

## How to Cite
If you use this repository, please cite the associated manuscript:

> *Reducing the environmental impacts of battery recycling through spatial and technological alignment*, 2024.

```
@article{<placeholder>,
  title={Reducing the environmental impacts of battery recycling through spatial and technological alignment},
  year={2024},
  author={Xi Tian, Fei Peng, Jon McKechnie, Amir F.N. Abdul-Manan, Yaobin Liu, Fanran Mengg*},
  note={Code available at https://github.com/<user>/EOL-Power-Battery-Recycling-Industry}
}
```

## Contact
For questions or collaboration proposals, please open an issue in the repository or contact the corresponding author via email listed in the manuscript.
