# Reducing the Environmental Impacts of Battery Recycling through Spatial and Technological Alignment

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data Availability](#data-availability)
- [System & Software Requirements](#system--software-requirements)
- [Quick Start](#quick-start)
- [Recommended Workflow](#recommended-workflow)
  - [1. Predict EOL battery streams](#1-predict-eol-battery-streams)
  - [2. Generate EOL scenarios](#2-generate-eol-scenarios)
  - [3. Quantify environmental impacts](#3-quantify-environmental-impacts)
- [Module Reference](#module-reference)
  - [Prediction module](#prediction-module)
  - [Environmental assessment module](#environmental-assessment-module)
  - [Scenario simulation module](#scenario-simulation-module)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contact](#contact)
  
## Project Overview
This repository accompanies the paper **"Reducing the environmental impacts of battery recycling through spatial and technological alignment"**. It contains the full modelling stack required to:

1. **EOL power batteries prediction** for passenger electric vehicles (PEV) and commercial electric vehicles (CEV) using machine-learning models with automatic clustering, feature engineering, and cross-validation.
2. **Environmental impact assessment** transport network modelling and energy-mix sensitivity analysis. This part of the environmental impact results is mainly calculated through the openlca software.
3. **Scenario simulation** for 52 supply–demand combinations, including detailed metal recovery balances and cross-regional transport allocation to recycling facilities.

Executing the workflow reproduces the paper's quantitative outputs. Each script can also be run independently to evaluate alternative scenarios or updated datasets.


## Repository Structure
```
.
├── EOL power batteries prediction/
│   ├── input data/                # Vehicle sales, stock, and auxiliary socio-economic variables
│   ├── output data/               # Model diagnostics and forecast results (generated)
│   ├── step1_prediction_CEV.py            # Clustered forecasting workflow for CEV
│   ├── step1_Prediction_PEV.py            # Clustered forecasting workflow for PEV
│   ├── step2_Prediction ... BS.py         # Scenario alignment – baseline
│   ├── step2_Prediction ... ED.py         # Scenario alignment – Enhance battery energy density
│   ├── step2_Prediction ... LE.py         # Scenario alignment – Life extension
│   └── step2_Prediction ... TP.py         # Scenario alignment – Adjust NCM battery to dominant
├── environment assessment/
│   ├── Cross regions transportation/
│   │   ├── input data/            # GIS-derived distances, facility capacities, enterprise counts
│   │   ├── output data/           # Annual transport flows by scenario (generated)
│   │   └── CrossTransportation.py # Formal & informal flow allocation and ton-kilometre calculation
│   └── output data/               # Energy-mix impact factors used by scenario simulations
├── scenario simulation/
│   ├── input data/                # Life-cycle inventory (LCI) data, scenario parameters, battery mix
│   ├── output data/               # Impact results for each supply-side scenario (generated)
│   ├── Simulation_BS_environment.py       # Baseline scenario calculator
│   ├── Simulation_TO_environment.py       # Technology-oriented scenario calculator
│   ├── Simulation_ES_environment.py       # Energy-structure scenarios (ES1–ES3)
│   ├── Simulation_SU_environment.py       # Increase secondary use of LFP batteries
│   ├── Simulation_AR_environment.py       # Increase the authorized enterprises recycing rate
│   └── sum_comparison total environment impact and metal.ipynb
└── README.md
```

> **Note:** The current repository follows the layout used for the published experiments. Future refactoring will group shared utilities (e.g., common loaders and plotting helpers) into dedicated modules; contributions that improve modularity are very welcome.

## System & Software Requirements
The code has been executed on the following configurations.

### Hardware
- CPU: Intel Core i5 (or equivalent)
- RAM: ≥ 8 GB
- Storage: ≥ 1 GB free space (to accommodate intermediate Excel exports)

### Operating Systems
| OS      | Tested Versions |
|---------|-----------------|
| Windows | 10, 11          |
| macOS   | 12 (Monterey)+  |
| Linux   | Ubuntu 20.04, 22.04 |

### Python
Python 3.8–3.11 are supported; 3.10 is recommended for parity with the paper.

### Python Dependencies
Install the core scientific stack before running any script:

```bash
pip install pandas==1.5.3 numpy==1.23.5 scikit-learn==1.2.2 \
            xgboost==1.7.6 tensorflow==2.11.0 keras==2.11.0 \
            mlxtend==0.21.0 scipy==1.10.1 seaborn==0.12.2 \
            matplotlib==3.7.1 openpyxl==3.1.2 geopandas==0.12.2 \
            rasterio==1.3.7
```

> Tip: create an isolated environment (e.g., `conda create -n eol-battery python=3.10`) before installing dependencies to avoid conflicts with system packages.

## Data Availability
All source data required to run the scripts are provided in the `input data` folders of each module. The repository also includes pre-computed outputs in the corresponding `output data` directories so results can be inspected without rerunning the models. When updating datasets, maintain the same schema as the supplied examples:

- **Excel workbooks** are read using sheet names shown in each script (e.g., `Sheet1` for `2016-2030_PEV and CEV_month+data.xlsx`).
- **CSV files** use UTF-8 encoding and comma separators with headers in the first row.
- **Units**:
  - Vehicle activity data are expressed in counts or thousand units as indicated in the spreadsheets.
  - Battery EOL outputs are in thousand tonnes (Weight) and gigawatt-hours (Capacity).
  - Transport outputs use tonnes, kilometres, and tonne-kilometres.

## Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/<org>/EOL-Power-Battery-Recycling-Industry.git
   cd EOL-Power-Battery-Recycling-Industry
   ```
2. **Create and activate a Python environment** (optional but recommended)
   ```bash
   conda create -n eol-battery python=3.10
   conda activate eol-battery
   ```
3. **Install dependencies** using the command shown above.
4. **Run the end-to-end workflow** (see [Recommended Workflow](#recommended-workflow)) or execute individual modules as needed.

## Recommended Workflow
The following sequence reproduces the manuscript figures and tables. All paths are relative to the repository root.

### 1. Predict EOL battery streams
1. **Passenger electric vehicles (PEV)**
   ```bash
   python "EOL power batteries prediction/step1_Prediction_PEV.py"
   ```
2. **Commercial electric vehicles (CEV)**
   ```bash
   python "EOL power batteries prediction/step1_prediction_CEV.py"
   ```

These scripts automatically:
- Clean and cluster the provincial time series.
- Train ensemble models with cross-validation.
- Export diagnostics (charts, metrics) and province-level forecasts to `EOL power batteries prediction/output data/results_electric vehicle/` and `.../results_EOL power battery/`.

> Outputs: `*_ClusterBest_Forecast.xlsx` (model diagnostics) and `*_Forecast_by_province.xlsx` (annual retirements).

### 2. Generate EOL scenarios
Convert vehicle retirements into EOL battery volumes under the four technology pathways:

```bash
python "EOL power batteries prediction/step2_Prediction EOL power battery from CEV and PEV in BS.py"
python "EOL power batteries prediction/step2_Prediction EOL power battery from CEV and PEV in TP.py"
python "EOL power batteries prediction/step2_Prediction EOL power battery from CEV and PEV in ED.py"
python "EOL power batteries prediction/step2_Prediction EOL power battery from CEV and PEV in LE.py"
```

Each script reads the outputs from Step 1, combines them with technology adoption assumptions, and produces pathway-specific EOL battery inventories saved in `EOL power batteries prediction/output data/results_EOL power battery/`.

### 3. Quantify environmental impacts

1. **Allocate cross-regional transport flows** (optional if using provided transport outputs):
   ```bash
   python "environment assessment/Cross regions transportation/CrossTransportation.py"
   ```
   Generates annual tonne-kilometre balances for baseline and three alternative spatial coordination policies in `environment assessment/Cross regions transportation/output data/`.

2. **Simulate environmental impacts and metal recovery** for each supply-side strategy:
   ```bash
   python "scenario simulation/Simulation_BS_environment.py"
   python "scenario simulation/Simulation_TO_environment.py"
   python "scenario simulation/Simulation_ES_environment.py"
   python "scenario simulation/Simulation_SU_environment.py"
   python "scenario simulation/Simulation_AR_environment.py"
   ```

   The scripts combine life-cycle impact factors (`scenario simulation/input data/`) with EOL scenario inventories to compute 11 environmental indicators and recovered metal quantities. Results are exported to Excel workbooks in `scenario simulation/output data/`, including a consolidated file covering all 52 combined scenarios.

## Module Reference

### Prediction module
| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `step1_Prediction_PEV.py` | Clusters provinces and trains machine-learning models to forecast PEV retirements through 2030. | `input data/2016-2030_PEV and CEV_month+data.xlsx` (`Sheet1`) | `output data/results_electric vehicle/PEV_*.xlsx`, diagnostic PNGs |
| `step1_prediction_CEV.py` | Same as above for CEV fleet with tailored feature set and hyperparameters. | `input data/2016-2030_PEV and CEV_month+data.xlsx` (`Sheet1`) | `output data/results_electric vehicle/CEV_*.xlsx`, diagnostic PNGs |
| `step2_Prediction EOL power battery from CEV and PEV in BS.py` | Combines Step 1 forecasts with Baseline recycling technology shares. | `output data/results_electric vehicle/*Forecast_by_province.xlsx`, `input data/annual_*_data.xlsx` | `output data/results_EOL power battery/BS_*.xlsx` |
| `step2_Prediction EOL power battery from CEV and PEV in TP.py` | Shift towards NCM as the dominant batteries | Same as above plus `battery proportion of 24 in TP.xlsx` | `output data/results_EOL power battery/TP_*.xlsx` |
| `step2_Prediction EOL power battery from CEV and PEV in ED.py` | Enhancement energy density pathway | Step 1 outputs, annual data | `output data/results_EOL power battery/ED_*.xlsx` |
| `step2_Prediction EOL power battery from CEV and PEV in LE.py` | Battery life extension pathway | Step 1 outputs, annual data | `output data/results_EOL power battery/LE_*.xlsx` |

### Environmental assessment module
| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `Cross regions transportation/CrossTransportation.py` | Allocates EOL flows between provinces, calculates tonne-kilometres, and summarises informal vs. formal collection. | `Cross regions transportation/input data/*.csv` | `Cross regions transportation/output data/*_transport_results.xlsx` |

The folder `environment assessment/output data/` stores energy mix impact factors already linked to the scenario simulations. Update these spreadsheets if regional electricity mixes or process inventories change.

### Scenario simulation module
| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `Simulation_BS_environment.py` | Calculates environmental burdens and recovered metals for the Baseline supply-side strategy. | `input data/LCA data.xlsx`, `input data/Process type proportion_BS.xlsx`, `input data/EOL LFP and NCM battery.xlsx` | `output data/Environmental impact and metal recovery results under BS scenario.xlsx` |
| `Simulation_TO_environment.py` | Technology optimisation supply-side assumptions. | `input data/100%TOSumLCA.xlsx` and shared LCA files | `output data/Environmental impact and metal recovery results under TO scenario.xlsx` |
| `Simulation_ES_environment.py` | Energy structure scenarios (ES1–ES3). | `input data/LCA data about ES*.xlsx` | `output data/Environmental impact and metal recovery results under ES scenario.xlsx` |
| `Simulation_SU_environment.py` | Increased secondary-use scenario. | `input data/LCA data with SU.xlsx` | `output data/Environmental impact and metal recovery results under SU scenario.xlsx` |
| `Simulation_AR_environment.py` | Advanced authorized recovery scenario. | `input data/Environmental_impacts_with_transport under4 scenario.xlsx` | `output data/Environmental impact and metal recovery results under AR scenario.xlsx` |
| `sum_comparison total environment impact and metal.ipynb` | Interactive workbook to compare results across scenarios and prepare figures. | Scenario outputs listed above | Visualisations and aggregate tables |

## Troubleshooting
| Issue | Possible cause & resolution |
|-------|----------------------------|
| `ImportError` for scientific packages | Confirm the virtual environment is active and dependencies are installed with the versions listed above. |
| Excel writer errors (`PermissionError`) | Ensure output workbooks are closed before rerunning scripts; Windows users may need to close Excel completely. |
| Missing column errors | Verify customised input files preserve header names and data types expected by the scripts (see inline assertions within each script). |
| Long runtime during model tuning | Reduce `N_TRIALS`/`N_SPLITS` parameters near the top of the Step 1 scripts for exploratory runs. |

## Citation
If you use this repository, please cite the associated article:

>  **Reducing the environmental impacts of battery recycling through spatial and technological alignment.**

```
@article{<placeholder>,
  title={Reducing the environmental impacts of battery recycling through spatial and technological alignment},
  year={2025},
  author={Xi Tian, Fei Peng, Jon McKechnie, Amir F.N. Abdul-Manan, Yaobin Liu, Fanran Meng*},
  note={Code available at https://github.com/<user>/EOL-Power-Battery-Recycling-Industry}
}
```

## Contact
For questions or collaboration requests:

- Prof. Fanran Meng – f.meng@sheffield.ac.uk
- Prof. Xi Tian – tianxi@ncu.edu.cn
- Ms. Fei Peng – pengfei24@email.ncu.edu.cn

Affiliations: Nanchang University (China) and University of Sheffield (UK).
