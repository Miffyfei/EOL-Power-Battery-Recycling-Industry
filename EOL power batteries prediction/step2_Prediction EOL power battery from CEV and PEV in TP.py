# -*- coding: utf-8 -*-
"""
TP (Battery Type Installation Adjustment) Scenario + CEV Vehicle Type Proportion × Weight Coefficient Weighting (Only affects BCEV weight)
- Only replace share_df with TP scenario; other parameters completely consistent with Baseline
- Added: BCEV_TYPE_SHARE + BCEV_TYPE_WEIGHT_FACTOR to weight BCEV's "weight (kg/vehicle)"
- Read: ./input/24种电池TP情景占比调整.xlsx (Sheet: PEV, CEV)
- Input sales:
    ./output data_prediction/annual_PEV_data.xlsx (City, Year, PEV_number)
    ./output data_prediction/annual_CEV_data.xlsx (City, Year, CEV_number)
- Output:
    ./output data_prediction/TP_EOL power battery from PEV.xlsx
    ./output data_prediction/TP_EOL power battery from CEV.xlsx
"""

import os
import random
import numpy as np
import pandas as pd
from math import gamma as gamma_fn
from scipy.stats import weibull_min

# =========================
# 0) Global Configuration (Consistent with Baseline)
# =========================
ANALYSIS_YEARS = list(range(2016, 2031))
MAX_AGE = 20
VEHICLE_SHAPE_DEFAULT = 3.5
MAX_REPLACEMENTS = 1

# ---- Technological progress annual segmentation table → Average lifespan μ (years) ----
VEH_LIFE_TABLE = pd.DataFrame({
    "start":   [2015, 2018, 2020, 2023, 2025, 2028, 2030],
    "end":     [2017, 2019, 2022, 2024, 2027, 2029, 2030],
    "pev_mu":  [8, 9, 10, 11, 12, 13, 15],
    "cev_mu":  [5, 6, 7, 8, 9, 10, 12],
})

def vehicle_mean_life_by_sale_year(sale_year: int, is_pev: bool) -> float:
    hit = VEH_LIFE_TABLE[(VEH_LIFE_TABLE["start"] <= sale_year) & (VEH_LIFE_TABLE["end"] >= sale_year)]
    if hit.empty:
        hit = VEH_LIFE_TABLE.iloc[[0 if sale_year < 2015 else -1]]
    return float(hit["pev_mu" if is_pev else "cev_mu"].values[0])

def weibull_scale_from_mean(mu: float, shape: float) -> float:
    return mu / gamma_fn(1.0 + 1.0/shape)

def annual_interval_probs(shape: float, scale: float, max_age: int):
    ages = np.arange(0, max_age + 1, dtype=float)
    F = weibull_min.cdf(ages, shape, scale=scale)  # F(0..A)
    P = F[1:] - F[:-1]                              # Interval probability (a-1,a]
    S_end = 1.0 - F[1:]                             # S(a)
    return P, S_end, F

def renewal_replacement_probs(p_b: np.ndarray, S_end_v: np.ndarray, max_replacements: int):
    """
    Rk[a-1]: The k-th replacement occurs at age a and the vehicle survives until the end of a
      R1[a] = p_b[a] * S_v_end[a]
      Rk[a] = (R_{k-1} * p_b)[a] * S_v_end[a]
    """
    A = len(p_b)
    R_list = []
    R1 = p_b * S_end_v
    R_list.append(R1.copy())
    prev = R1
    for _ in range(2, max_replacements + 1):
        conv = np.zeros(A)
        for i in range(A):
            s = 0.0
            for j in range(i):
                s += prev[j] * p_b[i - 1 - j]
            conv[i] = s
        Rk = conv * S_end_v
        R_list.append(Rk.copy())
        prev = Rk
    return R_list

# =========================
# 1) Baseline Key Parameters (Lifespan/Weight/Capacity), Unchanged
# =========================
PEV_COLS = [
    'BPEV_LFP','BPEV_NCM111','BPEV_NCM523','BPEV_NCM622','BPEV_NCM811','BPEV_NCA',
    'HPEV_LFP','HPEV_NCM111','HPEV_NCM523','HPEV_NCM622','HPEV_NCM811','HPEV_NCA'
]
CEV_COLS = [
    'BCEV_LFP','BCEV_NCM111','BCEV_NCM523','BCEV_NCM622','BCEV_NCM811','BCEV_NCA',
    'HCEV_LFP','HCEV_NCM111','HCEV_NCM523','HCEV_NCM622','HCEV_NCM811','HCEV_NCA'
]

PEV_BATT_WEIBULL = {
    'BPEV_LFP':{'shape':3.5,'scale':9.5},
    'BPEV_NCM111':{'shape':3.5,'scale':8.5},
    'BPEV_NCM523':{'shape':3.5,'scale':8},
    'BPEV_NCM622':{'shape':3.5,'scale':7.5},
    'BPEV_NCM811':{'shape':3.5,'scale':7},
    'BPEV_NCA':{'shape':3.5,'scale':7.5},
    'HPEV_LFP':{'shape':3.5,'scale':10.5},
    'HPEV_NCM111':{'shape':3.5,'scale':9.5},
    'HPEV_NCM523':{'shape':3.5,'scale':9},
    'HPEV_NCM622':{'shape':3.5,'scale':8.5},
    'HPEV_NCM811':{'shape':3.5,'scale':8},
    'HPEV_NCA':{'shape':3.5,'scale':8.5},
}
CEV_BATT_WEIBULL = {
    'BCEV_LFP':{'shape':3.5,'scale':6},
    'BCEV_NCM111':{'shape':3.5,'scale':5},
    'BCEV_NCM523':{'shape':3.5,'scale':4.5},
    'BCEV_NCM622':{'shape':3.5,'scale':4},
    'BCEV_NCM811':{'shape':3.5,'scale':4},
    'BCEV_NCA':{'shape':3.5,'scale':5},
    'HCEV_LFP':{'shape':3.5,'scale':6.5},
    'HCEV_NCM111':{'shape':3.5,'scale':5.5},
    'HCEV_NCM523':{'shape':3.5,'scale':5},
    'HCEV_NCM622':{'shape':3.5,'scale':4.5},
    'HCEV_NCM811':{'shape':3.5,'scale':4.5},
    'HCEV_NCA':{'shape':3.5,'scale':5.5},
}

PEV_BATT_WEIGHT_KG = {
    "BPEV_LFP":350,"BPEV_NCM111":349,"BPEV_NCM523":303,"BPEV_NCM622":305,
    "BPEV_NCM811":479,"BPEV_NCA":208,"HPEV_LFP":357,"HPEV_NCM111":333,
    "HPEV_NCM523":278,"HPEV_NCM622":250,"HPEV_NCM811":227,"HPEV_NCA":278,
}
CEV_BATT_WEIGHT_KG = {
    "BCEV_LFP":790,"BCEV_NCM111":762,"BCEV_NCM523":783,"BCEV_NCM622":833,
    "BCEV_NCM811":846,"BCEV_NCA":926,"HCEV_LFP":160,"HCEV_NCM111":170,
    "HCEV_NCM523":174,"HCEV_NCM622":188,"HCEV_NCM811":190,"HCEV_NCA":204
}

# Fixed capacity caliber (kWh/vehicle); capacity calculation already processed based on "installation year"
random.seed(42)  # For reproducibility
PEV_BATT_E_KWH = {
    'BPEV_LFP':50*random.uniform(0.6,0.8),'BPEV_NCM111':42*random.uniform(0.6,0.8),
    'BPEV_NCM523':40.5*random.uniform(0.6,0.8),'BPEV_NCM622':54*random.uniform(0.6,0.8),
    'BPEV_NCM811':78*random.uniform(0.6,0.8),'BPEV_NCA':75*random.uniform(0.6,0.8),
    'HPEV_LFP':15*random.uniform(0.6,0.8),'HPEV_NCM111':20*random.uniform(0.6,0.8),
    'HPEV_NCM523':18*random.uniform(0.6,0.8),'HPEV_NCM622':35*random.uniform(0.6,0.8),
    'HPEV_NCM811':40*random.uniform(0.6,0.8),'HPEV_NCA':15*random.uniform(0.6,0.8),
}
CEV_BATT_E_KWH = {
    'BCEV_LFP':150*random.uniform(0.6,0.8),'BCEV_NCM111':160*random.uniform(0.6,0.8),
    'BCEV_NCM523':180*random.uniform(0.6,0.8),'BCEV_NCM622':180*random.uniform(0.6,0.8),
    'BCEV_NCM811':200*random.uniform(0.6,0.8),'BCEV_NCA':220*random.uniform(0.6,0.8),
    'HCEV_LFP':30*random.uniform(0.6,0.8),'HCEV_NCM111':35*random.uniform(0.6,0.8),
    'HCEV_NCM523':40*random.uniform(0.6,0.8),'HCEV_NCM622':48*random.uniform(0.6,0.8),
    'HCEV_NCM811':55*random.uniform(0.6,0.8),'HCEV_NCA':64*random.uniform(0.6,0.8),
}

def get_params_baseline_pev(year:int):
    return PEV_BATT_E_KWH, PEV_BATT_WEIGHT_KG
def get_params_baseline_cev(year:int):
    return CEV_BATT_E_KWH, CEV_BATT_WEIGHT_KG

# =========================
# ★ Added: BCEV Vehicle Type Proportion & Weight Coefficient (Only affects BCEV's "weight")
# =========================
BCEV_TYPE_SHARE = {"Light truck":0.7,"Medium truck":0.1,"Heavy truck":0.1,"Bus/large passenger":0.1}
BCEV_TYPE_WEIGHT_FACTOR = {
    "Light truck": 1.00,
    "Medium truck": 1.30,
    "Heavy truck": 2.50,
    "Bus/large passenger": 12.50  # ~800 kg × 12.5 ≈ 10 tons
}
def adjusted_bcev_weight(base_weight: float) -> float:
    s = 0.0
    for t, share in BCEV_TYPE_SHARE.items():
        s += base_weight * share * BCEV_TYPE_WEIGHT_FACTOR.get(t, 1.0)
    return s

# =========================
# 2) Baseline's "Revised Capacity Caliber" Calculator (Plus BCEV Weight Weighting)
# =========================
def compute_city_year_eol_dynamic_params(
    df_sales: pd.DataFrame,
    sales_col: str,
    share_df: pd.DataFrame,
    batt_weibull: dict,
    is_pev: bool,
    out_path: str,
    get_params_for_year,                   # year -> (cap_dict, wt_dict)
    vehicle_shape: float = VEHICLE_SHAPE_DEFAULT,
    max_age: int = MAX_AGE,
    analysis_years: list = ANALYSIS_YEARS,
    max_replacements: int = MAX_REPLACEMENTS
):
    rows = []
    all_models = list(share_df.columns)

    for city in df_sales['City'].unique():
        sub = df_sales[df_sales['City'] == city]
        for y_s in sorted(sub['Year'].unique()):
            if y_s not in share_df.index:
                continue
            N_sales_total = float(sub.loc[sub['Year']==y_s, sales_col].sum())
            if N_sales_total <= 0:
                continue

            # Vehicle lifespan
            mu_v = vehicle_mean_life_by_sale_year(int(y_s), is_pev=is_pev)
            lam_v = weibull_scale_from_mean(mu_v, vehicle_shape)
            Pv, Svend_v, Fv_edge = annual_interval_probs(vehicle_shape, lam_v, max_age)  # Pv[a-1]
            S_v = np.concatenate(([1.0], 1.0 - Fv_edge[1:]))

            for model in all_models:
                share = float(share_df.loc[y_s, model])
                if share <= 0:
                    continue
                Nb = N_sales_total * share

                # Battery lifespan
                k_b = batt_weibull[model]['shape']
                lam_b = batt_weibull[model]['scale']
                Pb, _, Fb_edge = annual_interval_probs(k_b, lam_b, max_age)
                Fb = Fb_edge[1:]
                S_b = np.concatenate(([1.0], 1.0 - Fb))

                # Replacement (R1,R2)
                R_list = renewal_replacement_probs(Pb, Svend_v, max_replacements)
                Rsum = np.sum(np.vstack(R_list), axis=0) if len(R_list)>0 else np.zeros_like(Pb)

                # "Last replacement = r years" survival term with vehicle scrapping (for capacity caliber)
                A = max_age
                last_is_first = np.zeros((A+1, A+1))
                R1 = R_list[0] if len(R_list) >= 1 else np.zeros(A)

                for r in range(1, A+1):
                    for a in range(r+1, A+1):
                        survive_vehicle_ratio = (S_v[a-1] / S_v[r]) if S_v[r] > 0 else 0.0
                        no_second_until_a = S_b[a - r]
                        last_is_first[a, r] = R1[r-1] * no_second_until_a * survive_vehicle_ratio


                # Age a → Year y=y_s+a
                for a in range(1, max_age+1):
                    y = int(y_s + a)
                    if y not in analysis_years:
                        continue

                    # Weight (current year); Only BCEV undergoes "vehicle type proportion × coefficient" weighting
                    _, wt_year_dict = get_params_for_year(y)
                    base_weight = wt_year_dict[model]
                    model_wt_year = adjusted_bcev_weight(base_weight) if (not is_pev and model.startswith("BCEV")) else base_weight

                    count_repl = Nb * Rsum[a-1]
                    count_eol  = Nb * Pv[a-1]
                    count_total = count_repl + count_eol
                    w_thousand_t = count_total * model_wt_year / 1e6

                    # Capacity (installation year caliber)
                    cap_sale_dict, _ = get_params_for_year(int(y_s))
                    cap_sale = cap_sale_dict[model]
                    # Original with vehicle
                    e_gwh_orig = (Nb * Pv[a-1] * (1.0 - Fb[a-1])) * cap_sale / 1e6
                    # Replacement this year (current year capacity)
                    cap_year_dict, _ = get_params_for_year(y)
                    e_gwh_repl = (Nb * Rsum[a-1]) * cap_year_dict[model] / 1e6
                    # Replacement with vehicle (replacement year capacity for last replacement)
                    e_gwh_repl_eol = 0.0
                    for r in range(1, a):
                        cap_at_r = get_params_for_year(int(y_s + r))[0][model]
                        contrib_blocks = Nb * last_is_first[a, r] * Pv[a-1]
                        e_gwh_repl_eol += contrib_blocks * cap_at_r / 1e6


                    e_gwh_total = e_gwh_orig + e_gwh_repl + e_gwh_repl_eol

                    rows.append({
                        "City": city,
                        "Year": y,
                        "Battery type": model,
                        "Retired battery count": count_total,
                        "Weight (thousand t)": w_thousand_t,
                        "Capacity (GWh)": e_gwh_total
                    })

    out = (pd.DataFrame(rows)
           .groupby(["City","Year","Battery type"], as_index=False)
           .sum())
    if out_path and out_path != os.devnull:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_excel(out_path, index=False)
        print(f"Saved: {out_path}")
    return out

# =========================
# 3) Read TP Shares (from Excel)
# =========================
def load_tp_shares(path: str):
    """
    Read PEV/CEV shares from ./input/24种电池TP情景占比调整.xlsx
    - Sheet: 'PEV', 'CEV'
    - Rows: Years (column name 'Year' or '年份')
    - Columns: Various battery types
    - Align for 2016..2030, interpolate/forward-backward fill, and normalize annually (sum=1)
    """
    xls = pd.ExcelFile(path)
    pev_raw = pd.read_excel(xls, 'PEV')
    cev_raw = pd.read_excel(xls, 'CEV')

    def normalize(df, required_cols, year_start=2016, year_end=2030):
        df = df.copy()
        if 'Year' not in df.columns:
            if '年份' in df.columns:
                df.rename(columns={'年份':'Year'}, inplace=True)
            else:
                raise ValueError("Share table must contain 'Year' or '年份' column")
        df['Year'] = df['Year'].astype(int)
        df = df.set_index('Year').sort_index()

        for c in required_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[required_cols]

        idx = pd.Index(range(year_start, year_end+1), name='Year')
        df = df.reindex(idx).interpolate(axis=0).ffill().bfill()

        row_sum = df.sum(axis=1).replace(0, np.nan)
        df = df.div(row_sum, axis=0).fillna(0.0)
        df = df.clip(0.0, 1.0)
        df = df.div(df.sum(axis=1), axis=0).fillna(0.0)
        return df

    pev_share = normalize(pev_raw, PEV_COLS, 2016, 2030)
    cev_share = normalize(cev_raw, CEV_COLS, 2017, 2030)
    return pev_share, cev_share


# =========================
# 4) Main Entry
# =========================
if __name__ == "__main__":
    # Read sales data
    pev_path = './input data/annual_PEV_data.xlsx'
    cev_path = './input data/annual_CEV_data.xlsx'
    df_pev = pd.read_excel(pev_path)
    df_pev['Year'] = df_pev['Year'].astype(int)
    df_cev = pd.read_excel(cev_path)
    df_cev['Year'] = df_cev['Year'].astype(int)

    # Extract city information (including Province and Code)
    pev_city_info = df_pev[['City', 'Province', 'Code']].drop_duplicates()
    cev_city_info = df_cev[['City', 'Province', 'Code']].drop_duplicates()

    # Output paths
    out_pev = './output data/results_EOL power battery/TP_EOL power battery from PEV.xlsx'
    out_cev = './output data/results_EOL power battery/TP_EOL power battery from CEV.xlsx'

    # Read TP share data
    tp_share_path = './input data/battery proportion of 24 in TP.xlsx'
    pev_share_df, cev_share_df = load_tp_shares(tp_share_path)

    # Calculate PEV (no weight adjustment)
    result_pev = compute_city_year_eol_dynamic_params(
        df_sales=df_pev, sales_col='PEV_number',
        share_df=pev_share_df, batt_weibull=PEV_BATT_WEIBULL, is_pev=True,
        out_path=None, get_params_for_year=get_params_baseline_pev
    )

    # Add Province and Code to PEV results
    result_pev = result_pev.merge(pev_city_info, on='City', how='left')
    # Adjust column order
    new_order = ['City', 'Province', 'Code'] + [col for col in result_pev.columns if
                                                col not in ['City', 'Province', 'Code']]
    result_pev = result_pev[new_order]

    if out_pev and out_pev != os.devnull:
        os.makedirs(os.path.dirname(out_pev), exist_ok=True)
        result_pev.to_excel(out_pev, index=False)
        print(f"Saved: {out_pev}")

    # Calculate CEV (only BCEV weight adjusted by vehicle type proportion × coefficient)
    result_cev = compute_city_year_eol_dynamic_params(
        df_sales=df_cev, sales_col='CEV_number',
        share_df=cev_share_df, batt_weibull=CEV_BATT_WEIBULL, is_pev=False,
        out_path=None, get_params_for_year=get_params_baseline_cev
    )

    # Add Province and Code to CEV results
    result_cev = result_cev.merge(cev_city_info, on='City', how='left')
    # Adjust column order
    new_order = ['City', 'Province', 'Code'] + [col for col in result_cev.columns if
                                                col not in ['City', 'Province', 'Code']]
    result_cev = result_cev[new_order]

    if out_cev and out_cev != os.devnull:
        os.makedirs(os.path.dirname(out_cev), exist_ok=True)
        result_cev.to_excel(out_cev, index=False)
        print(f"Saved: {out_cev}")