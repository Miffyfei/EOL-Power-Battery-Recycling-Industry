# -*- coding: utf-8 -*-
"""
Lifespan Extension (LE) Scenario + CEV Vehicle Type Proportion × Weight Coefficient Weighting (Only affects BCEV weight)
- Consistent with Baseline/ED approach: Capacity based on installation year caliber, maximum two replacements
- Only difference: Battery lifespan Weibull's scale multiplied by le_scale_factor(install_year) based on "installation year"
- Only output LE results; do not repeat Baseline/ED calculations
- Output:
  ./output data_prediction/LE_EOL power battery from PEV.xlsx
  ./output data_prediction/LE_EOL power battery from CEV.xlsx
"""

import os
import random
import numpy as np
import pandas as pd
from math import gamma as gamma_fn
from scipy.stats import weibull_min

# =========================
# 0) Global Configuration
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
    return P, S_end, F  # F[1:] = F(a)

# =========================
# 1) Key Parameters (Consistent with Baseline/ED)
# =========================
pev_years = list(range(2016, 2031))
PEV_SHARE = pd.DataFrame({
    'BPEV_LFP': [0.543065476,0.355647668,0.29184876,0.251713961,0.288924559,0.4394,0.49042,0.49781,0.633802817,0.633802817,0.633802817,0.633802817,0.633802817,0.633802817,0.633802817],
    'BPEV_NCM111':[0.028836012,0.054317098,0.036518511,0.016046765,0.020545746,0.006929,0,0,0,0,0,0,0,0,0],
    'BPEV_NCM523':[0.168579762,0.33495544,0.301277719,0.336982066,0.27223114,0.174408,0.0791,0.03715,0.021126761,0.021126761,0.021126761,0.021126761,0.021126761,0.021126761,0.021126761],
    'BPEV_NCM622':[0.022181548,0.058843523,0.082166651,0.101629512,0.102728732,0.068952,0.07119,0.05944,0.006338028,0.006338028,0.006338028,0.006338028,0.006338028,0.006338028,0.006338028],
    'BPEV_NCM811':[0.001109077,0.002263212,0.018259256,0.058838138,0.113001605,0.146016,0.14238,0.14117,0.126760563,0.126760563,0.126760563,0.126760563,0.126760563,0.126760563,0.126760563],
    'BPEV_NCA':[0.001109077,0.002263212,0.018259256,0.021395687,0.005136437,0.009295,0.00791,0.00743,0.057042254,0.057042254,0.057042254,0.057042254,0.057042254,0.057042254,0.057042254],
    'HPEV_LFP':[0.166934524,0.084352332,0.09815124,0.068286039,0.071075441,0.0806,0.12958,0.17219,0.116197183,0.116197183,0.116197183,0.116197183,0.116197183,0.116197183,0.116197183],
    'HPEV_NCM111':[0.008863988,0.012882902,0.012281489,0.004353235,0.005054254,0.001271,0,0,0,0,0,0,0,0,0],
    'HPEV_NCM523':[0.051820238,0.07944456,0.101322281,0.091417934,0.06696886,0.031992,0.0209,0.01285,0.003873239,0.003873239,0.003873239,0.003873239,0.003873239,0.003873239,0.003873239],
    'HPEV_NCM622':[0.006818452,0.013956477,0.027633349,0.027570488,0.025271268,0.012648,0.01881,0.02056,0.001161972,0.001161972,0.001161972,0.001161972,0.001161972,0.001161972,0.001161972],
    'HPEV_NCM811':[0.000340923,0.000536788,0.006140744,0.015961862,0.027798395,0.026784,0.03762,0.04883,0.023239437,0.023239437,0.023239437,0.023239437,0.023239437,0.023239437,0.023239437],
    'HPEV_NCA':[0.000340923,0.000536788,0.006140744,0.005804313,0.001263563,0.001705,0.00209,0.00257,0.010457746,0.010457746,0.010457746,0.010457746,0.010457746,0.010457746,0.010457746]
}, index=pev_years)

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
PEV_BATT_WEIGHT_KG = {
    "BPEV_LFP":350,"BPEV_NCM111":349,"BPEV_NCM523":303,"BPEV_NCM622":305,
    "BPEV_NCM811":479,"BPEV_NCA":208,"HPEV_LFP":357,"HPEV_NCM111":333,
    "HPEV_NCM523":278,"HPEV_NCM622":250,"HPEV_NCM811":227,"HPEV_NCA":278,
}
random.seed(42)
PEV_BATT_E_KWH = {
    'BPEV_LFP':50*random.uniform(0.7,0.8),'BPEV_NCM111':42*random.uniform(0.7,0.8),'BPEV_NCM523':40.5*random.uniform(0.7,0.8),'BPEV_NCM622':54*random.uniform(0.7,0.8),
    'BPEV_NCM811':78*random.uniform(0.7,0.8),'BPEV_NCA':75*random.uniform(0.7,0.8),'HPEV_LFP':15*random.uniform(0.7,0.8),'HPEV_NCM111':20*random.uniform(0.7,0.8),
    'HPEV_NCM523':18*random.uniform(0.7,0.8),'HPEV_NCM622':35*random.uniform(0.7,0.8),'HPEV_NCM811':40*random.uniform(0.7,0.8),'HPEV_NCA':15*random.uniform(0.7,0.8),
}

# ---- Commercial Vehicle ----
cev_years = list(range(2017, 2031))
CEV_SHARE = pd.DataFrame({
    'BCEV_LFP':[0.410943396,0.376493506,0.306554622,0.345123967,0.508817204,0.607159763,0.6566,0.737810877,0.737810877,0.737810877,0.737810877,0.737810877,0.737810877,0.737810877],
    'BCEV_NCM111':[0.062762264,0.047109957,0.019542857,0.024542149,0.008023656,0,0,0,0,0,0,0,0,0],
    'BCEV_NCM523':[0.387033962,0.388657143,0.4104,0.325183471,0.20196129,0.097928994,0.049,0.024593696,0.024593696,0.024593696,0.024593696,0.024593696,0.024593696,0.024593696],
    'BCEV_NCM622':[0.067992453,0.105997403,0.123771429,0.122710744,0.079845161,0.088136095,0.0784,0.007378109,0.007378109,0.007378109,0.007378109,0.007378109,0.007378109,0.007378109],
    'BCEV_NCM811':[0.002615094,0.023554978,0.071657143,0.134981818,0.169083871,0.176272189,0.1862,0.147562175,0.147562175,0.147562175,0.147562175,0.147562175,0.147562175,0.147562175],
    'BCEV_NCA':[0.002615094,0.023554978,0.026057143,0.006135537,0.010763441,0.009792899,0.0098,0.066402979,0.066402979,0.066402979,0.066402979,0.066402979,0.066402979,0.066402979],
    'HCEV_LFP':[0.029056604,0.013506494,0.013445378,0.014876033,0.011182796,0.012840237,0.0134,0.012189123,0.012189123,0.012189123,0.012189123,0.012189123,0.012189123,0.012189123],
    'HCEV_NCM111':[0.004437736,0.001690043,0.000857143,0.001057851,0.000176344,0,0,0,0,0,0,0,0,0],
    'HCEV_NCM523':[0.027366038,0.013942857,0.018,0.014016529,0.00443871,0.002071006,0.001,0.000406304,0.000406304,0.000406304,0.000406304,0.000406304,0.000406304,0.000406304],
    'HCEV_NCM622':[0.004807547,0.003802597,0.005428571,0.005289256,0.001754839,0.001863905,0.0016,0.000121891,0.000121891,0.000121891,0.000121891,0.000121891,0.000121891,0.000121891],
    'HCEV_NCM811':[0.000184906,0.000845022,0.003142857,0.005818182,0.003716129,0.003727811,0.0038,0.002437825,0.002437825,0.002437825,0.002437825,0.002437825,0.002437825,0.002437825],
    'HCEV_NCA':[0.000184906,0.000845022,0.001142857,0.000264463,0.000236559,0.000207101,0.0002,0.001097021,0.001097021,0.001097021,0.001097021,0.001097021,0.001097021,0.001097021]
}, index=cev_years)

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
CEV_BATT_WEIGHT_KG = {
    "BCEV_LFP":790,"BCEV_NCM111":762,"BCEV_NCM523":783,"BCEV_NCM622":833,
    "BCEV_NCM811":846,"BCEV_NCA":926,"HCEV_LFP":160,"HCEV_NCM111":170,
    "HCEV_NCM523":174,"HCEV_NCM622":188,"HCEV_NCM811":190,"HCEV_NCA":204
}
random.seed(42)
CEV_BATT_E_KWH = {
    'BCEV_LFP':150*random.uniform(0.7,0.8),'BCEV_NCM111':160*random.uniform(0.7,0.8),'BCEV_NCM523':180*random.uniform(0.7,0.8),'BCEV_NCM622':180*random.uniform(0.7,0.8),
    'BCEV_NCM811':200*random.uniform(0.7,0.8),'BCEV_NCA':220*random.uniform(0.7,0.8),'HCEV_LFP':30*random.uniform(0.7,0.8),'HCEV_NCM111':35*random.uniform(0.7,0.8),
    'HCEV_NCM523':40*random.uniform(0.7,0.8),'HCEV_NCM622':48*random.uniform(0.7,0.8),'HCEV_NCM811':55*random.uniform(0.7,0.8),'HCEV_NCA':64*random.uniform(0.7,0.8),
}

def get_params_baseline_pev(year:int):
    return PEV_BATT_E_KWH, PEV_BATT_WEIGHT_KG
def get_params_baseline_cev(year:int):
    return CEV_BATT_E_KWH, CEV_BATT_WEIGHT_KG

# ★ Only for BCEV weight: Vehicle type proportion + weight coefficient (PEV not used)
BCEV_TYPE_SHARE = {"Light truck":0.7,"Medium truck":0.1,"Heavy truck":0.1,"Bus/large passenger":0.1}
BCEV_TYPE_WEIGHT_FACTOR = {"Light truck":1.0,"Medium truck":1.3,"Heavy truck":2.5,"Bus/large passenger":12.5}
def adjusted_bcev_weight(base_weight: float) -> float:
    return sum(base_weight * BCEV_TYPE_SHARE[t] * BCEV_TYPE_WEIGHT_FACTOR.get(t,1.0) for t in BCEV_TYPE_SHARE)

# =========================
# 2) LE: Scale adjustment factor for battery lifespan based on "installation year"
# =========================
def le_scale_factor(install_year:int) -> float:
    # Example: 7% extension starting from 2025
    return 1.07 if install_year >= 2025 else 1.0

# =========================
# 3) Calculator (Revised capacity caliber + LE lifespan + BCEV weight weighted)
# =========================
def compute_city_year_eol_with_LE(
    df_sales: pd.DataFrame,
    sales_col: str,
    share_df: pd.DataFrame,
    batt_weibull: dict,
    is_pev: bool,
    out_path: str,
    get_params_for_year,                 # year -> (cap_dict, wt_dict)
    vehicle_shape: float = VEHICLE_SHAPE_DEFAULT,
    max_age: int = MAX_AGE,
    analysis_years: list = ANALYSIS_YEARS,
    max_replacements: int = MAX_REPLACEMENTS
):
    rows = []
    all_models = list(share_df.columns)

    def pb_sb_for_install_year(k_b, lam_base, install_year):
        lam = lam_base * le_scale_factor(install_year)
        Pb, _, Fb_edge = annual_interval_probs(k_b, lam, max_age)
        Sb = np.concatenate(([1.0], 1.0 - Fb_edge[1:]))
        return Pb, Sb, Fb_edge[1:]

    for city in df_sales['City'].unique():
        sub = df_sales[df_sales['City'] == city]
        for y_s in sorted(sub['Year'].unique()):
            if y_s not in share_df.index:
                continue
            N_sales_total = float(sub.loc[sub['Year']==y_s, sales_col].sum())
            if N_sales_total <= 0:
                continue

            mu_v = vehicle_mean_life_by_sale_year(int(y_s), is_pev=is_pev)
            lam_v = weibull_scale_from_mean(mu_v, vehicle_shape)
            Pv, Svend_v, Fv_edge = annual_interval_probs(vehicle_shape, lam_v, max_age)
            S_v = np.concatenate(([1.0], 1.0 - Fv_edge[1:]))

            for model in all_models:
                share = float(share_df.loc[y_s, model]); 
                if share <= 0: 
                    continue
                Nb = N_sales_total * share

                k_b = batt_weibull[model]['shape']
                lam_b_base = batt_weibull[model]['scale']
                Pb_orig, S_b_orig, Fb_orig = pb_sb_for_install_year(k_b, lam_b_base, int(y_s))

                # R1 (First replacement)
                R1 = Pb_orig * Svend_v
                A = max_age

                # Only allow one replacement: Construct "last replacement = first" survival term with vehicle scrapping
                last_is_first = np.zeros((A + 1, A + 1))
                for r in range(1, A + 1):
                    _, S_b_after_r, _ = pb_sb_for_install_year(k_b, lam_b_base, int(y_s + r))
                    for a in range(r + 1, A + 1):
                        survive_vehicle_ratio = (S_v[a - 1] / S_v[r]) if S_v[r] > 0 else 0.0
                        no_second_until_a = S_b_after_r[a - r]  # "No second failure until year a after replacement"
                        last_is_first[a, r] = R1[r - 1] * no_second_until_a * survive_vehicle_ratio

                # Only one replacement
                Rsum = R1

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
                    e_gwh_orig = (Nb * Pv[a-1] * (1.0 - Fb_orig[a-1])) * cap_sale / 1e6

                    cap_year_dict, _ = get_params_for_year(y)
                    e_gwh_repl = (Nb * Rsum[a-1]) * cap_year_dict[model] / 1e6

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
# 4) Main Entry
# =========================
if __name__ == "__main__":
    pev_path = './input data/annual_PEV_data.xlsx'
    cev_path = './input data/annual_CEV_data.xlsx'
    df_pev = pd.read_excel(pev_path);
    df_pev['Year'] = df_pev['Year'].astype(int)
    df_cev = pd.read_excel(cev_path);
    df_cev['Year'] = df_cev['Year'].astype(int)

    # Extract city information (including Province and Code)
    pev_city_info = df_pev[['City', 'Province', 'Code']].drop_duplicates()
    cev_city_info = df_cev[['City', 'Province', 'Code']].drop_duplicates()

    out_pev = './output data/results_EOL power battery/LE_EOL power battery from PEV.xlsx'
    out_cev = './output data/results_EOL power battery/LE_EOL power battery from CEV.xlsx'

    # Calculate PEV results
    result_pev = compute_city_year_eol_with_LE(
        df_sales=df_pev, sales_col='PEV_number',
        share_df=PEV_SHARE, batt_weibull=PEV_BATT_WEIBULL, is_pev=True,
        out_path=None, get_params_for_year=get_params_baseline_pev
    )

    # Add Province and Code to PEV results
    result_pev = result_pev.merge(pev_city_info, on='City', how='left')
    # Adjust column order
    cols = result_pev.columns.tolist()
    new_order = ['City', 'Province', 'Code'] + [c for c in cols if c not in ['City', 'Province', 'Code']]
    result_pev = result_pev[new_order]

    # Save PEV results
    if out_pev and out_pev != os.devnull:
        os.makedirs(os.path.dirname(out_pev), exist_ok=True)
        result_pev.to_excel(out_pev, index=False)
        print(f"Saved: {out_pev}")

    # Calculate CEV results
    result_cev = compute_city_year_eol_with_LE(
        df_sales=df_cev, sales_col='CEV_number',
        share_df=CEV_SHARE, batt_weibull=CEV_BATT_WEIBULL, is_pev=False,
        out_path=None, get_params_for_year=get_params_baseline_cev
    )

    # Add Province and Code to CEV results
    result_cev = result_cev.merge(cev_city_info, on='City', how='left')
    # Remove duplicate merge rows

    # Adjust column order
    cols = result_cev.columns.tolist()
    new_order = ['City', 'Province', 'Code'] + [c for c in cols if c not in ['City', 'Province', 'Code']]
    result_cev = result_cev[new_order]

    # Save CEV results
    if out_cev and out_cev != os.devnull:
        os.makedirs(os.path.dirname(out_cev), exist_ok=True)
        result_cev.to_excel(out_cev, index=False)
        print(f"Saved: {out_cev}")