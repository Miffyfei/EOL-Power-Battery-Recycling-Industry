# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# =========================
# File paths
# =========================
EOL_PATH = './input data/EOL LFP and NCM battery.xlsx'
PROP_PATH = './input data/Process type proportion_BS.xlsx'
LCA_PATH = './input data/LCA data with SU.xlsx'
OUT_PATH = './output data/Environmental impact and metal recovery results under SU scenario.xlsx'

# =========================
# Recycling process method columns
# =========================
METHOD_COLS = [
    'Medium emission_Pyro_LFP', 'High emission_Hyd_LFP',
    'High emission_Hyd_NCM', 'High emission_Pyro_NCM',
    'Low emission_Hyd_LFP', 'Low emission_Hyd_NCM',
    'Low emission_Pyro+Hyd_NCM', 'Medium emission_Hyd_NCM'
]

# Secondary use method name
SU_METHOD = 'Secondary Use LFP'
SU_METHOD_ALIASES = [
    'Secondary Use LFP', 'Secondary_use_LFP', 'Second_use_LFP',
    'Seconde_use_LFP', 'second_use_lfp', 'secondaryuselfp', 'secondeuselfp'
]


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def get_applicable_methods(bt: str):
    if bt == 'LFP':
        return ['High emission_Hyd_LFP', 'Low emission_Hyd_LFP', 'Medium emission_Pyro_LFP']
    if bt == 'NCM':
        return ['High emission_Hyd_NCM', 'High emission_Pyro_NCM', 'Low emission_Hyd_NCM',
                'Low emission_Pyro+Hyd_NCM', 'Medium emission_Hyd_NCM']
    return []


def get_eff(method: str, bt: str):
    if ('NCM' in method and bt != 'NCM') or ('LFP' in method and bt != 'LFP'):
        return {}
    return {
        'High emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.80, 'manganese': 0.0},
        'Low emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.90, 'manganese': 0.0},
        'Medium emission_Pyro_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.85, 'manganese': 0.0},

        'High emission_Hyd_NCM': {'nickel': 0.90, 'cobalt': 0.90, 'lithium': 0.80, 'manganese': 0.90},
        'High emission_Pyro_NCM': {'nickel': 0.90, 'cobalt': 0.90, 'lithium': 0.80, 'manganese': 0.90},
        'Low emission_Hyd_NCM': {'nickel': 0.98, 'cobalt': 0.98, 'lithium': 0.90, 'manganese': 0.98},
        'Low emission_Pyro+Hyd_NCM': {'nickel': 0.985, 'cobalt': 0.985, 'lithium': 0.91, 'manganese': 0.985},
        'Medium emission_Hyd_NCM': {'nickel': 0.95, 'cobalt': 0.95, 'lithium': 0.85, 'manganese': 0.95},
    }.get(method, {})


METAL_CONTENT = {
    'LFP': {'lithium': 0.106, 'nickel': 0.0, 'cobalt': 0.0, 'manganese': 0.0},
    'NCM': {'lithium': 0.109879, 'nickel': 0.60, 'cobalt': 0.23475, 'manganese': 0.24},
}

IMPACTS = [
    "Abiotic depletion", "Abiotic depletion (fossil fuels)", "Acidification",
    "Eutrophication", "Fresh water aquatic ecotox.", "Global warming (GWP100a)",
    "Human toxicity", "Marine aquatic ecotoxicity", "Ozone layer depletion (ODP)",
    "Photochemical oxidation", "Terrestrial ecotoxicity"
]

SECOND_USE_RATIOS = [0.2, 0.4, 0.6]  # SU1, SU2, SU3


def detect_cols_and_factors(df: pd.DataFrame):
    if 'Weight' in df.columns:
        weight_col, weight_to_ton = 'Weight', 1000.0
    else:
        raise KeyError('EOL table missing weight column (Weight)')

    if 'Capacity' in df.columns:
        cap_col, cap_to_kwh = 'Capacity', 1e6
    else:
        raise KeyError('EOL table missing capacity column (Capacity (GWh))')

    return weight_col, weight_to_ton, cap_col, cap_to_kwh


# =========================
# Data reading and preprocessing
# =========================
print("Reading data...")

eol_df = pd.read_excel(EOL_PATH)
prop_df = pd.read_excel(PROP_PATH)
lca_df = pd.read_excel(LCA_PATH)

for df in [eol_df, prop_df]:
    for col in ['Year', 'Province', 'City', 'Scenario', 'Battery type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

eol_df['Year'] = eol_df['Year'].astype(int)
if 'Year' in prop_df.columns:
    prop_df['Year'] = prop_df['Year'].astype(int)

merge_cols = ['Year', 'Province', 'City']
if 'Code' in eol_df.columns and 'Code' in prop_df.columns:
    for df in (eol_df, prop_df):
        df['Code'] = df['Code'].astype(str).str.strip()
    merge_cols.append('Code')

df = eol_df.merge(prop_df[merge_cols + METHOD_COLS], on=merge_cols, how='left')

missing_methods = [m for m in METHOD_COLS if m not in df.columns]
if missing_methods:
    raise KeyError(f'Proportion table missing method columns: {missing_methods}')

W_COL, W_TO_TON, C_COL, C_TO_KWH = detect_cols_and_factors(df)


def build_lca_lookup(lca_df: pd.DataFrame):
    lca_lookup = {}
    cols_lower = {_norm(c): c for c in lca_df.columns}
    rename_map = {}

    if 'impact' not in cols_lower and 'environment' in cols_lower:
        rename_map[cols_lower['environment']] = 'Impact'
    elif 'impact' in cols_lower:
        rename_map[cols_lower['impact']] = 'Impact'

    if 'Province' in cols_lower:
        rename_map[cols_lower['Province']] = 'Province'

    su_col_found = None
    for c in lca_df.columns:
        if _norm(c) in {_norm(x) for x in SU_METHOD_ALIASES}:
            rename_map[c] = SU_METHOD
            su_col_found = c
    if su_col_found is None:
        pass

    if rename_map:
        lca_df = lca_df.rename(columns=rename_map)

    if 'Province' not in lca_df.columns or 'Impact' not in lca_df.columns:
        raise KeyError("LCA data must contain Province and Impact (or environment) columns")

    lca_df['Province'] = lca_df['Province'].astype(str).str.strip()
    lca_df['Impact'] = lca_df['Impact'].astype(str).str.strip()

    all_methods = [m for m in METHOD_COLS if m in lca_df.columns]
    if SU_METHOD in lca_df.columns:
        all_methods.append(SU_METHOD)

    for _, row in lca_df.iterrows():
        prov = row['Province']
        impact_raw = str(row['Impact']).strip()

        matched_impact = None
        for imp in IMPACTS:
            if imp.lower() == impact_raw.lower():
                matched_impact = imp
                break
        if matched_impact is None:
            for imp in IMPACTS:
                if impact_raw.lower() in imp.lower() or imp.lower() in impact_raw.lower():
                    matched_impact = imp
                    break
        if matched_impact is None:
            continue

        for method in all_methods:
            if method not in row or pd.isna(row[method]):
                continue

            if 'LFP' in method or method == SU_METHOD:
                bt = 'LFP'
            elif 'NCM' in method:
                bt = 'NCM'
            else:
                continue

            try:
                lca_lookup[(prov, method, bt, matched_impact)] = float(row[method])
            except Exception:
                lca_lookup[(prov, method, bt, matched_impact)] = 0.0

    print(f"[LCA] SU column unified as: {SU_METHOD}" if SU_METHOD in lca_df.columns else "[LCA] No SU column detected")
    print(f"[LCA] Number of entries recorded: {len(lca_lookup)}")
    return lca_lookup


lca_lookup = build_lca_lookup(lca_df)
print(f"LCA data entry count: {len(lca_lookup)}")


# =========================
# Corrected calculation function
# =========================
def calculate_su_scenario(second_use_ratio: float, scenario_name: str) -> pd.DataFrame:
    results = []

    for idx, row in df.iterrows():
        year = int(row['Year'])
        prov = row['Province']
        city = row['City']
        scen = row['Scenario'] if 'Scenario' in row else ''
        bt = row['Battery type']

        if pd.isna(row.get(W_COL)) or pd.isna(row.get(C_COL)):
            continue
        weight_t = float(row[W_COL]) * W_TO_TON
        cap_kwh = float(row[C_COL]) * C_TO_KWH

        applicable = get_applicable_methods(bt)
        if not applicable:
            continue

        props = {m: float(row[m]) if pd.notna(row[m]) else 0.0 for m in applicable}
        prop_sum = sum(props.values())
        if prop_sum <= 0:
            continue
        norm_props = {m: (v / prop_sum) for m, v in props.items()}

        impacts = {imp: 0.0 for imp in IMPACTS}
        metals = {'lithium': 0.0, 'nickel': 0.0, 'cobalt': 0.0, 'manganese': 0.0}

        # ========== Secondary use (only LFP & year>=2024) ==========
        su_ratio = second_use_ratio if (bt == 'LFP' and year >= 2024) else 0.0

        # SU portion (environmental impact + 100% metal recovery)
        if su_ratio > 0:
            su_w = weight_t * su_ratio
            su_e = cap_kwh * su_ratio  # Capacity for secondary use

            # Environmental impact
            for imp in IMPACTS:
                key = (prov, SU_METHOD, 'LFP', imp)
                unit_v = lca_lookup.get(key, 1.0)
                impacts[imp] += su_w * unit_v

            # Metal recovery: secondary use equals 100% recovery
            for metal, content in METAL_CONTENT.get(bt, {}).items():
                metals[metal] += su_e * content * 1.0  # 100% recovery efficiency

        # Remaining portion goes to conventional recycling allocation
        rem_w = weight_t * (1.0 - su_ratio)
        rem_e = cap_kwh * (1.0 - su_ratio)

        for m in applicable:
            p = norm_props[m]
            if p <= 0:
                continue
            w_m = rem_w * p
            e_m = rem_e * p

            # Environmental impact
            for imp in IMPACTS:
                key = (prov, m, bt, imp)
                unit_v = lca_lookup.get(key, 0.0)
                impacts[imp] += w_m * unit_v

            # Metal recovery
            eff = get_eff(m, bt)
            for metal, content in METAL_CONTENT.get(bt, {}).items():
                metals[metal] += e_m * content * eff.get(metal, 0.0)

        out_row = {
            'SU Scenario': scenario_name,
            'Year': year,
            'Province': prov,
            'City': city,
            'Scenario': scen,
            'Battery type': bt,
            'SU Ratio': su_ratio
        }
        if 'Code' in row.index and pd.notna(row['Code']):
            out_row['Code'] = row['Code']

        out_row.update(impacts)
        out_row.update(metals)
        results.append(out_row)

    return pd.DataFrame(results)


# =========================
# Execute calculation & export
# =========================
print("Starting secondary use scenario calculation...")

all_results = {}
scenario_names = ['SU1', 'SU2', 'SU3']

for i, ratio in enumerate(SECOND_USE_RATIOS):
    scen_name = scenario_names[i]
    print(f"Calculating scenario: {scen_name} (Secondary use ratio: {ratio})")
    result_df = calculate_su_scenario(ratio, scen_name)
    all_results[scen_name] = result_df
    print(f"Scenario {scen_name} completed, data rows: {len(result_df)}")

print("Outputting results to Excel...")
with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    for scen_name, result_df in all_results.items():
        meta_cols = ['SU Scenario', 'Scenario', 'Year', 'Province', 'City']
        if 'Code' in result_df.columns:
            meta_cols.append('Code')
        meta_cols.append('Battery type')
        meta_cols.append('SU Ratio')

        cols = meta_cols + IMPACTS + ['lithium', 'nickel', 'cobalt', 'manganese']
        existing = [c for c in cols if c in result_df.columns]
        out_df = result_df.reindex(columns=existing)

        sheet_name = f"{scen_name} scenario"[:31]
        out_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nCalculation completed! Results saved to: {OUT_PATH}")