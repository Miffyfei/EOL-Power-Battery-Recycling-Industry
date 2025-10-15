# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# =========================
# Paths (consistent with existing project)
# =========================
EOL_PATH = './input data/EOL LFP and NCM battery.xlsx'
PROP_PATH = './input data/Process type proportion_BS.xlsx'
LCA_PATH = './input data/LCA data.xlsx'
OUT_PATH = './output data/Environmental impact and metal recovery results under AR scenario.xlsx'

# =========================
# Process method columns (aligned with proportion table)
# =========================
METHOD_COLS = [
    'Medium emission_Pyro_LFP','High emission_Hyd_LFP',
    'High emission_Hyd_NCM','High emission_Pyro_NCM',
    'Low emission_Hyd_LFP','Low emission_Hyd_NCM',
    'Low emission_Pyro+Hyd_NCM','Medium emission_Hyd_NCM'
]

# Low emission method sets
LOW_METHODS = {
    'LFP': ['Low emission_Hyd_LFP'],
    'NCM': ['Low emission_Hyd_NCM', 'Low emission_Pyro+Hyd_NCM']
}

def get_applicable_methods(bt: str):
    if bt == 'LFP':
        return ['High emission_Hyd_LFP','Low emission_Hyd_LFP','Medium emission_Pyro_LFP']
    if bt == 'NCM':
        return ['High emission_Hyd_NCM','High emission_Pyro_NCM','Low emission_Hyd_NCM',
                'Low emission_Pyro+Hyd_NCM','Medium emission_Hyd_NCM']
    return []

# Recovery efficiency (consistent with method names)
def get_eff(method: str, bt: str):
    if ('NCM' in method and bt!='NCM') or ('LFP' in method and bt!='LFP'):
        return {}
    return {
        'High emission_Hyd_LFP':      {'nickel':0.0,'cobalt':0.0,'lithium':0.80,'manganese':0.0},
        'Low emission_Hyd_LFP':       {'nickel':0.0,'cobalt':0.0,'lithium':0.90,'manganese':0.0},
        'Medium emission_Pyro_LFP':   {'nickel':0.0,'cobalt':0.0,'lithium':0.85,'manganese':0.0},

        'High emission_Hyd_NCM':      {'nickel':0.90,'cobalt':0.90,'lithium':0.80,'manganese':0.90},
        'High emission_Pyro_NCM':     {'nickel':0.90,'cobalt':0.90,'lithium':0.80,'manganese':0.90},
        'Low emission_Hyd_NCM':       {'nickel':0.98,'cobalt':0.98,'lithium':0.90,'manganese':0.98},
        'Low emission_Pyro+Hyd_NCM':  {'nickel':0.985,'cobalt':0.985,'lithium':0.91,'manganese':0.985},
        'Medium emission_Hyd_NCM':    {'nickel':0.95,'cobalt':0.95,'lithium':0.85,'manganese':0.95},
    }.get(method, {})

# Metal content (kg/kWh)
METAL_CONTENT = {
    'LFP': {'lithium':0.106,'nickel':0.0,'cobalt':0.0,'manganese':0.0},
    'NCM': {'lithium':0.109879,'nickel':0.60,'cobalt':0.23475,'manganese':0.24},
}

# Environmental impact indicators
IMPACTS = [
    "Abiotic depletion","Abiotic depletion (fossil fuels)","Acidification",
    "Eutrophication","Fresh water aquatic ecotox.","Global warming (GWP100a)",
    "Human toxicity","Marine aquatic ecotoxicity","Ozone layer depletion (ODP)",
    "Photochemical oxidation","Terrestrial ecotoxicity"
]

# =========================
# Utility functions
# =========================
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+','', str(s).lower())

def detect_cols_and_factors(df: pd.DataFrame):
    # Weight: Weight (thousand t)
    if 'Weight' in df.columns:
        w_col, w2ton = 'Weight', 1000.0
    else:
        raise KeyError('EOL table missing weight column (Weight)')
    # Capacity: Capacity (GWh)
    if 'Capacity' in df.columns:
        c_col, c2kwh = 'Capacity', 1e6
    else:
        raise KeyError('EOL table missing capacity column (Capacity)')
    return w_col, w2ton, c_col, c2kwh

def build_lca_lookup(lca_df: pd.DataFrame):
    """
    Robust version compatible with case/aliases/duplicate columns:
    - Locate 'province' and 'impact/environment' columns by normalizing names (keep only alphanumeric and lowercase);
    - Use the located real column names directly without renaming to avoid duplicate column names;
    - If duplicate column names exist, perform horizontal merge first (prioritize non-empty values), then take the first column.
    """
    def _norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]+','', str(s).lower())

    # Normalize -> original column names (keep list if duplicates exist)
    buckets = {}
    for c in lca_df.columns:
        buckets.setdefault(_norm(c), []).append(c)

    # Locate key columns
    prov_cols = buckets.get('province', [])
    impact_cols = buckets.get('impact', []) or buckets.get('environment', [])

    if not prov_cols or not impact_cols:
        raise KeyError("LCA data must contain Province and Impact (or environment) columns (case/space insensitive).")

    # If duplicate columns, perform horizontal merge: prioritize non-empty values
    def coalesce_columns(df: pd.DataFrame, cols: list, new_name: str) -> pd.Series:
        if len(cols) == 1:
            s = df[cols[0]]
        else:
            s = df[cols].bfill(axis=1).ffill(axis=1).iloc[:, 0]
        return s.rename(new_name)

    prov_s   = coalesce_columns(lca_df, prov_cols,   'Province').astype(str).str.strip()
    impact_s = coalesce_columns(lca_df, impact_cols, 'Impact').astype(str).str.strip()

    # Create temporary view with only Province/Impact and method columns for easy iteration
    methods_in_lca = [m for m in METHOD_COLS if m in lca_df.columns]
    if not methods_in_lca:
        raise KeyError("No process method columns found in LCA data, please check if column names match METHOD_COLS.")

    lca_view = pd.concat([prov_s, impact_s, lca_df[methods_in_lca]], axis=1)

    # Build lookup table
    lookup = {}
    for _, row in lca_view.iterrows():
        prov = row['Province']
        impact_raw = row['Impact']

        # Match Impact name (strict + flexible)
        matched_impact = None
        for imp in IMPACTS:
            if imp.lower() == impact_raw.lower():
                matched_impact = imp; break
        if matched_impact is None:
            for imp in IMPACTS:
                if impact_raw.lower() in imp.lower() or imp.lower() in impact_raw.lower():
                    matched_impact = imp; break
        if matched_impact is None:
            continue  # Skip unrecognized rows

        for m in methods_in_lca:
            val = row[m]
            if pd.isna(val):
                continue
            bt = 'LFP' if 'LFP' in m else ('NCM' if 'NCM' in m else None)
            if bt is None:
                continue
            try:
                lookup[(prov, m, bt, matched_impact)] = float(val)
            except Exception:
                lookup[(prov, m, bt, matched_impact)] = 0.0

    print(f"[LCA] Number of entries recorded: {len(lookup)}")
    return lookup


def adjust_props_by_ar(row: pd.Series, bt: str, ratio: float, ncm_pyrohyd_share: float) -> dict:
    """
    Redistribute proportions according to AR logic (using "proportion transfer" to "take from medium/high emission and give to low emission"):
      - From the applicable method set for battery type bt, first separate into low and non-low categories;
      - Extract ratio (0.2/0.4/0.6) from the non-low total as transfer amount;
      - LFP: All added to Low emission_Hyd_LFP;
        NCM: Transfer amount * ncm_pyrohyd_share to Low emission_Pyro+Hyd_NCM,
             remainder to Low emission_Hyd_NCM.
      - Remaining non-low methods are proportionally reduced by their original shares.
    Note: This operates on "proportions", which will later be multiplied by each row's weight/capacity to get the corresponding amounts for each process.
    """
    applicable = get_applicable_methods(bt)
    # Original proportions (treat missing as 0), and normalize
    props = {m: float(row[m]) if (m in row and pd.notna(row[m])) else 0.0 for m in applicable}
    total = sum(props.values())
    if total <= 0:
        return {m: 0.0 for m in applicable}
    props = {m: v/total for m, v in props.items()}

    lows = [m for m in applicable if m in LOW_METHODS.get(bt, [])]
    non_lows = [m for m in applicable if m not in lows]

    non_low_sum = sum(props[m] for m in non_lows)
    if non_low_sum <= 0:
        return props  # No source to transfer from

    # Total amount to extract from non-low
    shift_total = min(ratio * non_low_sum, non_low_sum)

    # Proportionally reduce non-low
    for m in non_lows:
        reduce_m = shift_total * (props[m] / non_low_sum)
        props[m] = max(0.0, props[m] - reduce_m)

    # Add to low
    if bt == 'LFP':
        # Only one low emission method
        props['Low emission_Hyd_LFP'] = props.get('Low emission_Hyd_LFP', 0.0) + shift_total
    elif bt == 'NCM':
        # Two low emission methods, split by scenario share
        add_pyrohyd = shift_total * ncm_pyrohyd_share
        add_hyd     = shift_total - add_pyrohyd
        props['Low emission_Pyro+Hyd_NCM'] = props.get('Low emission_Pyro+Hyd_NCM', 0.0) + add_pyrohyd
        props['Low emission_Hyd_NCM']      = props.get('Low emission_Hyd_NCM', 0.0)      + add_hyd

    # Normalize again (for numerical stability)
    s = sum(props.values())
    if s > 0:
        props = {m: v/s for m, v in props.items()}
    return props

# =========================
# Read data and merge
# =========================
print("Reading EOL / Proportion / LCA ...")
eol = pd.read_excel(EOL_PATH)
prop = pd.read_excel(PROP_PATH)
lca = pd.read_excel(LCA_PATH)

# Clean data
for d in (eol, prop):
    for c in ['Year','Province','City','Scenario','Battery type']:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
eol['Year'] = eol['Year'].astype(int)
if 'Year' in prop.columns:
    prop['Year'] = prop['Year'].astype(int)

merge_cols = ['Year','Province','City']
if 'Code' in eol.columns and 'Code' in prop.columns:
    eol['Code'] = eol['Code'].astype(str).str.strip()
    prop['Code'] = prop['Code'].astype(str).str.strip()
    merge_cols.append('Code')

df = eol.merge(prop[merge_cols + METHOD_COLS], on=merge_cols, how='left')

missing = [m for m in METHOD_COLS if m not in df.columns]
if missing:
    raise KeyError(f'Proportion table missing method columns: {missing}')

W_COL, W_TO_TON, C_COL, C_TO_KWH = detect_cols_and_factors(df)
LCA_LOOKUP = build_lca_lookup(lca)

# =========================
# Calculation (AR scenario)
# =========================
# AR transfers 20/40/60% from non-low to low (starting from 2024)
AR_RATIOS = [0.20, 0.40, 0.60]                 # AR1 / AR2 / AR3
AR_NAMES  = ['AR1','AR2','AR3']
# Internal low emission for NCM: Share of Low emission_Pyro+Hyd_NCM (remainder to Low emission_Hyd_NCM)
AR_NCM_PYROHYD_SHARE = {
    'AR1': 0.05,   # AR1
    'AR2': 0.15,   # AR2
    'AR3': 0.20    # AR3
}

def run_ar_scenario(ratio: float, scen_name: str) -> pd.DataFrame:
    rows = []
    share_pyrohyd = AR_NCM_PYROHYD_SHARE[scen_name]

    for idx, r in df.iterrows():
        year = int(r['Year'])
        bt = r['Battery type']
        prov = r['Province']; city = r['City']
        scen = r['Scenario'] if 'Scenario' in r else ''

        # Weight/Capacity
        if pd.isna(r.get(W_COL)) or pd.isna(r.get(C_COL)):
            continue
        w_ton = float(r[W_COL]) * W_TO_TON
        e_kwh = float(r[C_COL]) * C_TO_KWH

        applicable = get_applicable_methods(bt)
        if not applicable:
            continue

        if year >= 2024:
            props = adjust_props_by_ar(r, bt, ratio, share_pyrohyd)
            effective_ratio = ratio
        else:
            # No adjustment before 2024: use original proportions and normalize
            props = {m: (float(r[m]) if pd.notna(r[m]) else 0.0) for m in applicable}
            s = sum(props.values())
            if s > 0:
                props = {m: v/s for m, v in props.items()}
            effective_ratio = 0.0

        # Aggregate impacts/metals
        impacts = {imp: 0.0 for imp in IMPACTS}
        metals  = {'lithium':0.0,'nickel':0.0,'cobalt':0.0,'manganese':0.0}

        for m, p in props.items():
            if p <= 0:
                continue
            w_m = w_ton * p
            e_m = e_kwh * p

            for imp in IMPACTS:
                unit_v = LCA_LOOKUP.get((prov, m, bt, imp), 0.0)
                impacts[imp] += w_m * unit_v

            eff = get_eff(m, bt)
            for metal, content in METAL_CONTENT.get(bt, {}).items():
                metals[metal] += e_m * content * eff.get(metal, 0.0)

        out = {
            'AR Scenario': scen_name,
            'AR Ratio': effective_ratio,
            'Year': year, 'Province': prov, 'City': city,
            'Scenario': scen, 'Battery type': bt
        }
        if 'Code' in r.index and pd.notna(r['Code']):
            out['Code'] = r['Code']
        out.update(impacts); out.update(metals)
        rows.append(out)
    return pd.DataFrame(rows)

print("Starting AR scenario calculation ...")
all_out = {}
for name, ratio in zip(AR_NAMES, AR_RATIOS):
    print(f"  -> {name} (transfer {int(ratio*100)}% from non-low to low; Pyro+Hyd share in NCM={AR_NCM_PYROHYD_SHARE[name]})")
    out = run_ar_scenario(ratio, name)
    all_out[name] = out
    print(f"     Rows: {len(out)}")

# =========================
# Export
# =========================
print("Writing to Excel ...")
with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as wr:
    for name, out in all_out.items():
        # Align column order
        meta = ['AR Scenario','Scenario','Year','Province','City']
        if 'Code' in out.columns: meta.append('Code')
        meta += ['Battery type','AR Ratio']
        cols = meta + IMPACTS + ['lithium','nickel','cobalt','manganese']
        out = out.reindex(columns=[c for c in cols if c in out.columns])
        sheet = f'{name} scenario'[:31]
        out.to_excel(wr, sheet_name=sheet, index=False)
print(f"Complete! Results saved to: {OUT_PATH}")