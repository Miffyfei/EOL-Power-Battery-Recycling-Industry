# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# =========================
# Paths (using your absolute paths)
# =========================
EOL_PATH = './input data/EOL LFP and NCM battery.xlsx'
PROP_PATH = './input data/Process type proportion_BS.xlsx'
LCA_PATH = './input data/LCA data.xlsx'
LCA_TO1 = './input data/30%TOSumLCA.xlsx'
LCA_TO2 = './input data/60%TOSumLCA.xlsx'
LCA_TO3 = './input data/100%TOSumLCA.xlsx'

OUT_PATH = r'./output data/Environmental impact and metal recovery results under TO scenario.xlsx'

# =========================
# Fixed 8 method columns (aligned with proportion table/LCA)
# =========================
METHOD_COLS = [
    'Medium emission_Pyro_LFP', 'High emission_Hyd_LFP',
    'High emission_Hyd_NCM', 'High emission_Pyro_NCM',
    'Low emission_Hyd_LFP', 'Low emission_Hyd_NCM',
    'Low emission_Pyro+Hyd_NCM', 'Medium emission_Hyd_NCM'
]


def get_applicable_methods(bt: str):
    if bt == 'LFP':
        return ['High emission_Hyd_LFP', 'Low emission_Hyd_LFP', 'Medium emission_Pyro_LFP']
    if bt == 'NCM':
        return ['High emission_Hyd_NCM', 'High emission_Pyro_NCM', 'Low emission_Hyd_NCM',
                'Low emission_Pyro+Hyd_NCM', 'Medium emission_Hyd_NCM']
    return []


# Recovery efficiency (consistent with method names) - Modified to set different ratios according to TO scenario
def get_eff(method: str, bt: str, to_scenario: str):
    if ('NCM' in method and bt != 'NCM') or ('LFP' in method and bt != 'LFP'):
        return {}

    # TO1 scenario recovery efficiency
    if to_scenario == 'TO1':
        return {
            'High emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.815, 'manganese': 0.0},
            'Low emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.903, 'manganese': 0.0},
            'Medium emission_Pyro_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.865, 'manganese': 0.0},

            'High emission_Hyd_NCM': {'nickel': 0.915, 'cobalt': 0.915, 'lithium': 0.815, 'manganese': 0.915},
            'High emission_Pyro_NCM': {'nickel': 0.915, 'cobalt': 0.915, 'lithium': 0.815, 'manganese': 0.915},
            'Low emission_Hyd_NCM': {'nickel': 0.983, 'cobalt': 0.983, 'lithium': 0.903, 'manganese': 0.983},
            'Low emission_Pyro+Hyd_NCM': {'nickel': 0.985, 'cobalt': 0.985, 'lithium': 0.91, 'manganese': 0.985},
            'Medium emission_Hyd_NCM': {'nickel': 0.95, 'cobalt': 0.95, 'lithium': 0.865, 'manganese': 0.95},
        }.get(method, {})

    # TO2 scenario recovery efficiency - increased recovery rate
    elif to_scenario == 'TO2':
        return {
            'High emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.83, 'manganese': 0.0},
            'Low emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.906, 'manganese': 0.0},
            'Medium emission_Pyro_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.88, 'manganese': 0.0},

            'High emission_Hyd_NCM': {'nickel': 0.925, 'cobalt': 0.925, 'lithium': 0.83, 'manganese': 0.925},
            'High emission_Pyro_NCM': {'nickel': 0.925, 'cobalt': 0.925, 'lithium': 0.83, 'manganese': 0.925},
            'Low emission_Hyd_NCM': {'nickel': 0.988, 'cobalt': 0.988, 'lithium': 0.906, 'manganese': 0.988},
            'Low emission_Pyro+Hyd_NCM': {'nickel': 0.985, 'cobalt': 0.985, 'lithium': 0.91, 'manganese': 0.985},
            'Medium emission_Hyd_NCM': {'nickel': 0.96, 'cobalt': 0.96, 'lithium': 0.88, 'manganese': 0.96},
        }.get(method, {})

    # TO3 scenario recovery efficiency - further increased recovery rate
    elif to_scenario == 'TO3':
        return {
            'High emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.85, 'manganese': 0.0},
            'Low emission_Hyd_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.91, 'manganese': 0.0},
            'Medium emission_Pyro_LFP': {'nickel': 0.0, 'cobalt': 0.0, 'lithium': 0.895, 'manganese': 0.0},

            'High emission_Hyd_NCM': {'nickel': 0.935, 'cobalt': 0.935, 'lithium': 0.85, 'manganese': 0.935},
            'High emission_Pyro_NCM': {'nickel': 0.935, 'cobalt': 0.935, 'lithium': 0.85, 'manganese': 0.935},
            'Low emission_Hyd_NCM': {'nickel': 0.993, 'cobalt': 0.993, 'lithium': 0.91, 'manganese': 0.993},
            'Low emission_Pyro+Hyd_NCM': {'nickel': 0.996, 'cobalt': 0.996, 'lithium': 0.91, 'manganese': 0.996},
            'Medium emission_Hyd_NCM': {'nickel': 0.97, 'cobalt': 0.97, 'lithium': 0.895, 'manganese': 0.97},
        }.get(method, {})

    # Default case (baseline scenario or other unknown scenarios)
    else:
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


# Metal content (kg/kWh)
METAL_CONTENT = {
    'LFP': {'lithium': 0.106, 'nickel': 0.0, 'cobalt': 0.0, 'manganese': 0.0},
    'NCM': {'lithium': 0.109879, 'nickel': 0.60, 'cobalt': 0.23475, 'manganese': 0.24},
}

# Fixed 11 environmental impacts (in your given order)
IMPACTS = [
    "Abiotic depletion", "Abiotic depletion (fossil fuels)", "Acidification",
    "Eutrophication", "Fresh water aquatic ecotox.", "Global warming (GWP100a)",
    "Human toxicity", "Marine aquatic ecotoxicity", "Ozone layer depletion (ODP)",
    "Photochemical oxidation", "Terrestrial ecotoxicity"
]

SCENARIOS = ['BS', 'TP', 'ED', 'LE']


# Determine weight conversion factor
def get_weight_conversion(eol_path):
    df_sample = pd.read_excel(eol_path, nrows=1)
    if 'Weight' in df_sample.columns:
        return 1000.0  # Convert thousand tons to tons
    else:
        return 1.0


WEIGHT_TO_TON = get_weight_conversion(EOL_PATH)
CAP_TO_KWH = 1e6  # GWh -> kWh

# -------------------------
# Chinese province to English province mapping
# -------------------------
PROVINCE_MAPPING = {
    '安徽': 'Anhui', '北京': 'Beijing', '重庆': 'Chongqing', '福建': 'Fujian',
    '甘肃': 'Gansu', '广东': 'Guangdong', '广西': 'Guangxi', '贵州': 'Guizhou',
    '海南': 'Hainan', '河北': 'Hebei', '黑龙江': 'Heilongjiang', '河南': 'Henan',
    '湖北': 'Hubei', '湖南': 'Hunan', '内蒙古': 'Inner Mongolia', '江苏': 'Jiangsu',
    '江西': 'Jiangxi', '吉林': 'Jilin', '辽宁': 'Liaoning', '宁夏': 'Ningxia',
    '青海': 'Qinghai', '陕西': 'Shaanxi', '山东': 'Shandong', '上海': 'Shanghai',
    '山西': 'Shanxi', '四川': 'Sichuan', '天津': 'Tianjin', '西藏': 'Tibet',
    '新疆': 'Xinjiang', '云南': 'Yunnan', '浙江': 'Zhejiang'
}

# -------------------------
# Read and merge data
# -------------------------
eol = pd.read_excel(EOL_PATH)
prop = pd.read_excel(PROP_PATH)

# Standardize dtype/whitespace
for c in ['Scenario', 'Province', 'City', 'Battery type']:
    if c in eol: eol[c] = eol[c].astype(str).str.strip()
eol['Year'] = eol['Year'].astype(int)
if 'Code' in eol.columns:
    eol['Code'] = eol['Code'].astype(str).str.strip()

for c in ['Province', 'City']:
    if c in prop: prop[c] = prop[c].astype(str).str.strip()
prop['Year'] = prop['Year'].astype(int)
if 'Code' in prop.columns:
    prop['Code'] = prop['Code'].astype(str).str.strip()

# Verify all proportion columns are present
missing = [c for c in METHOD_COLS if c not in prop.columns]
if missing:
    existing = [c for c in prop.columns if any(x in c for x in ['Hyd', 'Pyro', 'emission', 'Recovery', 'NCM', 'LFP'])]
    raise KeyError(f'Proportion table missing method columns: {missing}\nDetected relevant columns in proportion table: {existing}')

# Merge data
merge_cols = ['Year', 'Province', 'City']
if 'Code' in eol.columns and 'Code' in prop.columns:
    merge_cols.append('Code')

df = eol.merge(prop[merge_cols + METHOD_COLS], on=merge_cols, how='left')

# Check necessary columns
need = ['Weight', 'Capacity', 'Scenario', 'Battery type']
for c in need:
    if c not in df.columns:
        raise KeyError(f'EOL missing column: {c}')
if df[['Weight', 'Capacity']].isna().any().any():
    raise ValueError('EOL Weight/Capacity contains missing values.')


# -------------------------
# Modified LCA data reading function
# -------------------------
def read_lca_lookup(xlsx_path: str, is_to_scenario=False) -> dict:
    """Read LCA data and build lookup dictionary, adapted to your data format"""
    try:
        # Read Excel file
        lca_df = pd.read_excel(xlsx_path)

        if is_to_scenario:
            # TO scenario file processing logic
            print(f"\n=== Diagnostic Information (TO scenario) ===")
            print(f"Reading LCA file: {xlsx_path}")
            print(f"LCA file shape: {lca_df.shape}")
            print(f"LCA file column names: {list(lca_df.columns)}")

            lookup = {}

            # Correct column mapping (adjusted according to actual file structure)
            method_mapping = {
                'Medium emission_Pyro_LFP': 'Medium emission_Pyro_LFP',
                'High emission_Hyd_LFP': 'High emission_Hyd_LFP',
                'High emission_Hyd_NCM': 'High emission_Hyd_NCM',
                'High emission_Pyro_NCM': 'High emission_Pyro_NCM',
                'Low emission_Hyd_LFP': 'Low emission_Hyd_LFP',
                'Low emission_Hyd_NCM': 'Low emission_Hyd_NCM',
                'Low emission_Pyro+Hyd_NCM': 'Low emission_Pyro+Hyd_NCM',
                'Medium emission_Hyd_NCM': 'Medium emission_Hyd_NCM'
            }

            # Process each row of data
            for idx, row in lca_df.iterrows():
                try:
                    # Correctly read province and environmental impact from file
                    chinese_province = str(row.get('province', row.iloc[0] if len(row) > 0 else '')).strip()
                    impact = str(row.get('environment', row.iloc[1] if len(row) > 1 else '')).strip()

                    # Map province
                    if chinese_province in PROVINCE_MAPPING:
                        province = PROVINCE_MAPPING[chinese_province]
                    else:
                        province = chinese_province

                    # Match environmental impact type
                    matched_impact = None
                    for imp in IMPACTS:
                        if imp.lower() == impact.lower():
                            matched_impact = imp
                            break

                    if not matched_impact:
                        # Try fuzzy matching
                        for imp in IMPACTS:
                            if imp.lower() in impact.lower() or impact.lower() in imp.lower():
                                matched_impact = imp
                                break

                    # Since TO scenario files don't have battery type column, need to infer from method name
                    for method_display_name, method_key in method_mapping.items():
                        # Find corresponding value in data columns
                        column_candidates = [method_key, method_display_name]
                        value = None

                        for col_candidate in column_candidates:
                            if col_candidate in lca_df.columns:
                                value = row.get(col_candidate, np.nan)
                                break
                            # Also check column position index
                            col_idx = list(lca_df.columns).index(
                                col_candidate) if col_candidate in lca_df.columns else -1
                            if col_idx >= 0 and col_idx < len(row):
                                value = row.iloc[col_idx]
                                break

                        if pd.notna(value):
                            # Determine battery type
                            battery_type = 'NCM' if 'NCM' in method_key else 'LFP'

                            key = (province, method_key, battery_type, matched_impact)
                            lookup[key] = float(value)

                except Exception as row_error:
                    print(f"Error processing row {idx}: {row_error}")
                    continue

            print(f"Number of TO scenario LCA entries successfully read: {len(lookup)}")
            return lookup
        else:
            # TO scenario file processing logic
            print(f"\n=== Diagnostic Information (TO scenario) ===")
            print(f"Reading LCA file: {xlsx_path}")
            print(f"LCA file shape: {lca_df.shape}")
            print(f"LCA file column names: {list(lca_df.columns)}")

            lookup = {}

            # TO scenario file column mapping
            method_mapping = {
                'ME_Pyro_LFP': 'Medium emission_Pyro_LFP',
                'HE_Hyd_LFP': 'High emission_Hyd_LFP',
                'HE_Hyd_NCM': 'High emission_Hyd_NCM',
                'HE_Pyro_NCM': 'High emission_Pyro_NCM',
                'LE_Hyd_LFP': 'Low emission_Hyd_LFP',
                'LE_Hyd_NCM': 'Low emission_Hyd_NCM',
                'LE_PyroHyd_NCM': 'Low emission_Pyro+Hyd_NCM',
                'ME_Hyd_NCM': 'Medium emission_Hyd_NCM'
            }

            # Process each row of data
            for idx, row in lca_df.iterrows():
                try:
                    province_chinese = str(row.get('Province', ''))
                    battery_type = str(row.get('Battery Type', ''))

                    # Map province
                    if province_chinese in PROVINCE_MAPPING:
                        province = PROVINCE_MAPPING[province_chinese]
                    else:
                        province = province_chinese

                    # Process each method column
                    for col_name, method_name in method_mapping.items():
                        if col_name in lca_df.columns:
                            value = row.get(col_name, np.nan)
                            if pd.notna(value):
                                # Create key for each environmental impact
                                for impact in IMPACTS:
                                    key = (province, method_name, battery_type, impact)
                                    # TO scenario data doesn't have year dimension, store value directly
                                    lookup[key] = float(value)
                except Exception as row_error:
                    print(f"Error processing row {idx}: {row_error}")
                    continue

            print(f"Number of TO scenario LCA entries successfully read: {len(lookup)}")
            return lookup

    except Exception as e:
        print(f"Error reading LCA file {xlsx_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


# Read LCA data
print("Reading LCA data...")
L_BASE = read_lca_lookup(LCA_PATH)
L_TO1 = read_lca_lookup(LCA_TO1, is_to_scenario=True)
L_TO2 = read_lca_lookup(LCA_TO2, is_to_scenario=True)
L_TO3 = read_lca_lookup(LCA_TO3, is_to_scenario=True)

print(f"\nBaseline LCA data entry count: {len(L_BASE)}")
print(f"TO1 LCA data entry count: {len(L_TO1)}")
print(f"TO2 LCA data entry count: {len(L_TO2)}")
print(f"TO3 LCA data entry count: {len(L_TO3)}")


# -------------------------
# Key modification: Fix unit_impact function to ensure using correct LCA data
# -------------------------
def unit_impact(prov, method, bt, impact, to_scenario, to_lookup):
    """
    Get unit environmental impact
    to_scenario: TO scenario name ('TO1', 'TO2', 'TO3')
    to_lookup: Corresponding TO scenario LCA data
    """
    base_key = (prov, method, bt, impact)

    # Key modification: Select corresponding LCA data based on TO scenario
    if to_scenario in ['TO1', 'TO2', 'TO3']:
        # Use TO scenario LCA data
        if base_key in to_lookup:
            return to_lookup[base_key]
        else:
            # If not found in TO scenario data, use baseline data as fallback
            print(f"Warning: Data not found in TO scenario {to_scenario}: {base_key}, using baseline data")
            return L_BASE.get(base_key, 0.0)
    else:
        # For non-TO scenarios, use baseline LCA data
        return L_BASE.get(base_key, 0.0)


def run_es(es_name, es_lookup):
    rows = []

    for _, r in df.iterrows():
        if r['Scenario'] not in SCENARIOS:
            continue

        year = int(r['Year'])
        prov = r['Province']
        city = r['City']
        code = r['Code'] if 'Code' in r.index and pd.notna(r['Code']) else ''
        bt = r['Battery type']

        # Get weight and capacity
        weight = float(r['Weight'])
        cap = float(r['Capacity'])

        w_ton = weight * WEIGHT_TO_TON  # Convert to tons
        e_kwh = cap * CAP_TO_KWH  # Convert to kWh

        applicable = get_applicable_methods(bt)
        if not applicable:
            print(f'Warning: {bt} battery type has no applicable recycling methods')
            continue

        # Check proportion data completeness
        missing_props = [m for m in applicable if m not in r.index or pd.isna(r[m])]
        if missing_props:
            print(f'Warning: {prov}-{city}-{year}-{bt} missing proportion data: {missing_props}')
            continue

        total_prop = sum(float(r[m]) for m in applicable if pd.notna(r[m]))
        if not np.isclose(total_prop, 1.0, atol=1e-6):
            print(f'Warning: Proportion sum not equal to 1: {prov}-{city}-{year}-{bt}: {total_prop:.6f}')

        # Initialize results
        impacts = {imp: 0.0 for imp in IMPACTS}
        metals = {'lithium': 0.0, 'nickel': 0.0, 'cobalt': 0.0, 'manganese': 0.0}

        # Calculate environmental impact and metal recovery for each method
        for m in applicable:
            p = float(r[m]) if pd.notna(r[m]) else 0.0
            if p <= 0:
                continue

            w_m = w_ton * p  # Weight processed by this method (tons)
            e_m = e_kwh * p  # Capacity processed by this method (kWh)

            # Environmental impact (calculated by weight) - Key modification: pass TO scenario name and corresponding LCA data
            for imp in IMPACTS:
                u = unit_impact(prov, m, bt, imp, es_name, es_lookup)
                impacts[imp] += w_m * u  # Total environmental impact = weight × unit environmental impact

            # Metal recovery (calculated by capacity) - Use corresponding TO scenario recovery efficiency
            eff = get_eff(m, bt, es_name)
            for metal, content in METAL_CONTENT.get(bt, {}).items():
                metals[metal] += e_m * content * eff.get(metal, 0.0)  # Metal recovery amount = capacity × content × efficiency

        # Build result row
        result_row = {
            'TO Scenario': es_name,  # Modified column name to TO Scenario
            'Scenario': r['Scenario'],
            'Year': year,
            'Province': prov,
            'City': city,
            'Battery type': bt
        }

        # Add Code (if exists)
        if 'Code' in r.index and pd.notna(r['Code']):
            result_row['Code'] = r['Code']

        # Add environmental impact and metal recovery results
        result_row.update(impacts)
        result_row.update(metals)

        rows.append(result_row)

    return pd.DataFrame(rows)


# Execute calculation and output
to_scenarios = [('TO1', L_TO1), ('TO2', L_TO2), ('TO3', L_TO3)]

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as wr:
    for to_name, to_lookup in to_scenarios:
        print(f'\nCalculating {to_name} scenario...')
        print(f"Using LCA data entry count: {len(to_lookup)}")
        out = run_es(to_name, to_lookup)

        # Define output column order
        meta_cols = ['TO Scenario', 'Scenario', 'Year', 'Province', 'City']
        if 'Code' in out.columns:
            meta_cols.append('Code')
        meta_cols.append('Battery type')

        cols = meta_cols + IMPACTS + ['lithium', 'nickel', 'cobalt', 'manganese']

        # Ensure all columns exist
        existing_cols = [col for col in cols if col in out.columns]
        out = out.reindex(columns=existing_cols)

        sheet_name = f'{to_name} scenario'
        # Truncate worksheet name (Excel limit 31 characters)
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        out.to_excel(wr, sheet_name=sheet_name, index=False)
        print(f'{to_name} scenario calculation completed, total {len(out)} rows of data')

print('\nAll calculations completed! Results saved to:', OUT_PATH)