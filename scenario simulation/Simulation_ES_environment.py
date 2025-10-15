# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# =========================
# Paths (using your absolute paths)
# =========================
EOL_PATH = './input data/EOL LFP and NCM battery.xlsx'
PROP_PATH = './input data/Process type proportion_BS.xlsx'
LCA_PATH = './input data/LCA data.xlsx'
LCA_ES1 = './input data/LCA data about ES1.xlsx'
LCA_ES2 = './input data/LCA data about ES2.xlsx'
LCA_ES3 = './input data/LCA data about ES3.xlsx'

OUT_PATH = r'./output data/Environmental impact and metal recovery results under ES scenario.xlsx'

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


# Recovery efficiency (consistent with method names)
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
# Modified LCA data reading function with error fixes and province mapping
# -------------------------
def read_lca_lookup(xlsx_path: str) -> dict:
    """Read LCA data and build lookup dictionary, adapted to your data format"""
    try:
        # Read Excel file
        lca_df = pd.read_excel(xlsx_path)

        lca_df = pd.read_excel(xlsx_path)

        print(f"\n=== Diagnostic Information ===")
        print(f"Reading LCA file: {xlsx_path}")
        print(f"LCA file shape: {lca_df.shape}")
        print(f"LCA file column names: {list(lca_df.columns)}")
        print(f"LCA first 5 rows:")
        print(lca_df.head())
        print(f"=== End of Diagnostics ===")

        lookup = {}
        # Add column type check in read_lca_lookup function
        print(f"LCA column data types:")
        for col in lca_df.columns:
            print(f"  {col}: {lca_df[col].dtype}")

        # Based on your provided headers, first column is CN_province, second column is environment
        # Rename columns for processing
        col_rename = {}
        for col in lca_df.columns:
            if 'CN_province' in str(col):
                col_rename[col] = 'Province'
            elif 'environment' in str(col).lower():
                col_rename[col] = 'Impact'

        if col_rename:
            lca_df = lca_df.rename(columns=col_rename)

        # Ensure necessary columns exist
        if 'Province' not in lca_df.columns:
            # Use first column as province column
            lca_df = lca_df.rename(columns={lca_df.columns[0]: 'Province'})

        if 'Impact' not in lca_df.columns:
            # Use second column as environmental impact column
            lca_df = lca_df.rename(columns={lca_df.columns[1]: 'Impact'})

        # Process data
        provinces_found = set()
        impacts_found = set()

        for idx, row in lca_df.iterrows():
            try:
                # Skip empty rows - safer handling
                province_val = row.get('Province', np.nan)
                impact_val = row.get('Impact', np.nan)

                # Use safer method to check for missing values
                if pd.isna(province_val) or pd.isna(impact_val):
                    continue

                chinese_province = str(province_val).strip()
                impact = str(impact_val).strip()

                # Map Chinese province to English province
                if chinese_province in PROVINCE_MAPPING:
                    province = PROVINCE_MAPPING[chinese_province]
                else:
                    # If no mapping found, try to use directly (might already be English)
                    province = chinese_province

                provinces_found.add(province)
                impacts_found.add(impact)

                # Check if impact is in predefined list
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

                if not matched_impact:
                    continue  # If no matching impact, skip this row

                # Process each method
                for method in METHOD_COLS:
                    # Get method value more safely
                    method_val = row.get(method, np.nan)
                    if pd.notna(method_val):
                        # Determine battery type
                        if 'NCM' in method:
                            battery_type = 'NCM'
                        elif 'LFP' in method:
                            battery_type = 'LFP'
                        else:
                            continue

                        key = (province, method, battery_type, matched_impact)
                        try:
                            lookup[key] = float(method_val)
                        except (ValueError, TypeError):
                            print(f"Warning: Unable to convert value: {province}, {method}, {matched_impact}, value: {method_val}")
                            lookup[key] = 0.0
            except Exception as row_error:
                print(f"Error processing row {idx}: {row_error}")
                continue

        print(f"Provinces found in LCA file: {provinces_found}")
        print(f"Environmental impacts found in LCA file: {impacts_found}")
        print(f"Number of LCA entries successfully read: {len(lookup)}")

        return lookup

    except Exception as e:
        print(f"Error reading LCA file {xlsx_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


# Read LCA data
print("Reading LCA data...")
L_BASE = read_lca_lookup(LCA_PATH)
L_ES1 = read_lca_lookup(LCA_ES1)
L_ES2 = read_lca_lookup(LCA_ES2)
L_ES3 = read_lca_lookup(LCA_ES3)

print(f"\nBaseline LCA data entry count: {len(L_BASE)}")
print(f"ES1 LCA data entry count: {len(L_ES1)}")
print(f"ES2 LCA data entry count: {len(L_ES2)}")
print(f"ES3 LCA data entry count: {len(L_ES3)}")

# Check data completeness - more detailed testing
print("\n=== Data Matching Test ===")
test_provinces = list(df['Province'].unique())[:3]  # Test first 3 provinces
test_methods = METHOD_COLS[:2]  # Test first 2 methods
test_impacts = IMPACTS[:2]  # Test first 2 environmental impacts

for prov in test_provinces:
    for method in test_methods:
        # Determine battery type
        if 'NCM' in method:
            bt = 'NCM'
        elif 'LFP' in method:
            bt = 'LFP'
        else:
            continue

        for imp in test_impacts:
            key = (prov, method, bt, imp)
            if key in L_BASE:
                print(f"Match found: {key} -> {L_BASE[key]}")
            else:
                print(f"No match found: {key}")

# Check data completeness
print("\n=== Environmental Impact Matching Check ===")
for imp in IMPACTS:
    found = False
    for method in METHOD_COLS:
        if 'NCM' in method:
            bt = 'NCM'
        elif 'LFP' in method:
            bt = 'LFP'
        else:
            continue

        # Check if data exists for any province
        for prov in df['Province'].unique():
            if (prov, method, bt, imp) in L_BASE:
                found = True
                print(f"Found data for {imp}: {prov}, {method}")
                break
        if found:
            break

    if not found:
        print(f'Warning: Indicator "{imp}" in LCA baseline may not have matching values')


def unit_impact(prov, method, bt, impact, year, es_lookup):
    base_key = (prov, method, bt, impact)
    es_key = (prov, method, bt, impact)

    base_v = L_BASE.get(base_key, 0.0)
    es_v = es_lookup.get(es_key, base_v)  # If ES data is missing, use baseline value

    if year <= 2023:
        return base_v
    else:  # year >= 2024
        return es_v


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

            # Environmental impact (calculated by weight)
            for imp in IMPACTS:
                u = unit_impact(prov, m, bt, imp, year, es_lookup)
                impacts[imp] += w_m * u  # Total environmental impact = weight × unit environmental impact

            # Metal recovery (calculated by capacity)
            eff = get_eff(m, bt)
            for metal, content in METAL_CONTENT.get(bt, {}).items():
                metals[metal] += e_m * content * eff.get(metal, 0.0)  # Metal recovery amount = capacity × content × efficiency

        # Build result row
        result_row = {
            'ES': es_name,
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
es_list = [('ES1', L_ES1), ('ES2', L_ES2), ('ES3', L_ES3)]

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as wr:
    for es_name, look in es_list:
        print(f'\nCalculating {es_name} scenario...')
        out = run_es(es_name, look)

        # Define output column order
        meta_cols = ['ES', 'Scenario', 'Year', 'Province', 'City']
        if 'Code' in out.columns:
            meta_cols.append('Code')
        meta_cols.append('Battery type')

        cols = meta_cols + IMPACTS + ['lithium', 'nickel', 'cobalt', 'manganese']

        # Ensure all columns exist
        existing_cols = [col for col in cols if col in out.columns]
        out = out.reindex(columns=existing_cols)

        sheet_name = f'{es_name} scenario'
        # Truncate worksheet name (Excel limit 31 characters)
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        out.to_excel(wr, sheet_name=sheet_name, index=False)
        print(f'{es_name} scenario calculation completed, total {len(out)} rows of data')

print('\nAll calculations completed! Results saved to:', OUT_PATH)