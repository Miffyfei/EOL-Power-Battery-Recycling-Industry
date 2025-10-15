# -*- coding: utf-8 -*-
"""
Cross-regional transportation capacity-EOL battery matching (annual independent; four scenarios; including informal local recycling; formal by city)
Scenarios:
  - baseline_inprov: Both formal and informal within province, formal only goes to cities with capacity in the same province
  - radius300 / citycluster / neighbor_prov: Formal can cross provinces to any city with capacity in target province; informal still only within province
Rules and implementation points:
  - In baseline scenario, first calibrate K for annual EOL based on target "national formal share" (annual shares: 2020:0.25, 2021:0.30, 2022:0.35, 2023:0.40, 2024:0.45)
    Get annual K(year). Other three scenarios use the same K for each year.
  - Formal allocation follows "city → (candidate province) → candidate province's capacity cities" order matching: actually lands on "target cities" (cities with capacity).
  - Distance:
      * Formal: Use city→target_city city distance; if same city, use 10 km.
      * Informal: Only within province, aggregate "city remaining amount = EOL - formal collected amount" within province; t·km = intra-province average inter-city distance * informal total amount;
                Intra-province average inter-city distance = mean of distances between city pairs in the province (excluding inf and 0); if only one city in province or all unavailable, use 10 km.
  - Informal total amount = EOL - formal total amount (by city, province, year), ensuring unrecycled_tons is always 0.
  - Informal split by enterprise type: use (medium effective capacity : micro effective capacity) weights for allocation; if both are 0, then 50/50.
  - Output consistent/enhanced with previous, ensuring each province has rows every year (aligned with cities.csv province complete set).
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Set, List, Tuple

# ========================== Configuration ==========================
CITIES_CSV            = "./input data/cities.csv"             # Complete city list (city_code, city_name, province)
CITY_CLUSTERS_CSV     = "./input data/city_info.csv"          # City cluster mapping (city_code, cluster_id)
EOL_CITY_YEAR_CSV     = "./input data/eol_city_year.csv"      # (city_code, year, eol_tons)
DIST_CC_WIDE_CSV      = "./input data/dist_city_city.csv"     # Wide table: index & columns = city_code
CAPA_CITY_YEAR_CSV    = "./input data/formal_capacity_by_province.csv"  # Actually city capacity: province,year,city,city_code,capacity_tons
PROV_ADJ_CSV          = "./input data/province_adjacency.csv" # Province adjacency
INFORMAL_COUNTS_CSV   = "./input data/informal_counts_by_province.csv"  # (year, province, Abbreviation, micro_cnt, medium_cnt)

# Scenario list
SCENARIOS = ["baseline_inprov", "radius300", "citycluster", "neighbor_prov"]

# Baseline scenario "national formal enterprise collection share" target (annual)
FORMAL_TARGET_SHARE_YEAR = {
    2020: 0.25,
    2021: 0.30,
    2022: 0.35,
    2023: 0.40,
    2024: 0.45,
}
# Default for other years if exists
DEFAULT_FORMAL_SHARE = 0.45

# Informal enterprise capacity (only for type allocation weights)
MEDIUM_CAP_PER_FIRM   = 5000.0  # t/year/firm
MICRO_CAP_PER_FIRM    = 2500.0  # t/year/firm
INFORMAL_UTIL_RATE    = 0.65    # For weights (not for upper limit)

# Transportation distance parameters
RADIUS_KM = 300.0
LOCAL_CITY_DISTANCE_KM = 10.0

# Results directory
OUT_DIR = "./output data"
os.makedirs(OUT_DIR, exist_ok=True)

# ========================== Utilities ==========================
def norm_str(s):
    return "" if pd.isna(s) else str(s).strip()

def norm_lower(s):
    return norm_str(s).lower()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [norm_str(c).lower() for c in df.columns]
    return df

def ensure_province_col(df: pd.DataFrame) -> pd.DataFrame:
    if "province" in df.columns:
        df["province"] = df["province"].map(norm_lower)
        return df
    cands = [c for c in df.columns if "province" in c]
    df["province"] = df[cands[0]].map(norm_lower) if cands else ""
    return df
def coalesce_cols(df: pd.DataFrame, cols: list, default_val: str = "") -> pd.Series:
    """
    Sequentially take the first "non-empty string" value from cols and return as a Series.
    If none exist or all are empty, return default_val.
    """
    out = None
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).fillna("").str.strip()
            out = s if out is None else out.mask(out != "", out).fillna("").where(out != "", s)
    if out is None:
        out = pd.Series([default_val]*len(df))
    out = out.fillna("").astype(str)
    return out

# ========================== Read base tables ==========================
cities_df     = pd.read_csv(CITIES_CSV, dtype=str).fillna("")
city_clusters = pd.read_csv(CITY_CLUSTERS_CSV, dtype=str).fillna("")
eol_city_year = pd.read_csv(EOL_CITY_YEAR_CSV, dtype=str).fillna("")
dist_raw      = pd.read_csv(DIST_CC_WIDE_CSV, index_col=0, dtype=str)
cap_city_year = pd.read_csv(CAPA_CITY_YEAR_CSV, dtype=str).fillna("")
prov_adj      = pd.read_csv(PROV_ADJ_CSV, dtype=str).fillna("")
informal_cnt  = pd.read_csv(INFORMAL_COUNTS_CSV, dtype=str).fillna("")

cities_df     = normalize_cols(cities_df)
city_clusters = normalize_cols(city_clusters)
eol_city_year = normalize_cols(eol_city_year)
cap_city_year = normalize_cols(cap_city_year)
prov_adj      = normalize_cols(prov_adj)
informal_cnt  = normalize_cols(informal_cnt)

# Validate cities.csv
req_city_cols = {"city_code", "city_name", "province"}
miss = req_city_cols - set(cities_df.columns)
if miss:
    raise ValueError(f"cities.csv missing required columns: {miss}")
cities_df["city_code"] = cities_df["city_code"].map(norm_str)
cities_df["city_name"] = cities_df["city_name"].map(norm_str)
cities_df = ensure_province_col(cities_df)

# Merge city clusters
if "cluster_id" not in city_clusters.columns:
    city_clusters["cluster_id"] = ""
city_clusters["city_code"] = city_clusters["city_code"].map(norm_str)
city_clusters = city_clusters[["city_code", "cluster_id"]].drop_duplicates()
cities_df = cities_df.merge(city_clusters, on="city_code", how="left")
cities_df["cluster_id"] = cities_df["cluster_id"].fillna("").map(norm_str)

# eol_city_year
need_eol = {"city_code", "year", "eol_tons"}
if not need_eol.issubset(eol_city_year.columns):
    raise ValueError(f"eol_city_year.csv must contain columns: {need_eol}")
eol_city_year["city_code"] = eol_city_year["city_code"].map(norm_str)
eol_city_year["year"] = eol_city_year["year"].astype(int)
eol_city_year["eol_tons"] = pd.to_numeric(eol_city_year["eol_tons"], errors="coerce").fillna(0.0)

# Formal capacity (by city)
need_cap = {"province", "year", "city", "city_code", "capacity_tons"}
if not need_cap.issubset(cap_city_year.columns):
    raise ValueError(f"formal_capacity_by_province.csv must contain columns: {need_cap}")
cap_city_year = ensure_province_col(cap_city_year)
cap_city_year["year"] = cap_city_year["year"].astype(int)
cap_city_year["capacity_tons"] = pd.to_numeric(cap_city_year["capacity_tons"], errors="coerce").fillna(0.0)
cap_city_year["city_code"] = cap_city_year["city_code"].map(norm_str)

# Province adjacency
prov_adj = ensure_province_col(prov_adj)
neighbor_col = "neighbor_province" if "neighbor_province" in prov_adj.columns else ""
if not neighbor_col:
    for c in prov_adj.columns:
        if "neighbor" in c and "number" not in c:
            neighbor_col = c
            break
if neighbor_col:
    tmp = []
    for r in prov_adj.itertuples(index=False):
        p = norm_lower(getattr(r, "province"))
        raw = norm_str(getattr(r, neighbor_col))
        raw = raw.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        for sep in [",", "；", ";", "|", "、", "，"]:
            raw = raw.replace(sep, " ")
        nbs = [norm_lower(x) for x in raw.split() if x]
        for nb in nbs:
            if nb and nb != p:
                tmp.append((p, nb))
    prov_adj_clean = pd.DataFrame(tmp, columns=["province", "neighbor_province"]).drop_duplicates()
else:
    prov_adj_clean = pd.DataFrame(columns=["province", "neighbor_province"])

# Informal enterprise counts (for type allocation weights)
informal_cnt = ensure_province_col(informal_cnt)
if "year" not in informal_cnt.columns:
    raise ValueError("informal_counts_by_province.csv must contain year column")
informal_cnt["year"] = informal_cnt["year"].astype(int)
for c in ["medium_cnt", "micro_cnt"]:
    if c not in informal_cnt.columns:
        informal_cnt[c] = 0
    informal_cnt[c] = pd.to_numeric(informal_cnt[c], errors="coerce").fillna(0).astype(int)

# Complete mappings
city2prov     = dict(zip(cities_df["city_code"], cities_df["province"]))
city2cluster  = dict(zip(cities_df["city_code"], cities_df["cluster_id"]))
prov2cities   = cities_df.groupby("province")["city_code"].apply(set).to_dict()
cluster2cities= cities_df.groupby("cluster_id")["city_code"].apply(set).to_dict()

all_city_codes: Set[str] = set(cities_df["city_code"])
all_provs: Set[str] = set(cities_df["province"].unique())
prov_neighbors: Dict[str, Set[str]] = prov_adj_clean.groupby("province")["neighbor_province"].apply(set).to_dict()
for p in all_provs:
    prov_neighbors.setdefault(p, set())

# Distance matrix
dist_raw.index = [norm_str(i) for i in dist_raw.index]
dist_raw.columns = [norm_str(c) for c in dist_raw.columns]
def to_dist_val(x):
    s = norm_str(x)
    if s == "" or s.lower() in {"na", "nan", "inf"}:
        return np.inf
    try:
        v = float(s)
        return (np.inf if np.isnan(v) else v)
    except:
        return np.inf
dist_wide = dist_raw.stack().map(to_dist_val).unstack()

for cid in all_city_codes:
    if cid not in dist_wide.columns:
        dist_wide[cid] = np.inf
for cid in all_city_codes:
    if cid not in dist_wide.index:
        dist_wide.loc[cid] = np.inf
dist_wide = dist_wide.loc[sorted(all_city_codes), sorted(all_city_codes)]
np.fill_diagonal(dist_wide.values, 0.0)

def city_city_distance(a: str, b: str) -> float:
    a = norm_str(a); b = norm_str(b)
    try:
        return float(dist_wide.at[a, b])
    except Exception:
        return np.inf

# Pre-calculation: city -> target province closest city and distance (for province radius/neighbor province filtering)
rows = []
for oc in sorted(all_city_codes):
    for tp in sorted(all_provs):
        cities = prov2cities.get(tp, set())
        if not cities:
            rows.append((oc, tp, np.inf, "")); continue
        best_d, best_city = np.inf, ""
        for tc in cities:
            d = city_city_distance(oc, tc)
            if d < best_d:
                best_d, best_city = d, tc
        rows.append((oc, tp, best_d, best_city))
min_dist_cp = pd.DataFrame(rows, columns=["city_code", "target_province", "min_distance_km", "target_city"])

# Intra-province average inter-city distance (for informal use)
def provincial_avg_intra_city_distance() -> Dict[str, float]:
    avg_dist = {}
    for p in all_provs:
        cities = sorted(list(prov2cities.get(p, set())))
        if len(cities) <= 1:
            avg_dist[p] = LOCAL_CITY_DISTANCE_KM
            continue
        ds = []
        for i in range(len(cities)):
            for j in range(i+1, len(cities)):
                d = city_city_distance(cities[i], cities[j])
                if np.isfinite(d) and d > 0:
                    ds.append(d)
        if ds:
            avg_dist[p] = float(np.mean(ds))
        else:
            avg_dist[p] = LOCAL_CITY_DISTANCE_KM
    return avg_dist

prov_avg_intra_km = provincial_avg_intra_city_distance()

# Candidate provinces (by scenario)
def candidate_provinces_baseline(city: str) -> Set[str]:
    own = city2prov.get(city, "")
    return {own} if own else set()

def candidate_provinces_radius300(city: str) -> Set[str]:
    sub = min_dist_cp[(min_dist_cp["city_code"]==city) & (min_dist_cp["min_distance_km"]<=RADIUS_KM)]
    cands = set(sub["target_province"].tolist())
    own = city2prov.get(city, "")
    if own: cands.add(own)
    return {p for p in cands if p}

def candidate_provinces_citycluster(city: str) -> Set[str]:
    cid = city2cluster.get(city, "")
    cand_cities = cluster2cities.get(cid, set())
    cands = {city2prov.get(c, "") for c in cand_cities if c}
    own = city2prov.get(city, "")
    if own: cands.add(own)
    return {p for p in cands if p}

def candidate_provinces_neighbor(city: str) -> Set[str]:
    own = city2prov.get(city, "")
    if not own: return set()
    return {own} | prov_neighbors.get(own, set())

# Capacity city set (different each year)
def candidate_target_cities_for(city: str, scenario: str, year: int) -> List[str]:
    """Return list of "target cities (with capacity)" that origin city can go to in specified scenario/year"""
    if scenario == "baseline_inprov":
        cprov = candidate_provinces_baseline(city)
    elif scenario == "radius300":
        cprov = candidate_provinces_radius300(city)
    elif scenario == "citycluster":
        cprov = candidate_provinces_citycluster(city)
    elif scenario == "neighbor_prov":
        cprov = candidate_provinces_neighbor(city)
    else:
        raise ValueError("unknown scenario")

    cap_cities_y = cap_city_year[cap_city_year["year"]==year]
    if cprov:
        cap_cities_y = cap_cities_y[cap_cities_y["province"].isin(cprov)]
    # Only keep cities with capacity in current year
    cap_cities_y = cap_cities_y[cap_cities_y["capacity_tons"] > 0]
    return sorted(cap_cities_y["city_code"].unique().tolist())

# Formal capacity remaining (city level)
def formal_capacity_dict_for_year(year: int) -> Dict[str, float]:
    cap = cap_city_year[cap_city_year["year"]==year].groupby("city_code", as_index=False)["capacity_tons"].sum()
    d = dict(zip(cap["city_code"], cap["capacity_tons"]))
    for c in all_city_codes:
        d.setdefault(c, 0.0)
    return d

# =============== K calibration (baseline scenario; annual) ===============
def year_formal_target_share(y: int) -> float:
    return float(FORMAL_TARGET_SHARE_YEAR.get(y, DEFAULT_FORMAL_SHARE))

def allocate_formal_flows_given_share(year: int, scenario: str, share: float) -> Tuple[pd.DataFrame, float]:
    """
    Given 'national formal share share', allocate formal flows by scenario (using share*EOL as upper limit for city formal supply, allocate city by city).
    Returns: flows_df, national_formal_served
    """
    cap_left = formal_capacity_dict_for_year(year)  # Target city (cap-city) remaining
    eol_y = eol_city_year[eol_city_year["year"]==year].copy()
    if eol_y.empty:
        return pd.DataFrame(columns=["year","origin_city","origin_province","target_province","target_city","tons","distance_km","ton_km","scenario"]), 0.0

    eol_y["formal_supply"] = eol_y["eol_tons"] * share
    # Origin province (for labeling)
        # Merge city province, but avoid KeyError caused by province_x/province_y
    eol_y = eol_y.merge(
        cities_df[["city_code","province"]].rename(columns={"province":"province_city"}),
        on="city_code", how="left"
    )
    # If eol_city_year already has province, use it first; otherwise use province_city mapped from city
    eol_y["origin_province"] = coalesce_cols(eol_y, ["province", "province_city"]).map(norm_lower)
    # Clean temporary columns
    drop_cols = [c for c in ["province","province_city"] if c in eol_y.columns]
    if drop_cols: eol_y = eol_y.drop(columns=drop_cols)

    flows = []
    # Large cities first, to consume supply faster (optional)
    eol_y = eol_y.sort_values("formal_supply", ascending=False)
    for r in eol_y.itertuples(index=False):
        oc = norm_str(r.city_code)
        origin_prov = norm_lower(r.origin_province)
        remain = float(getattr(r, "formal_supply"))
        if remain <= 0:
            continue

        # Candidate target "capacity cities"
        cand_targets = candidate_target_cities_for(oc, scenario, year)
        # Sort by "same province priority + distance ascending"
        def key_func(tc):
            tprov = city2prov.get(tc, "")
            dkm = city_city_distance(oc, tc)
            return (0 if tprov==origin_prov else 1, dkm)

        cand_targets = sorted(cand_targets, key=key_func)

        for tc in cand_targets:
            if remain <= 0: break
            cap = cap_left.get(tc, 0.0)
            if cap <= 0: continue
            # Distance
            dkm = city_city_distance(oc, tc)
            if oc == tc:  # Same city
                dkm = LOCAL_CITY_DISTANCE_KM
            tons = min(remain, cap)
            flows.append((
                year, oc, origin_prov, city2prov.get(tc,""), tc, tons, dkm, tons*dkm, scenario
            ))
            cap_left[tc] = cap - tons
            remain -= tons

    flows_df = pd.DataFrame(flows, columns=["year","origin_city","origin_province","target_province","target_city","tons","distance_km","ton_km","scenario"])
    return flows_df, float(flows_df["tons"].sum()) if not flows_df.empty else 0.0

def calibrate_k_for_year(year: int, scenario: str) -> Tuple[float, pd.DataFrame]:
    """
    Only execute in baseline_inprov scenario: find K such that "national formal actual served amount / national EOL" ≈ target share (tolerance 0.1%)
    Specific implementation: use share*=k method, but here use equivalent "direct scan formal_share", stable enough.
    Returns: best_share, flows_df (allocated using best_share)
    """
    target = year_formal_target_share(year)
    if scenario != "baseline_inprov":
        # Other scenarios don't calibrate, return baseline share and empty flows (outside will reuse baseline K)
        flows0, served0 = allocate_formal_flows_given_share(year, scenario, target)
        return target, flows0

    # National EOL
    eol_tot = float(eol_city_year.loc[eol_city_year["year"]==year, "eol_tons"].sum())
    if eol_tot <= 0:
        return target, pd.DataFrame(columns=["year","origin_city","origin_province","target_province","target_city","tons","distance_km","ton_km","scenario"])

    # Grid search (can also use binary search, here consider discrete 0.05 step then refine)
    candidates = list(np.clip(np.linspace(max(0, target-0.15), min(1.0, target+0.15), 13), 0, 1))
    best = (1e9, target, None)  # err, share, flows
    for s in candidates:
        flows_s, served_s = allocate_formal_flows_given_share(year, "baseline_inprov", s)
        rate = served_s / eol_tot if eol_tot>0 else 0.0
        err = abs(rate - target)
        if err < best[0]:
            best = (err, s, flows_s)
    # Small range refinement
    lo = max(0.0, best[1]-0.05); hi = min(1.0, best[1]+0.05)
    for s in np.linspace(lo, hi, 21):
        flows_s, served_s = allocate_formal_flows_given_share(year, "baseline_inprov", s)
        rate = served_s / eol_tot if eol_tot>0 else 0.0
        err = abs(rate - target)
        if err < best[0]:
            best = (err, s, flows_s)
            if err <= 1e-3:  # Tolerance: 0.1 percentage point
                break

    best_share, flows_df = best[1], best[2]
    print(f"[Calib] baseline_inprov {year}: target={target:.4f}, best_share={best_share:.4f}, "
          f"formal_rate={ (float(flows_df['tons'].sum())/eol_tot if not flows_df.empty and eol_tot>0 else 0.0):.4f}")
    return best_share, flows_df

# =============== Run scenarios (reuse K) ===============
def run_scenario(scenario: str, fixed_share_by_year: Dict[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    fixed_share_by_year: Annual fixed "national formal share" (from baseline scenario calibration results)
    Returns: flows, recv(formal receiving side), origin(source province balance), city_out(city level), bytype(province by type)
    """
    years = sorted(eol_city_year["year"].unique().tolist())
    all_flows, all_recv, all_city, all_origin, all_bytype = [], [], [], [], []

    for y in years:
        share = float(fixed_share_by_year.get(y, year_formal_target_share(y)))

        # 1) Formal allocation (by share)
        flows_y, served_y = allocate_formal_flows_given_share(y, scenario, share)
        if flows_y is None or flows_y.empty:
            flows_y = pd.DataFrame(columns=["year","origin_city","origin_province","target_province","target_city","tons","distance_km","ton_km","scenario"])
        all_flows.append(flows_y)

        # 2) City level: formal served amount
        if not flows_y.empty:
            formal_city = flows_y.groupby("origin_city", as_index=False)["tons"].sum().rename(columns={"tons":"formal_tons_served"})
        else:
            formal_city = pd.DataFrame(columns=["origin_city","formal_tons_served"])

                # eol + city province (renamed to province_city to avoid province_x/y)
        city_all = eol_city_year[eol_city_year["year"]==y][["city_code","year","eol_tons"] + (["province"] if "province" in eol_city_year.columns else [])].copy()
        if "province" in city_all.columns:
            city_all = city_all.rename(columns={"province":"province_eol"})
        city_all = city_all.merge(
            cities_df[["city_code","province"]].rename(columns={"province":"province_city"}),
            on="city_code", how="left"
        )
        # Unify to origin_province: prioritize province in eol (if exists), otherwise use province_city mapped from city
        city_all["origin_province"] = coalesce_cols(city_all, ["province_eol", "province_city"]).map(norm_lower)
        # Clean temporary columns
        for c in ["province_eol","province_city"]:
            if c in city_all.columns: city_all.drop(columns=[c], inplace=True)


        city_all = city_all.merge(formal_city, left_on="city_code", right_on="origin_city", how="left") \
                           .drop(columns=["origin_city"], errors="ignore")
        city_all["formal_tons_served"] = city_all["formal_tons_served"].fillna(0.0)

        # Informal = remaining (ensure unrecycled=0)
        city_all["informal_tons_served"] = (city_all["eol_tons"] - city_all["formal_tons_served"]).clip(lower=0.0)
        city_all["unrecycled"] = 0.0

        # 3) Informal t·km (by province average inter-city distance)
        INFORMAL_LOCAL_DISTANCE_KM = 10.0
        city_all["informal_tkm"] = city_all["informal_tons_served"] * INFORMAL_LOCAL_DISTANCE_KM


        # City output
        city_out = city_all.rename(columns={"origin_province":"province"})[[
            "year","city_code","province","eol_tons","formal_tons_served","informal_tons_served","unrecycled","informal_tkm"
        ]]
        city_out["scenario"] = scenario
        all_city.append(city_out)

        # 4) Province receiving side (formal)
        if not flows_y.empty:
            recv_served = flows_y.groupby(["target_province"], as_index=False).agg(
                served_tons=("tons","sum"),
                inbound_ton_km=("ton_km","sum")
            ).rename(columns={"target_province":"province"})
        else:
            recv_served = pd.DataFrame(columns=["province","served_tons","inbound_ton_km"])
        # Capacity (aggregated by province)
        cap_y_prov = cap_city_year[cap_city_year["year"]==y].groupby("province", as_index=False)["capacity_tons"].sum()
        recv_y = cap_y_prov.merge(recv_served, on="province", how="left").fillna({"served_tons":0.0,"inbound_ton_km":0.0})
        recv_y["year"] = y
        recv_y["avg_haul_km"] = np.where(recv_y["served_tons"]>0, recv_y["inbound_ton_km"]/recv_y["served_tons"], 0.0)
        recv_y["utilization"] = np.where(recv_y["capacity_tons"]>0, recv_y["served_tons"]/recv_y["capacity_tons"], 0.0)

        # Source province outbound t·km (formal)
        if not flows_y.empty:
            outbound = flows_y.groupby(["origin_province"], as_index=False)["ton_km"].sum() \
                              .rename(columns={"origin_province":"province","ton_km":"outbound_ton_km"})
        else:
            outbound = pd.DataFrame(columns=["province","outbound_ton_km"])
        recv_y = recv_y.merge(outbound, on="province", how="left").fillna({"outbound_ton_km":0.0})
        recv_y["scenario"] = scenario
        all_recv.append(recv_y)

        # 5) Province source side balance (generated amount/formal/informal/total; unrecycled=0)
        origin = city_out.groupby(["province","year"], as_index=False).agg(
            generated_tons=("eol_tons","sum"),
            formal_served_tons=("formal_tons_served","sum"),
            informal_total_tons=("informal_tons_served","sum"),
            informal_ton_km=("informal_tkm","sum")
        )
        origin["recovered_tons"] = origin["formal_served_tons"] + origin["informal_total_tons"]
        origin["unrecycled_tons"] = 0.0
        origin["scenario"] = scenario
        all_origin.append(origin)

        # 6) Province level "by enterprise type" split (informal by medium/micro weights)
        # Weight source: year, province micro_cnt/medium_cnt → calculate effective "capacity weights"
        inf_cnt_y = informal_cnt[informal_cnt["year"]==y][["province","micro_cnt","medium_cnt"]].copy()
        inf_cnt_y = ensure_province_col(inf_cnt_y)
        # Calculate effective "capacity" (only for weights; not for upper limit)
        inf_cnt_y["cap_micro_eff"]  = inf_cnt_y["micro_cnt"]  * MICRO_CAP_PER_FIRM  * INFORMAL_UTIL_RATE
        inf_cnt_y["cap_medium_eff"] = inf_cnt_y["medium_cnt"] * MEDIUM_CAP_PER_FIRM * INFORMAL_UTIL_RATE

        bytype = origin.merge(inf_cnt_y, on="province", how="left") \
                       .merge(recv_y[["province","capacity_tons","inbound_ton_km","outbound_ton_km"]], on="province", how="left")
        bytype[["cap_micro_eff","cap_medium_eff","capacity_tons","inbound_ton_km","outbound_ton_km"]] = \
            bytype[["cap_micro_eff","cap_medium_eff","capacity_tons","inbound_ton_km","outbound_ton_km"]].fillna(0.0)

        # Allocation
        w_den = bytype["cap_micro_eff"] + bytype["cap_medium_eff"]
        w_mic = np.where(w_den>0, bytype["cap_micro_eff"]/w_den, 0.5)
        w_med = np.where(w_den>0, bytype["cap_medium_eff"]/w_den, 0.5)
        bytype["informal_micro_tons"]  = bytype["informal_total_tons"] * w_mic
        bytype["informal_medium_tons"] = bytype["informal_total_tons"] * w_med

        # Share
        denom = bytype["generated_tons"].replace(0, np.nan)
        bytype["formal_share"]     = (bytype["formal_served_tons"]/denom).fillna(0.0)
        bytype["informal_share"]   = (bytype["informal_total_tons"]/denom).fillna(0.0)
        bytype["unrecycled_share"] = 0.0

        bytype["scenario"] = scenario
        all_bytype.append(bytype)

    # Merge export (complete all province×year)
    flows = pd.concat(all_flows, ignore_index=True) if all_flows else pd.DataFrame()
    recv  = pd.concat(all_recv,  ignore_index=True) if all_recv  else pd.DataFrame()
    city  = pd.concat(all_city,  ignore_index=True) if all_city  else pd.DataFrame()
    origin= pd.concat(all_origin,ignore_index=True) if all_origin else pd.DataFrame()
    bytyp = pd.concat(all_bytype,ignore_index=True) if all_bytype else pd.DataFrame()

    # Province year completion
    all_years = sorted(eol_city_year["year"].unique().tolist())
    full_index = pd.MultiIndex.from_product([all_years, sorted(all_provs)], names=["year","province"])
    def _reindex(df, cols_keep):
        if df.empty:
            return pd.DataFrame(index=full_index).reset_index()[["year","province"]+cols_keep].fillna(0.0)
        out = df.set_index(["year","province"]).reindex(full_index).reset_index()
        for c in cols_keep:
            if c not in out.columns: out[c] = 0.0
        out = out[["year","province"]+cols_keep]
        return out.fillna(0.0)

    recv  = _reindex(recv,  [c for c in ["capacity_tons","served_tons","inbound_ton_km","outbound_ton_km","avg_haul_km","utilization","scenario"] if c in recv.columns])
    origin= _reindex(origin,[c for c in ["generated_tons","formal_served_tons","informal_total_tons","informal_ton_km","recovered_tons","unrecycled_tons","scenario"] if c in origin.columns])
    bytyp = _reindex(bytyp, [c for c in ["capacity_tons","inbound_ton_km","outbound_ton_km",
                                         "generated_tons","formal_served_tons","informal_total_tons",
                                         "informal_micro_tons","informal_medium_tons",
                                         "formal_share","informal_share","unrecycled_share","scenario"] if c in bytyp.columns])

    # Export
    flows.to_csv(os.path.join(OUT_DIR, f"flows_{scenario}.csv"), index=False, encoding="utf-8-sig")
    recv.to_csv(os.path.join(OUT_DIR, f"results_{scenario}_province_year_metrics.csv"), index=False, encoding="utf-8-sig")
    origin.to_csv(os.path.join(OUT_DIR, f"results_{scenario}_origin_province_year_balance.csv"), index=False, encoding="utf-8-sig")
    city.to_csv(os.path.join(OUT_DIR, f"results_{scenario}_city_year_unrecycled.csv"), index=False, encoding="utf-8-sig")  # unrecycled always 0
    bytyp.to_csv(os.path.join(OUT_DIR, f"results_{scenario}_province_year_bytype.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] Exported scenario {scenario}")
    return flows, recv, origin, city, bytyp

# ========================== Main process ==========================
# 1) Baseline scenario annual calibration share (find best_share = essentially "converted K * target share", and get baseline flows)
BEST_SHARE_BY_YEAR: Dict[int, float] = {}
BASELINE_FLOWS_BY_YEAR: Dict[int, pd.DataFrame] = {}
for y in sorted(eol_city_year["year"].unique()):
    best_share, flows_b = calibrate_k_for_year(y, "baseline_inprov")
    BEST_SHARE_BY_YEAR[y] = best_share
    BASELINE_FLOWS_BY_YEAR[y] = flows_b

# 2) Four scenarios run (other three scenarios reuse annual best_share)
for sc in SCENARIOS:
    run_scenario(sc, fixed_share_by_year=BEST_SHARE_BY_YEAR)

# 3) National level verification (print)
for sc in SCENARIOS:
    origin_sc = pd.read_csv(os.path.join(OUT_DIR, f"results_{sc}_origin_province_year_balance.csv"))
    chk = origin_sc.groupby("year", as_index=False)[["formal_served_tons","informal_total_tons","recovered_tons"]].sum()
    eol = eol_city_year.groupby("year", as_index=False)["eol_tons"].sum()
    m = eol.merge(chk, on="year", how="left").fillna(0.0)
    m["formal_rate"]   = np.where(m["eol_tons"]>0, m["formal_served_tons"]/m["eol_tons"], 0.0)
    m["informal_rate"] = np.where(m["eol_tons"]>0, m["informal_total_tons"]/m["eol_tons"], 0.0)
    print(f"\n[National check] {sc}\n", m.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

# 4) Summary export
def _save_concat(patterns: List[str], outname: str):
    dfs=[]
    for sc in SCENARIOS:
        p = os.path.join(OUT_DIR, patterns.format(scenario=sc))
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(os.path.join(OUT_DIR, outname), index=False, encoding="utf-8-sig")

_save_concat("results_{scenario}_province_year_metrics.csv",       "results_ALL_scenarios_province_metrics.csv")
_save_concat("results_{scenario}_origin_province_year_balance.csv","results_ALL_scenarios_origin_balance.csv")
_save_concat("results_{scenario}_city_year_unrecycled.csv",        "results_ALL_scenarios_city_year_unrecycled.csv")
_save_concat("results_{scenario}_province_year_bytype.csv",        "results_ALL_scenarios_province_year_bytype.csv")


# ========================== City transport details Excel export ==========================
def export_city_transport_excel(
    out_path: str = os.path.join(OUT_DIR, "city_transport_summary.xlsx")
):
    """
    Export city-level transportation weight and total transportation distance (ton-kilometers) by scenario:
      - Year, city, city_code,
      - Formal_Recycling_t, Informal_Recycling_t  (t)
      - formal_recycling_t*km, informal_recycling_t*km (t·km)
    One sheet per scenario; includes Notes explanation sheet.
    """
    # City name mapping
    city_name_map = dict(zip(cities_df["city_code"], cities_df["city_name"]))

    # Scenario name → worksheet name
    sheet_names = {
        "baseline_inprov": "Baseline",
        "radius300":       "Radius 300km",
        "citycluster":     "City Cluster",
        "neighbor_prov":   "Neighbor Province",
    }

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for sc in SCENARIOS:
            # Read scenario's city summary and flow details
            city_csv  = os.path.join(OUT_DIR, f"results_{sc}_city_year_unrecycled.csv")
            flows_csv = os.path.join(OUT_DIR, f"flows_{sc}.csv")
            if not os.path.exists(city_csv):
                print(f"[Warn] {city_csv} missing, skip {sc}")
                continue

            df_city = pd.read_csv(city_csv)
            # Ensure fields exist
            needed_city_cols = {"year","city_code","formal_tons_served","informal_tons_served","informal_tkm"}
            missing = needed_city_cols - set(df_city.columns)
            if missing:
                raise ValueError(f"{city_csv} missing columns: {missing}")

            # Formal ton-kilometers: from flows (aggregate ton_km by origin_city)
            if os.path.exists(flows_csv):
                df_flows = pd.read_csv(flows_csv)
                df_flows.columns = [c.strip().lower() for c in df_flows.columns]
                # Error tolerance: empty file
                if not df_flows.empty and {"year","origin_city","ton_km"}.issubset(df_flows.columns):
                    formal_tkm = (df_flows.groupby(["year","origin_city"], as_index=False)["ton_km"]
                                         .sum()
                                         .rename(columns={"origin_city":"city_code",
                                                          "ton_km":"formal_tkm"}))
                else:
                    formal_tkm = pd.DataFrame(columns=["year","city_code","formal_tkm"])
            else:
                formal_tkm = pd.DataFrame(columns=["year","city_code","formal_tkm"])

            # Merge: city tons, informal ton-kilometers + formal ton-kilometers
            out = df_city.merge(
                formal_tkm, on=["year","city_code"], how="left"
            )
            out["formal_tkm"]   = pd.to_numeric(out.get("formal_tkm", 0.0), errors="coerce").fillna(0.0)
            out["informal_tkm"] = pd.to_numeric(out["informal_tkm"], errors="coerce").fillna(0.0)

            # Organize fields and rename
            out["city"] = out["city_code"].map(city_name_map).fillna("")
            out = out[["year","city","city_code","formal_tons_served","informal_tons_served","formal_tkm","informal_tkm"]]
            out = out.rename(columns={
                "year": "Year",
                "formal_tons_served": "Formal_Recycling_t",
                "informal_tons_served": "Informal_Recycling_t",
                "formal_tkm": "formal_recycling_t*km",
                "informal_tkm": "informal_recycling_t*km",
            })
            # Type/sort
            out["Year"] = out["Year"].astype(int)
            out = out.sort_values(["Year","city_code"]).reset_index(drop=True)

            # Write to corresponding worksheet
            sheet = sheet_names.get(sc, sc)
            out.to_excel(writer, sheet_name=sheet, index=False)

        # Notes page
        notes = pd.DataFrame({
            "Field": [
                "Year",
                "city",
                "city_code",
                "Formal_Recycling_t",
                "Informal_Recycling_t",
                "formal_recycling_km",
                "informal_recycling_km",
            ],
            "Unit": [
                "-", "-", "-",
                "t",
                "t",
                "t·km (ton-kilometers)",
                "t·km (ton-kilometers)",
            ],
            "Definition": [
                "Year",
                "City Chinese name or English name (from cities.csv city_name)",
                "City code (6-digit string)",
                "Total weight collected by formal enterprises in this city this year (tons)",
                "Total weight collected by informal enterprises in this city this year (tons)",
                "Total transportation distance for formal collection (ton-kilometers), ∑(tons × distance_km)",
                "Total transportation distance for informal collection (ton-kilometers), calculated as 'intra-province average inter-city distance × informal total amount'",
            ]
        })
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"[OK] Excel exported → {out_path}")

# == Call export ==
export_city_transport_excel()