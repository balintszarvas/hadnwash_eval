import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------
# File paths – adjust if your directories differ
# ----------------------------------------------------------------------------
AGG_DIR = Path('aggregated_data')
FEATS_FILE = AGG_DIR / 'dataset_with_engineered_features.csv'
AGG5S_FILE = AGG_DIR / '5s_overlap1s_aggregated.csv'
OUTPUT_FILE = AGG_DIR / 'merged_features_5s.csv'

# ----------------------------------------------------------------------------
# 1. Load both datasets
# ----------------------------------------------------------------------------
print(f'Loading engineered feature file: {FEATS_FILE}')
feats_df = pd.read_csv(FEATS_FILE, parse_dates=['datetime'])
print(f'Original engineered features shape: {feats_df.shape}')

print(f'Loading 5-s aggregated file: {AGG5S_FILE}')
agg_df = pd.read_csv(AGG5S_FILE, parse_dates=['datetime'])
print(f'5-s aggregated shape: {agg_df.shape}')

# ---------------------------------------------------------------------
# 2.a Adjust end_time so that it refers to the logical window end
# ---------------------------------------------------------------------
# Determine the sampling interval (assumes constant rate)
if len(agg_df) > 1:
    sample_interval_sec = (agg_df['datetime'].iloc[1] - agg_df['datetime'].iloc[0]).total_seconds()
else:
    sample_interval_sec = 0.25  # default fallback

# Shift end_time and wrap around every 40 s wash period
if 'end_time' in agg_df.columns:
    agg_df['end_time'] = (agg_df['end_time'] + sample_interval_sec) % 40

# ---------------------------------------------------------------------
# 2.b Drop absolute timestamp breadcrumbs
# ---------------------------------------------------------------------
agg_df.drop(columns=[c for c in ['abs_start_dt', 'abs_end_dt'] if c in agg_df.columns], inplace=True)

# ----------------------------------------------------------------------------
# 2. Clean engineered-features table
#    – drop first 3 data rows (keep header)
#    – drop unwanted column
# ----------------------------------------------------------------------------
feats_df = feats_df.iloc[3:].reset_index(drop=True)

col_to_drop = 'score_temp_mode_ws_8'
if col_to_drop in feats_df.columns:
    feats_df.drop(columns=[col_to_drop], inplace=True)
    print(f'Dropped column "{col_to_drop}"')
else:
    print(f'Column "{col_to_drop}" not found (already absent)')
print(f'Engineered features shape after cleaning: {feats_df.shape}')

# ----------------------------------------------------------------------------
# 3. Merge on datetime (inner join keeps matching windows)
# ----------------------------------------------------------------------------
merged_df = pd.merge(agg_df, feats_df, on='datetime', how='inner', suffixes=('', '_engineered'))
print(f'Merged dataset shape: {merged_df.shape}')

# ----------------------------------------------------------------------------
# 4. Save result
# ----------------------------------------------------------------------------
merged_df.to_csv(OUTPUT_FILE, index=False)
print(f'Saved merged dataset to {OUTPUT_FILE}') 