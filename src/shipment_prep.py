import pandas as pd
import numpy as np

# Operational target milestones (SLA) in hours
EXPRESS_HOURS = 36.0
STANDARD_HOURS = 60.0
ECONOMY_HOURS = 96.0

def clean_shipments(raw_df):
    """
    Step A1: Data Hygiene Engine
    - Removes structural duplicates by shipment_id.
    - Strips whitespace and standardizes column strings to Title case.
    - Purges the 999.0 kg machine error sentinel records.
    - Removes currency symbol blocks and converts values to clean floats.
    """
    df = raw_df.copy()
    
    # 1. Eliminate duplicate records
    df = df.drop_duplicates(subset=['shipment_id']).reset_index(drop=True)
    
    # 2. Rectify categorical case and padding variations
    if 'service_type' in df.columns:
        df['service_type'] = df['service_type'].astype(str).str.strip().str.title()
    if 'payment_mode' in df.columns:
        df['payment_mode'] = df['payment_mode'].astype(str).str.strip().str.title()
        
    # 3. Drop the anomalous 999.0 kg machine sentinel records
    if 'parcel_weight_kg' in df.columns:
        df = df[df['parcel_weight_kg'] != 999.0].reset_index(drop=True)
        
    # 4. Strip currency markings (INR and commas) to make it numeric
    if 'declared_value' in df.columns:
        df['declared_value'] = df['declared_value'].astype(str)\
                               .str.replace('₹', '', regex=False)\
                               .str.replace('INR', '', regex=False)\
                               .str.replace(',', '', regex=False)\
                               .str.strip()
        df['declared_value'] = pd.to_numeric(df['declared_value'], errors='coerce')
        
    # 5. Convert date fields to standard datetime format
    if 'dispatch_time' in df.columns:
        df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
        
    return df

def build_features(cleaned_df, hubs_df):
    """
    Step A2: Predictive Feature Engineering
    - Merges core shipment records with regional hub details.
    - Imputes remaining missing cells using dataset median baselines.
    - Calculates target promised SLA delivery windows.
    """
    df = cleaned_df.copy()
    
    # 1. Enrich table with structural hub configurations
    if hubs_df is not None and 'hub_id' in df.columns:
        df = df.merge(hubs_df, on='hub_id', how='left')
        
    # 2. Handle missing cells using column medians
    for col in ['distance_km', 'parcel_weight_kg', 'declared_value']:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    if 'payment_mode' in df.columns:
        # Fill missing values using the most common strategy (mode)
        df['payment_mode'] = df['payment_mode'].fillna(df['payment_mode'].mode()[0])
        
    # 3. Establish promised delivery windows based on service tier
    if 'service_type' in df.columns:
        conditions = [
            (df['service_type'] == 'Express'),
            (df['service_type'] == 'Standard'),
            (df['service_type'] == 'Economy')
        ]
        choices = [EXPRESS_HOURS, STANDARD_HOURS, ECONOMY_HOURS]
        df['promised_sla_hours'] = np.select(conditions, choices, default=STANDARD_HOURS)
        
    return df
