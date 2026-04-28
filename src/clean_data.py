from pathlib import Path
import pandas as pd
import numpy as np

def main():

    # -----------------------------
    # Paths
    # -----------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent

    INPUT_PATH = BASE_DIR / "data" / "raw" / "raw_data.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cleaned_wildlife_strikes.csv"

    # ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # creates folders if needed :contentReference[oaicite:0]{index=0}

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(INPUT_PATH)

    # -----------------------------
    # Keep relevant columns
    # -----------------------------
    cols_to_keep = [
        'INCIDENT_DATE', 'INCIDENT_YEAR', 'INCIDENT_MONTH', 'TIME', 'TIME_OF_DAY',
        'AIRPORT', 'STATE', 'LATITUDE', 'LONGITUDE', 'RUNWAY', 'OPID', 'OPERATOR',
        'AIRCRAFT', 'AC_CLASS', 'AC_MASS', 'NUM_ENGS', 'TYPE_ENG',
        'PHASE_OF_FLIGHT', 'SPECIES', 'NUM_SEEN', 'NUM_STRUCK', 'SIZE',
        'DAMAGE_LEVEL', 'NR_INJURIES', 'NR_FATALITIES', 'COST_REPAIRS',
        'COST_OTHER', 'COST_REPAIRS_INFL_ADJ', 'COST_OTHER_INFL_ADJ',
        'HEIGHT', 'SPEED', 'DISTANCE', 'WARNED'
    ]

    df = df[cols_to_keep].copy()

    # -----------------------------
    # Fill categorical
    # -----------------------------
    cat_cols = [
        'STATE', 'TIME_OF_DAY', 'RUNWAY', 'AC_CLASS', 'TYPE_ENG',
        'PHASE_OF_FLIGHT', 'SIZE', 'DAMAGE_LEVEL', 'NUM_STRUCK', 'NUM_SEEN',
        'WARNED'
    ]
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # -----------------------------
    # Convert numeric
    # -----------------------------
    numeric_cols = [
        'LATITUDE', 'LONGITUDE', 'HEIGHT', 'SPEED', 'DISTANCE',
        'NR_INJURIES', 'NR_FATALITIES',
        'COST_REPAIRS', 'COST_OTHER',
        'COST_REPAIRS_INFL_ADJ', 'COST_OTHER_INFL_ADJ'
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # -----------------------------
    # Date + time
    # -----------------------------
    df['INCIDENT_DATE'] = pd.to_datetime(
        df['INCIDENT_DATE'],
        format='%Y-%m-%d',
        errors='coerce'
    )

    time_dt = pd.to_datetime(df['TIME'], format='%H:%M', errors='coerce')

    df['HOUR'] = time_dt.dt.hour
    df['TIME'] = time_dt.dt.time

    df['TIME_BUCKET'] = pd.cut(
        df['HOUR'],
        bins=[-1, 5, 11, 17, 23],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )

    # -----------------------------
    # Binary indicators
    # -----------------------------
    df['HAS_DAMAGE'] = df['DAMAGE_LEVEL'].map({
        'N': 0,
        'M': 1,
        'S': 1,
        'D': 1,
        'M?': np.nan,
        'Unknown': np.nan
    })

    df['HAS_INJURY'] = (df['NR_INJURIES'] > 0).astype(int)
    df['HAS_FATALITY'] = (df['NR_FATALITIES'] > 0).astype(int)

    # -----------------------------
    # Costs
    # -----------------------------
    df['TOTAL_COST'] = df['COST_REPAIRS'] + df['COST_OTHER']
    df['TOTAL_COST_INFL_ADJ'] = (
        df['COST_REPAIRS_INFL_ADJ'] + df['COST_OTHER_INFL_ADJ']
    )

    # -----------------------------
    # Categorical mappings
    # -----------------------------
    df['WARNED_CLEAN'] = df['WARNED'].map({
        'Yes': 1,
        'No': 0,
        'Unknown': np.nan
    })

    engine_map = {
        'A': 'Piston', 'B': 'Turbojet', 'C': 'Turboprop',
        'D': 'Turbofan', 'E': 'Glider', 'F': 'Helicopter',
        'Y': 'Other', 'Unknown': 'Unknown'
    }
    df['ENGINE_TYPE'] = df['TYPE_ENG'].map(engine_map).fillna('Unknown')

    damage_map = {
        'N': 'None', 'M': 'Minor', 'M?': 'Undetermined',
        'S': 'Substantial', 'D': 'Destroyed', 'Unknown': 'Unknown'
    }
    df['DAMAGE_CATEGORY'] = df['DAMAGE_LEVEL'].map(damage_map)

    phase_map = {
        'Take-off Run': 'Takeoff', 'Departure': 'Takeoff',
        'Climb': 'Climb', 'En Route': 'Cruise',
        'Descent': 'Descent', 'Approach': 'Approach',
        'Arrival': 'Approach', 'Landing Roll': 'Landing',
        'Taxi': 'Ground', 'Parked': 'Ground',
        'Local': 'Unknown', 'Unknown': 'Unknown'
    }
    df['PHASE_GROUP'] = df['PHASE_OF_FLIGHT'].map(phase_map)

    df['OPERATOR_TYPE'] = df['OPID'].replace({
        'PVT': 'Private',
        'BUS': 'Business',
        'GOV': 'Government',
        'MIL': 'Military'
    })

    # -----------------------------
    # Drop unused
    # -----------------------------
    df = df.drop(columns=['TYPE_ENG', 'DAMAGE_LEVEL', 'PHASE_OF_FLIGHT'])

    # -----------------------------
    # Save
    # -----------------------------
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()