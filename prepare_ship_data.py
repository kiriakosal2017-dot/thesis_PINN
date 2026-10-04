"""Merge raw per-vessel CSV files and normalise their column names to the reference
schema, producing a single Excel file per vessel that the downstream loaders can read
without any further renaming.

The mapping from a vessel identifier to its raw-data location and column map lives in a
local ``ship_registry.py`` (not version-controlled, since it contains private paths and
vessel-specific details). Copy ``ship_registry.example.py`` to ``ship_registry.py`` and
fill it in before running this script.
"""

import sys
import glob
import pandas as pd
from pathlib import Path

# The onboard logger appends the sensor source and unit to every column header
# (e.g. "M/E Shaft RPM_TRQM (rpm)"). The right-hand values are the canonical names the
# rest of the pipeline expects.
LAROS_COLUMN_MAP = {
    "TIME": "TIME",
    "Speed Through Water_TRQM (knots)": "Speed-Through-Water",
    "Longitudinal Water Speed_BRG_SLOG (knots)": "Longitudinal_water_speed_BRG_SLOG",
    "Wind Speed_BRG_WIND (m/s)": "Wind_speed_BRG_WIND",
    "Wind Speed_m/s_BRG_WIND (m/s)": "Wind_Speed_m/s_BRG_WIND",
    "Wind Angle_BRG_WIND (degrees)": "Wind_angle_BRG_WIND",
    "Fore draft_AMS (m)": "Fore draft_AMS",
    "Aft draft_AMS (m)": "Aft draft_AMS",
    "Middle draft(P)_AMS (m)": "Middle draft(P)_AMS",
    "Middle draft(S)_AMS (m)": "Middle draft(S)_AMS",
    "M/E Shaft RPM_TRQM (rpm)": "Propeller-Shaft-RPM",
    "Speed Over Ground_BRG_GPS_ (knots)": "Speed-Over-Ground",
    "Shaft Power_TRQM (kW)": "Shaft Power_TRQM",
    "Shaft Torque_TRQM (kNm)": "Shaft Torque_TRQM",
    "Shaft Thrust_TRQM (kN)": "Shaft Thrust_TRQM",
    "True Heading_BRG_GYRO (degrees)": "Heading_BRG_GYRO",
    "Longitude_BRG_GPS (degrees)": "Vessel-Longitude",
    "Latitude_BRG_GPS (degrees)": "Vessel-Latitude",
    "True Course Over Ground_BRG_GPS (degrees)": "True_Course_over_ground_BRG_GPS_",
    "Magnetic Variation_degrees E/W_BRG_GPS (degrees)": "Magnetic_variation_BRG_GPS_",
    "Rate of Turn_BRG_GYRO (degrees/min)": "Rate_of_turn_BRG_GYRO",
    "Water Depth Relative to the Transducer_BRG_ECHO (m)": "Water_depth_relative_to_the_transducer_BRG_ECHO",
    "ME RPM_AMS (rpm)": "ME RPM_AMS",
    "Trim_AMS (m)": "Trim_AMS",
    "List_AMS (degrees)": "List_AMS",
    "Wind Speed_knots_BRG_WIND (knots)": "Rel-Wind-Speed",
    "Starboard Rudder_BRG_AUTOP (degrees)": "Starboard_rudder_sensor_BRG_AUTOP",
}

# A second logger firmware variant uses slightly different sensor labels.
LAROS_COLUMN_MAP_V2 = {
    "TIME": "TIME",
    "Speed over water_BRG_SLOG (knots)": "Speed-Through-Water",
    "Longitudinal water speed_BRG_SLOG (knots)": "Longitudinal_water_speed_BRG_SLOG",
    "Wind speed_BRG_WIND (NULL)": "Wind_speed_BRG_WIND",
    "Wind Speed_m/s_BRG_WIND (m/s)": "Wind_Speed_m/s_BRG_WIND",
    "Wind Speed_knots_BRG_WIND (knots)": "Rel-Wind-Speed",
    "Wind angle_BRG_WIND (degrees)": "Wind_angle_BRG_WIND",
    "FWD DRAFT_AMS (m)": "Fore draft_AMS",
    "AFT DRAFT_AMS (m)": "Aft draft_AMS",
    "MIDDLE DRAFT(P)_AMS (m)": "Middle draft(P)_AMS",
    "MIDDLE DRAFT(S)_AMS (m)": "Middle draft(S)_AMS",
    "Shaft_Rpm_TRQM (rpm)": "Propeller-Shaft-RPM",
    "Speed over ground_BRG_GPS (knots)": "Speed-Over-Ground",
    "Shaft_Power_TRQM (kW)": "Shaft Power_TRQM",
    "Shaft_Torque_TRQM (kNm)": "Shaft Torque_TRQM",
    "Shaft_Thrust_TRQM (kN)": "Shaft Thrust_TRQM",
    "Heading_BRG_GYRO (degrees True)": "Heading_BRG_GYRO",
    "Latitude_BRG_GPS ( N/S)": "Vessel-Latitude",
    "Longitude_BRG_GPS ( E/W)": "Vessel-Longitude",
    "Course Over Ground_BRG_GPS_ (degrees True)": "True_Course_over_ground_BRG_GPS_",
    "Rate of Turn_BRG_GYRO (degrees/min)": "Rate_of_turn_BRG_GYRO",
    "ME RPM_AMS (rpm)": "ME RPM_AMS",
    "Trim_AMS (m)": "Trim_AMS",
}


def build_metis_column_map(vessel):
    """Build the column mapping for a Metis-platform export.

    Metis exports use fully-qualified sensor descriptions that embed the vessel
    identifier, e.g. 'Vessel Hull Through Water Longitudinal Speed ... - <vessel> [VALUE]'.
    The keys are generated programmatically from the identifier rather than kept as a
    hard-coded dict per vessel.
    """
    s = vessel.capitalize()
    return {
        f"Time [TIMESTAMP]": "TIME",
        f"Vessel Hull Through Water Longitudinal Speed (Instrument Speedlog) - {s} [VALUE]": "Speed-Through-Water",
        f"Vessel External Conditions Wind Relative Speed (Instrument Anemometer) - {s} [VALUE]": "Rel-Wind-Speed",
        f"Vessel External Conditions Wind Relative Angle (Instrument Anemometer) - {s} [VALUE]": "Wind_angle_BRG_WIND",
        f"Vessel External Conditions Wind True Speed (Provider MeteoBlue) - {s} [VALUE]": "Wind_Speed_m/s_BRG_WIND",
        f"Vessel Hull Fore Draft (Control Alarm Monitoring System) - {s} [VALUE]": "Fore draft_AMS",
        f"Vessel Hull Aft Draft (Control Alarm Monitoring System) - {s} [VALUE]": "Aft draft_AMS",
        f"Vessel Hull MidP Draft (Control Alarm Monitoring System) - {s} [VALUE]": "Middle draft(P)_AMS",
        f"Vessel Hull MidS Draft (Control Alarm Monitoring System) - {s} [VALUE]": "Middle draft(S)_AMS",
        f"Vessel Propeller Shaft Rotational Speed (Instrument Torquemeter) - {s} [VALUE]": "Propeller-Shaft-RPM",
        f"Vessel Hull Over Ground Speed (Instrument GPS 1) - {s} [VALUE]": "Speed-Over-Ground",
        f"Vessel Propeller Shaft Mechanical Power (Instrument Torquemeter) - {s} [VALUE]": "Shaft Power_TRQM",
        f"Vessel Propeller Shaft Torque (Instrument Torquemeter) - {s} [VALUE]": "Shaft Torque_TRQM",
        f"Vessel Propeller Shaft Thrust (Instrument Torquemeter) - {s} [VALUE]": "Shaft Thrust_TRQM",
        f"Vessel Hull Heading True Angle (Instrument Gyrocompass) - {s} [VALUE]": "Heading_BRG_GYRO",
        f"Vessel Hull Longitude Angle (Instrument GPS 1) - {s} [VALUE]": "Vessel-Longitude",
        f"Vessel Hull Latitude Angle (Instrument GPS 1) - {s} [VALUE]": "Vessel-Latitude",
        f"Main Engine Rotational Speed (Control Alarm Monitoring System) - {s} [VALUE]": "ME RPM_AMS",
        f"Vessel Hull Trim Draft (Metis Processing) - {s} [VALUE]": "Trim_AMS",
    }


# The reference schema every vessel file is reduced to, so the downstream loaders see an
# identical column set regardless of source. Engine temperatures, tank levels, and other
# vessel-specific signals are intentionally excluded.
REFERENCE_SCHEMA_COLUMNS = [
    "TIME", "Speed-Through-Water", "Rel-Wind-Speed", "Rel-Wind-Direction",
    "Fore draft_AMS", "Aft draft_AMS",
    "Middle draft(P)_AMS", "Middle draft(S)_AMS",
    "Propeller-Shaft-RPM", "Speed-Over-Ground",
    "Shaft Power_TRQM", "Shaft Torque_TRQM", "Shaft Thrust_TRQM",
    "Heading_BRG_GYRO", "Vessel-Longitude", "Vessel-Latitude",
    "Longitudinal_water_speed_BRG_SLOG",
    "Water_depth_relative_to_the_transducer_BRG_ECHO",
    "Wind_angle_BRG_WIND", "Wind_speed_BRG_WIND", "Wind_Speed_m/s_BRG_WIND",
    "True_Course_over_ground_BRG_GPS_", "Magnetic_variation_BRG_GPS_",
    "Rate_of_turn_BRG_GYRO",
    "ME RPM_AMS", "Starboard_rudder_sensor_BRG_AUTOP",
    "Trim_AMS", "List_AMS",
]


def load_registry():
    """Load the local vessel registry (identifier -> (glob pattern(s), column map))."""
    try:
        from ship_registry import SHIP_CSV_GLOBS
    except ImportError as exc:
        raise SystemExit(
            "ship_registry.py not found. Copy ship_registry.example.py to "
            "ship_registry.py and fill in your vessels' raw-data paths."
        ) from exc
    return SHIP_CSV_GLOBS


def merge_and_rename(vessel: str, registry) -> Path:
    patterns, col_map = registry[vessel]
    if isinstance(patterns, str):
        patterns = [patterns]

    # Collect every raw export matching the vessel's glob pattern(s).
    all_files = []
    for pat in patterns:
        all_files.extend(sorted(glob.glob(pat)))

    if not all_files:
        raise FileNotFoundError(f"No CSV files found for {vessel}")

    print(f"Merging {len(all_files)} files for {vessel.upper()}...")
    for f in all_files:
        print(f"  {f}")

    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)

    # Concatenate the periodic export files into a single frame; the index is reset so
    # row numbers are contiguous before the time-sort below.
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after concat: {len(merged)}")

    # Rename only the columns that are actually present; the available sensors vary by
    # vessel and export vintage, so a strict rename would raise on absent keys.
    rename_map = {}
    for raw_col, target_col in col_map.items():
        if raw_col in merged.columns:
            rename_map[raw_col] = target_col
    merged.rename(columns=rename_map, inplace=True)

    # pandas appends ".1" suffixes when a concat produces duplicate column names (a column
    # renamed while the original is still present); drop those artifacts.
    dup_cols = [c for c in merged.columns if c.endswith('.1')]
    if dup_cols:
        merged.drop(columns=dup_cols, inplace=True)
        print(f"Dropped {len(dup_cols)} duplicate columns: {dup_cols}")

    # Reduce to the reference schema so every vessel produces a file with identical structure.
    keep_cols = [c for c in REFERENCE_SCHEMA_COLUMNS if c in merged.columns]
    extra = [c for c in merged.columns if c not in REFERENCE_SCHEMA_COLUMNS]
    if extra:
        print(f"Dropping {len(extra)} columns not in the reference schema.")
    merged = merged[keep_cols]

    # Coerce TIME to datetime (errors="coerce" turns unparseable strings into NaT), sort
    # chronologically, then discard rows whose timestamp is missing entirely.
    if "TIME" in merged.columns:
        merged["TIME"] = pd.to_datetime(merged["TIME"], errors="coerce")
        merged.sort_values("TIME", inplace=True)
        merged.dropna(subset=["TIME"], inplace=True)
        merged.reset_index(drop=True, inplace=True)

    # Remove rows that are exact duplicates across all columns, common at revision
    # boundaries where export windows overlap.
    before = len(merged)
    merged.drop_duplicates(inplace=True)
    after = len(merged)
    if before != after:
        print(f"Removed {before - after} duplicate rows.")

    out_path = Path(f"PhD/{vessel.upper()}_Synchronized_usable_data.xlsx")
    merged.to_excel(out_path, index=False)
    print(f"Saved: {out_path} ({len(merged)} rows, {len(merged.columns)} columns)")
    return out_path


if __name__ == "__main__":
    registry = load_registry()
    if len(sys.argv) < 2:
        print("Usage: python prepare_ship_data.py <vessel|all>")
        print(f"Available vessels: {', '.join(registry.keys())}")
        sys.exit(1)

    target = sys.argv[1].lower()
    if target == "all":
        for name in registry:
            merge_and_rename(name, registry)
    elif target in registry:
        merge_and_rename(target, registry)
    else:
        print(f"Unknown vessel: {target}. Available: {', '.join(registry.keys())}")
        sys.exit(1)
