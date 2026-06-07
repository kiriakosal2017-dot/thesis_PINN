"""Local vessel registry for prepare_ship_data.py.

Copy this file to ``ship_registry.py`` (which is not version-controlled) and fill in your
own vessels. Each entry maps a vessel identifier to:
  - one glob pattern, or a list of glob patterns, for its raw export files, and
  - the column map used to normalise that vessel's headers to the reference schema.

Pick the column map that matches the source: ``LAROS_COLUMN_MAP`` or its variant
``LAROS_COLUMN_MAP_V2`` for the onboard-logger exports, or ``build_metis_column_map(name)``
for Metis-platform exports (which embed the vessel name in each header).
"""
from prepare_ship_data import (
    LAROS_COLUMN_MAP,
    LAROS_COLUMN_MAP_V2,
    build_metis_column_map,
)

SHIP_CSV_GLOBS = {
    # "vessel_a": ("path/to/vessel_a/*.clean.csv", LAROS_COLUMN_MAP),
    # "vessel_b": (
    #     [
    #         "path/to/vessel_b/rev1/*.clean.csv",
    #         "path/to/vessel_b/rev2/*.clean.csv",
    #     ],
    #     LAROS_COLUMN_MAP_V2,
    # ),
    # "vessel_c": ("path/to/vessel_c/*.csv", build_metis_column_map("vessel_c")),
}
