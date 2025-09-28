# dpwhlib/__init__.py
from .flags import (
    preprocess_projects,
    compute_project_flags_fast,
    compute_project_flags,  # legacy alias
)
from .contractor import compute_contractor_indicators
from .io import read_base_csv_from_path, save_csv_bytes
