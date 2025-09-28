from .pipeline import compute_all_flags
from .matching import run_all_strategies_with_templates

from .flags import (
    preprocess_projects,
    compute_project_flags_fast,
    compute_project_flags,   # legacy alias
)
from .contractor import compute_contractor_indicators
