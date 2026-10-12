"""Loading of persisted forecast artifacts."""

from functools import lru_cache

import pandas as pd
from fastapi import HTTPException

from backend.core.config import PREDICTIONS_PATH


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    """Load the saved forecast table once per process."""
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Forecast data is missing. Run the pipeline in README.md first.",
        )
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
