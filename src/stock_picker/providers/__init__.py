"""Small provider layer for switching between mock and optional real data adapters."""

from stock_picker.providers.data_providers import (
    CsvFeatureProvider,
    CsvUniverseProvider,
    MockFeatureProvider,
    MockUniverseProvider,
    PassThroughFeatureProvider,
    PredefinedUniverseProvider,
    RealDataFeatureProvider,
    SnapshotFeatureProvider,
    StateUniverseProvider,
    YFinanceFeatureProvider,
    build_mock_universe_df,
    load_candidate_frame,
)
from stock_picker.providers.macro_providers import (
    FredMacroProvider,
    JsonMacroProvider,
    MockMacroProvider,
    StateMacroProvider,
    build_mock_macro_context,
    load_macro_context,
)

__all__ = [
    "CsvFeatureProvider",
    "CsvUniverseProvider",
    "MockFeatureProvider",
    "MockUniverseProvider",
    "PassThroughFeatureProvider",
    "PredefinedUniverseProvider",
    "RealDataFeatureProvider",
    "SnapshotFeatureProvider",
    "StateUniverseProvider",
    "YFinanceFeatureProvider",
    "FredMacroProvider",
    "JsonMacroProvider",
    "MockMacroProvider",
    "StateMacroProvider",
    "build_mock_macro_context",
    "build_mock_universe_df",
    "load_candidate_frame",
    "load_macro_context",
]
