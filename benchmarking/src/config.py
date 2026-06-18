from pathlib import Path

DEFAULT_WORKSPACE = Path("benchmarking/inputs/simple_workspace_nonp.json")
DEFAULT_TARGET = "L_ch0"
DEFAULT_MODE = "FAST_RUN"
DEFAULT_N_RUNS = 5

RESULTS_DIR = Path("benchmarking/results")
PLOTS_DIR = Path("benchmarking/plots")
REPORTS_DIR = Path("benchmarking/reports")

WORKSPACE_LABELS = {
    "simple_workspace_nonp": "Simple nonp",
    "simple_workspace": "Simple",
    "simple_workspace_generic_nonp": "Generic nonp",
    "simple_workspace_generic": "Generic",
}
