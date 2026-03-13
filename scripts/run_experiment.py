import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from counternet.cf_explainer import VanillaCF, VAE_CF, DiverseCF
from counternet.pipeline import Experiment
from counternet.model import CounterNetModel
from counternet.utils import load_configs

experiment = Experiment(
    explainers=[CounterNetModel],
    m_configs=[load_configs(PROJECT_ROOT / "assets" / "configs" / "adult.json")],
)
experiment.run()
