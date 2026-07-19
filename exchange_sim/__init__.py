"""exchange-simulator — an agent-based market where microstructure emerges."""
from exchange_sim.agents import (
    Agent,
    FundamentalTrader,
    Liquidator,
    MarketMaker,
    MarketView,
    MomentumTrader,
    NoiseTrader,
    OrderIntent,
)
from exchange_sim.engine import Exchange, Tape, default_market, flash_crash
from exchange_sim.stats import (
    autocorr,
    excess_kurtosis,
    max_drawdown,
    stylized_facts,
    volatility_clustering,
)

__version__ = "0.1.0"
__all__ = [
    "Agent", "MarketMaker", "NoiseTrader", "MomentumTrader", "FundamentalTrader",
    "Liquidator", "MarketView", "OrderIntent",
    "Exchange", "Tape", "default_market", "flash_crash",
    "excess_kurtosis", "autocorr", "volatility_clustering", "max_drawdown",
    "stylized_facts",
    "__version__",
]
