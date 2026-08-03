"""exchange-simulator — an agent-based market where microstructure, and the
failures that come with leverage, emerge from agent interaction."""
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
from exchange_sim.balance import (
    BalanceSheet,
    DefaultEvent,
    FundingNetwork,
    MarginCall,
    RiskLimits,
    StressReport,
)
from exchange_sim.engine import (
    Exchange,
    Tape,
    credit_crisis,
    default_market,
    flash_crash,
    systemic_market,
)
from exchange_sim.fragility import (
    FragilityReport,
    ShockOutcome,
    SystemState,
    leverage_sweep,
    measure_state,
    shock_response,
)
from exchange_sim.history import (
    CSVSource,
    DataUnavailable,
    Series,
    SP500Source,
    StooqSource,
    SyntheticEpisode,
    replay,
    replay_report,
)
from exchange_sim.institutions import (
    DealerBank,
    FundingDesk,
    HFTMarketMaker,
    LeveredHedgeFund,
    PensionFund,
    RetailTrader,
)
from exchange_sim.stats import (
    autocorr,
    excess_kurtosis,
    max_drawdown,
    stylized_facts,
    volatility_clustering,
)

__version__ = "0.2.0"
__all__ = [
    # agents
    "Agent", "MarketMaker", "NoiseTrader", "MomentumTrader", "FundamentalTrader",
    "Liquidator", "MarketView", "OrderIntent",
    # institutions
    "DealerBank", "HFTMarketMaker", "LeveredHedgeFund", "PensionFund",
    "RetailTrader", "FundingDesk",
    # balance sheets and credit
    "BalanceSheet", "RiskLimits", "FundingNetwork", "DefaultEvent", "MarginCall",
    "StressReport",
    # engine
    "Exchange", "Tape", "default_market", "systemic_market", "flash_crash",
    "credit_crisis",
    # fragility
    "SystemState", "ShockOutcome", "FragilityReport", "measure_state",
    "shock_response", "leverage_sweep",
    # history
    "Series", "StooqSource", "SP500Source", "CSVSource", "SyntheticEpisode",
    "DataUnavailable", "replay", "replay_report",
    # stats
    "excess_kurtosis", "autocorr", "volatility_clustering", "max_drawdown",
    "stylized_facts",
    "__version__",
]
