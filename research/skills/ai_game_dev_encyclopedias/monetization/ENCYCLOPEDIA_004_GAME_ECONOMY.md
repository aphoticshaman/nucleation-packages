# ENCYCLOPEDIA OF GAME ECONOMY AND MONETIZATION SYSTEMS
## Volume IV: Free-to-Play Economics, Virtual Currency Design, and Ethical Revenue Optimization
### Technical Reference for AI-Automated Game Development Pipelines
### Version 1.0 | November 2025

---

# PREFACE

This encyclopedia provides exhaustive technical documentation for designing, implementing, and optimizing game economy systems—specifically for free-to-play and premium-with-DLC models. Written at research-laboratory standards, it covers the mathematics of virtual economies, player psychology without exploitation, and production-ready implementation patterns.

The goal: enable AI systems to generate sustainable, ethical, and highly profitable monetization systems that respect players while generating the revenue needed for ongoing development.

**Ethical Framework**: This encyclopedia explicitly rejects predatory monetization (loot boxes targeting vulnerable populations, pay-to-win mechanics, FOMO-driven dark patterns). The focus is on value exchange: players pay for cosmetics, convenience, and content they genuinely want.

---

# TABLE OF CONTENTS

## PART I: ECONOMIC FOUNDATIONS
- Chapter 1: Virtual Economy Theory
- Chapter 2: Currency Systems Design
- Chapter 3: Price Psychology and Anchoring
- Chapter 4: Supply and Demand in Closed Systems

## PART II: PLAYER SEGMENTATION
- Chapter 5: Player Spending Archetypes
- Chapter 6: Engagement vs Monetization Metrics
- Chapter 7: Lifetime Value Prediction
- Chapter 8: Cohort Analysis and Retention

## PART III: MONETIZATION MODELS
- Chapter 9: Premium Purchase Design
- Chapter 10: Battle Pass Architecture
- Chapter 11: Cosmetic Shop Systems
- Chapter 12: Subscription Models

## PART IV: PROGRESSION ECONOMICS
- Chapter 13: XP and Leveling Curves
- Chapter 14: Unlock Systems and Gating
- Chapter 15: Prestige and Meta-Progression
- Chapter 16: Seasonal Content Rotation

## PART V: BALANCE AND FAIRNESS
- Chapter 17: Pay-to-Win Prevention
- Chapter 18: Free Player Experience Design
- Chapter 19: Competitive Integrity
- Chapter 20: Catch-Up Mechanics

## PART VI: IMPLEMENTATION SYSTEMS
- Chapter 21: Economy Simulation Tools
- Chapter 22: A/B Testing Infrastructure
- Chapter 23: Dynamic Pricing Systems
- Chapter 24: Anti-Fraud and Exploit Prevention

## PART VII: ANALYTICS AND OPTIMIZATION
- Chapter 25: Key Performance Indicators
- Chapter 26: Funnel Analysis
- Chapter 27: Revenue Forecasting
- Chapter 28: Churn Prediction and Prevention

## PART VIII: REGIONAL AND LEGAL
- Chapter 29: Regional Pricing Strategies
- Chapter 30: Legal Compliance (Loot Boxes, Age Ratings)
- Chapter 31: Refund and Support Systems
- Chapter 32: Platform Fee Optimization

---

# PART I: ECONOMIC FOUNDATIONS

---

## Chapter 1: Virtual Economy Theory

### 1.1 Fundamentals of Virtual Economies

```python
"""
VIRTUAL ECONOMY FUNDAMENTALS

A virtual economy is a closed economic system within a game where:
- Resources are created by game systems (not external production)
- Value is determined by utility within the game context
- Exchange occurs between players and/or with the game itself

Key differences from real economies:
- The "central bank" (developer) has perfect control over money supply
- Resources can be created from nothing (faucets)
- Resources can be destroyed completely (sinks)
- No physical scarcity constraints
- Perfect information possible (though often hidden)

DESIGN GOALS:
1. Sustainable engagement (players want to continue)
2. Fair value exchange (players feel purchases are worth it)
3. Revenue generation (business sustainability)
4. Competitive integrity (skill > money in competition)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class ResourceType(Enum):
    """Types of virtual resources."""
    SOFT_CURRENCY = "soft"      # Earned through play (gold, coins)
    HARD_CURRENCY = "hard"      # Purchased with real money (gems, crystals)
    ENERGY = "energy"           # Regenerating limiter (stamina)
    MATERIALS = "materials"     # Crafting/upgrade components
    COSMETIC = "cosmetic"       # Visual customization items
    FUNCTIONAL = "functional"   # Gameplay-affecting items
    PROGRESSION = "progression" # XP, levels, ranks


@dataclass
class Resource:
    """Definition of a virtual resource."""
    id: str
    name: str
    resource_type: ResourceType
    description: str
    
    # Economy parameters
    soft_cap: Optional[int] = None      # Held amount before diminishing returns
    hard_cap: Optional[int] = None      # Maximum holdable amount
    decay_rate: float = 0.0             # Per-day decay (0 = no decay)
    tradeable: bool = False             # Can players trade this?
    
    # Value anchoring
    real_money_value: Optional[float] = None  # USD equivalent if purchasable
    time_value_minutes: Optional[float] = None  # Time to earn equivalent


@dataclass
class Faucet:
    """Source of resource generation (input into economy)."""
    id: str
    name: str
    resource_id: str
    
    # Generation parameters
    amount_per_trigger: int
    trigger_type: str  # "time", "action", "achievement", "purchase"
    cooldown_seconds: float = 0
    daily_limit: Optional[int] = None
    lifetime_limit: Optional[int] = None
    
    # Conditions
    requires_level: int = 1
    requires_purchase: bool = False


@dataclass
class Sink:
    """Resource consumption point (output from economy)."""
    id: str
    name: str
    resource_id: str
    
    # Consumption parameters
    amount_per_use: int
    use_type: str  # "upgrade", "purchase", "repair", "entry_fee"
    
    # What you get
    output_type: str  # "item", "progress", "access", "cosmetic"
    output_value: float  # Relative value of output
    
    # Optionality
    is_required: bool = False  # True = mandatory progression
    has_free_alternative: bool = True  # Can achieve without spending


class VirtualEconomy:
    """Complete virtual economy simulation and management."""
    
    def __init__(self, name: str):
        self.name = name
        self.resources: Dict[str, Resource] = {}
        self.faucets: Dict[str, Faucet] = {}
        self.sinks: Dict[str, Sink] = {}
        
        # Player state tracking (for simulation)
        self.player_holdings: Dict[str, Dict[str, int]] = {}  # player_id -> resource_id -> amount
        
        # Economic metrics
        self.total_generated: Dict[str, int] = {}
        self.total_consumed: Dict[str, int] = {}
    
    def add_resource(self, resource: Resource):
        """Register a resource in the economy."""
        self.resources[resource.id] = resource
        self.total_generated[resource.id] = 0
        self.total_consumed[resource.id] = 0
    
    def add_faucet(self, faucet: Faucet):
        """Register a faucet (generation source)."""
        if faucet.resource_id not in self.resources:
            raise ValueError(f"Unknown resource: {faucet.resource_id}")
        self.faucets[faucet.id] = faucet
    
    def add_sink(self, sink: Sink):
        """Register a sink (consumption point)."""
        if sink.resource_id not in self.resources:
            raise ValueError(f"Unknown resource: {sink.resource_id}")
        self.sinks[sink.id] = sink
    
    def generate_resource(self, player_id: str, faucet_id: str, 
                         multiplier: float = 1.0) -> int:
        """Generate resources from a faucet."""
        faucet = self.faucets.get(faucet_id)
        if not faucet:
            return 0
        
        amount = int(faucet.amount_per_trigger * multiplier)
        resource = self.resources[faucet.resource_id]
        
        # Initialize player if needed
        if player_id not in self.player_holdings:
            self.player_holdings[player_id] = {}
        
        current = self.player_holdings[player_id].get(faucet.resource_id, 0)
        
        # Apply caps
        if resource.hard_cap:
            amount = min(amount, resource.hard_cap - current)
        
        if amount > 0:
            self.player_holdings[player_id][faucet.resource_id] = current + amount
            self.total_generated[faucet.resource_id] += amount
        
        return amount
    
    def consume_resource(self, player_id: str, sink_id: str) -> bool:
        """Attempt to consume resources at a sink."""
        sink = self.sinks.get(sink_id)
        if not sink:
            return False
        
        holdings = self.player_holdings.get(player_id, {})
        current = holdings.get(sink.resource_id, 0)
        
        if current < sink.amount_per_use:
            return False  # Insufficient funds
        
        self.player_holdings[player_id][sink.resource_id] = current - sink.amount_per_use
        self.total_consumed[sink.resource_id] += sink.amount_per_use
        
        return True
    
    def get_balance(self, player_id: str, resource_id: str) -> int:
        """Get player's current balance of a resource."""
        return self.player_holdings.get(player_id, {}).get(resource_id, 0)
    
    def economy_health_metrics(self) -> Dict[str, float]:
        """Calculate economy health indicators."""
        metrics = {}
        
        for resource_id in self.resources:
            generated = self.total_generated.get(resource_id, 0)
            consumed = self.total_consumed.get(resource_id, 0)
            
            # Sink ratio: consumed / generated
            # Healthy: 0.7 - 0.95 (some accumulation, but mostly spent)
            if generated > 0:
                sink_ratio = consumed / generated
            else:
                sink_ratio = 0
            
            metrics[f"{resource_id}_sink_ratio"] = sink_ratio
            metrics[f"{resource_id}_total_supply"] = generated - consumed
        
        return metrics
    
    def inflation_check(self, resource_id: str, 
                       historical_prices: List[float]) -> Dict[str, float]:
        """Check for inflation/deflation in a resource.
        
        Args:
            resource_id: Resource to analyze
            historical_prices: Time series of exchange rate (if tradeable)
        
        Returns:
            Inflation metrics
        """
        if len(historical_prices) < 2:
            return {"inflation_rate": 0.0, "trend": "stable"}
        
        prices = np.array(historical_prices)
        
        # Calculate period-over-period changes
        changes = np.diff(prices) / prices[:-1]
        
        avg_change = np.mean(changes)
        volatility = np.std(changes)
        
        # Trend detection
        if avg_change > 0.02:  # >2% average increase
            trend = "inflationary"
        elif avg_change < -0.02:
            trend = "deflationary"
        else:
            trend = "stable"
        
        return {
            "inflation_rate": avg_change,
            "volatility": volatility,
            "trend": trend,
            "latest_price": prices[-1],
            "price_change_30d": (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        }


class EconomySimulator:
    """Simulate economy over time to predict health."""
    
    def __init__(self, economy: VirtualEconomy):
        self.economy = economy
        self.rng = np.random.default_rng(42)
    
    def simulate_player_day(self, player_id: str, 
                           player_profile: Dict[str, float]) -> Dict[str, int]:
        """Simulate one day of player activity.
        
        Args:
            player_id: Player identifier
            player_profile: Activity levels (0-1) for different actions
        
        Returns:
            {resource_id: net_change} for the day
        """
        changes = {}
        
        # Process faucets
        for faucet_id, faucet in self.economy.faucets.items():
            # Probability of triggering based on profile
            trigger_prob = player_profile.get(faucet.trigger_type, 0.5)
            
            if self.rng.random() < trigger_prob:
                amount = self.economy.generate_resource(player_id, faucet_id)
                changes[faucet.resource_id] = changes.get(faucet.resource_id, 0) + amount
        
        # Process sinks
        for sink_id, sink in self.economy.sinks.items():
            # Probability of using based on profile and optionality
            if sink.is_required:
                use_prob = 0.9
            else:
                use_prob = player_profile.get("spending_propensity", 0.3)
            
            if self.rng.random() < use_prob:
                if self.economy.consume_resource(player_id, sink_id):
                    changes[sink.resource_id] = changes.get(sink.resource_id, 0) - sink.amount_per_use
        
        return changes
    
    def simulate_economy(self, num_players: int, num_days: int,
                        profile_distribution: Dict[str, Tuple[float, float]]
                       ) -> Dict[str, List[float]]:
        """Simulate entire economy over time.
        
        Args:
            num_players: Number of simulated players
            num_days: Days to simulate
            profile_distribution: {profile_key: (mean, std)} for generating profiles
        
        Returns:
            Time series of key metrics
        """
        # Generate player profiles
        profiles = {}
        for i in range(num_players):
            player_id = f"player_{i}"
            profiles[player_id] = {
                key: np.clip(self.rng.normal(mean, std), 0, 1)
                for key, (mean, std) in profile_distribution.items()
            }
        
        # Track metrics over time
        metrics_history = {
            "total_supply": [],
            "sink_ratio": [],
            "active_players": []
        }
        
        for day in range(num_days):
            daily_active = 0
            
            for player_id, profile in profiles.items():
                # Daily login probability
                if self.rng.random() < profile.get("daily_login", 0.5):
                    self.simulate_player_day(player_id, profile)
                    daily_active += 1
            
            # Record metrics
            health = self.economy.economy_health_metrics()
            
            total_supply = sum(
                health.get(f"{r}_total_supply", 0) 
                for r in self.economy.resources
            )
            
            avg_sink = np.mean([
                health.get(f"{r}_sink_ratio", 0) 
                for r in self.economy.resources
            ])
            
            metrics_history["total_supply"].append(total_supply)
            metrics_history["sink_ratio"].append(avg_sink)
            metrics_history["active_players"].append(daily_active)
        
        return metrics_history
```

### 1.2 Faucet-Sink Balance Theory

```python
"""
FAUCET-SINK BALANCE

The core principle of virtual economy design:
- Faucets add resources to the economy
- Sinks remove resources from the economy
- Balance determines inflation/deflation

HEALTHY ECONOMY INDICATORS:
- Sink ratio 0.7-0.95 (most resources eventually spent)
- No runaway accumulation (soft caps working)
- Clear value hierarchy (players understand what's valuable)
- Meaningful choices (multiple valid spending paths)

COMMON FAILURE MODES:
1. Hyperinflation: Too many faucets, not enough sinks
   - Resources become worthless
   - Prices in player-to-player trade skyrocket
   - New players can't catch up

2. Deflation: Too many sinks, not enough faucets
   - Resources too scarce
   - Players feel starved
   - Frustration drives churn

3. Stagnation: Balanced but boring
   - No meaningful economic decisions
   - Resources feel pointless
   - Players ignore economy entirely
"""

@dataclass
class FaucetSinkAnalysis:
    """Analyze faucet-sink balance."""
    
    resource_id: str
    faucet_rate_per_day: float  # Expected daily generation per player
    sink_rate_per_day: float    # Expected daily consumption per player
    
    # Derived metrics
    @property
    def daily_net_flow(self) -> float:
        """Net resource change per player per day."""
        return self.faucet_rate_per_day - self.sink_rate_per_day
    
    @property
    def sink_ratio(self) -> float:
        """Proportion of generated resources that are consumed."""
        if self.faucet_rate_per_day <= 0:
            return 0.0
        return self.sink_rate_per_day / self.faucet_rate_per_day
    
    @property
    def days_to_cap(self) -> Optional[float]:
        """Days until average player hits resource cap."""
        if self.daily_net_flow <= 0:
            return None  # Never caps
        # Would need cap value to calculate
        return None
    
    def health_assessment(self) -> Dict[str, any]:
        """Assess health of this resource's economy."""
        ratio = self.sink_ratio
        
        if ratio < 0.5:
            status = "CRITICAL_INFLATION"
            recommendation = "Add more sinks or reduce faucets"
        elif ratio < 0.7:
            status = "INFLATIONARY"
            recommendation = "Consider adding soft sinks"
        elif ratio < 0.85:
            status = "HEALTHY_GROWTH"
            recommendation = "Monitor for long-term accumulation"
        elif ratio < 0.95:
            status = "HEALTHY_EQUILIBRIUM"
            recommendation = "Ideal balance, maintain current"
        elif ratio < 1.0:
            status = "TIGHT"
            recommendation = "May feel scarce to players"
        else:
            status = "DEFLATIONARY"
            recommendation = "Resources draining, add faucets"
        
        return {
            "status": status,
            "sink_ratio": ratio,
            "daily_net_flow": self.daily_net_flow,
            "recommendation": recommendation
        }


class FaucetDesigner:
    """Design and validate faucet systems."""
    
    @staticmethod
    def daily_login_faucet(resource_id: str, base_amount: int,
                          streak_bonus: float = 0.1,
                          max_streak_days: int = 7) -> Dict:
        """Design daily login reward faucet.
        
        Args:
            resource_id: Resource to give
            base_amount: Day 1 amount
            streak_bonus: Per-day increase (0.1 = 10% more each day)
            max_streak_days: Cap on streak bonuses
        
        Returns:
            Faucet configuration
        """
        # Calculate expected daily output
        # Assume 60% daily return rate, streak builds over time
        avg_streak = 3.5  # Typical for 60% return rate
        avg_multiplier = 1 + (streak_bonus * min(avg_streak, max_streak_days))
        expected_daily = base_amount * avg_multiplier * 0.6  # 60% login
        
        return {
            "type": "daily_login",
            "resource_id": resource_id,
            "base_amount": base_amount,
            "streak_bonus": streak_bonus,
            "max_streak": max_streak_days,
            "expected_daily_per_player": expected_daily,
            "schedule": [
                {"day": i, "amount": int(base_amount * (1 + streak_bonus * min(i, max_streak_days)))}
                for i in range(max_streak_days + 1)
            ]
        }
    
    @staticmethod
    def gameplay_faucet(resource_id: str, amount_per_action: int,
                       actions_per_hour: float, hours_per_day: float = 2.0) -> Dict:
        """Design gameplay-earned faucet.
        
        Args:
            resource_id: Resource to give
            amount_per_action: Resource per completed action
            actions_per_hour: Expected actions per hour of play
            hours_per_day: Average play time per day
        
        Returns:
            Faucet configuration with daily projections
        """
        daily_actions = actions_per_hour * hours_per_day
        expected_daily = amount_per_action * daily_actions
        
        return {
            "type": "gameplay",
            "resource_id": resource_id,
            "amount_per_action": amount_per_action,
            "expected_actions_per_hour": actions_per_hour,
            "assumed_hours_per_day": hours_per_day,
            "expected_daily_per_player": expected_daily
        }
    
    @staticmethod
    def achievement_faucet(resource_id: str, 
                          achievements: List[Dict[str, any]]) -> Dict:
        """Design achievement-based faucet.
        
        Args:
            resource_id: Resource to give
            achievements: List of {name, amount, expected_completion_day}
        
        Returns:
            Faucet configuration with timeline
        """
        total_one_time = sum(a["amount"] for a in achievements)
        
        # Group by expected completion time
        by_day = {}
        for a in achievements:
            day = a.get("expected_completion_day", 1)
            by_day[day] = by_day.get(day, 0) + a["amount"]
        
        return {
            "type": "achievement",
            "resource_id": resource_id,
            "total_one_time_rewards": total_one_time,
            "achievements": achievements,
            "distribution_by_day": by_day,
            "note": "One-time rewards, not recurring"
        }


class SinkDesigner:
    """Design and validate sink systems."""
    
    @staticmethod
    def upgrade_sink(resource_id: str, 
                    upgrade_costs: List[int],
                    expected_upgrades_per_day: float = 0.5) -> Dict:
        """Design upgrade cost sink.
        
        Args:
            resource_id: Resource consumed
            upgrade_costs: Cost per level [level1, level2, ...]
            expected_upgrades_per_day: How often players upgrade
        
        Returns:
            Sink configuration
        """
        avg_cost = np.mean(upgrade_costs)
        expected_daily = avg_cost * expected_upgrades_per_day
        
        return {
            "type": "upgrade",
            "resource_id": resource_id,
            "costs_by_level": upgrade_costs,
            "total_to_max": sum(upgrade_costs),
            "average_cost": avg_cost,
            "expected_daily_sink": expected_daily
        }
    
    @staticmethod
    def entry_fee_sink(resource_id: str, fee_amount: int,
                      entries_per_day: float = 3.0) -> Dict:
        """Design entry fee sink (for modes, dungeons, etc.).
        
        Args:
            resource_id: Resource consumed
            fee_amount: Cost per entry
            entries_per_day: Expected daily entries
        
        Returns:
            Sink configuration
        """
        return {
            "type": "entry_fee",
            "resource_id": resource_id,
            "fee_amount": fee_amount,
            "expected_entries_per_day": entries_per_day,
            "expected_daily_sink": fee_amount * entries_per_day
        }
    
    @staticmethod
    def cosmetic_shop_sink(resource_id: str,
                          items: List[Dict[str, any]],
                          purchase_rate: float = 0.1) -> Dict:
        """Design cosmetic shop sink.
        
        Args:
            resource_id: Resource used for purchases
            items: List of {name, cost, appeal_score}
            purchase_rate: Probability of purchase per item shown per day
        
        Returns:
            Sink configuration
        """
        weighted_avg_cost = np.average(
            [i["cost"] for i in items],
            weights=[i.get("appeal_score", 1) for i in items]
        )
        
        # Expected purchases per day
        expected_purchases = len(items) * purchase_rate * 0.1  # Assume 10% see each item
        expected_daily = weighted_avg_cost * expected_purchases
        
        return {
            "type": "cosmetic_shop",
            "resource_id": resource_id,
            "items": items,
            "weighted_average_cost": weighted_avg_cost,
            "expected_daily_sink": expected_daily,
            "note": "Voluntary sink - player chooses to spend"
        }


def design_balanced_economy(
    soft_currency_name: str = "Gold",
    target_daily_earn_minutes: float = 30.0,
    target_save_days_for_premium: int = 7
) -> VirtualEconomy:
    """Design a balanced soft currency economy.
    
    Args:
        soft_currency_name: Name of soft currency
        target_daily_earn_minutes: Minutes of play to earn daily target
        target_save_days_for_premium: Days to save for premium item
    
    Returns:
        Configured VirtualEconomy
    """
    economy = VirtualEconomy(f"{soft_currency_name}_economy")
    
    # Calculate target daily earn
    # Premium item should cost: daily_earn * save_days
    # Daily earn = enough to buy standard item daily + save toward premium
    
    standard_item_cost = 100  # Baseline
    daily_target = standard_item_cost * 1.3  # 30% surplus for premium saving
    premium_cost = daily_target * target_save_days_for_premium
    
    # Define soft currency
    currency = Resource(
        id="gold",
        name=soft_currency_name,
        resource_type=ResourceType.SOFT_CURRENCY,
        description="Primary soft currency earned through play",
        soft_cap=int(premium_cost * 2),  # Can hold 2 premium items worth
        hard_cap=int(premium_cost * 5),   # Hard cap at 5
        time_value_minutes=target_daily_earn_minutes / daily_target
    )
    economy.add_resource(currency)
    
    # Design faucets to hit daily target
    faucet_designer = FaucetDesigner()
    
    # 40% from gameplay
    gameplay_faucet = Faucet(
        id="gameplay_gold",
        name="Match Rewards",
        resource_id="gold",
        amount_per_trigger=int(daily_target * 0.4 / 10),  # 10 matches
        trigger_type="action",
        daily_limit=int(daily_target * 0.5)  # Soft cap with buffer
    )
    economy.add_faucet(gameplay_faucet)
    
    # 30% from daily login
    login_faucet = Faucet(
        id="daily_login",
        name="Daily Bonus",
        resource_id="gold",
        amount_per_trigger=int(daily_target * 0.3),
        trigger_type="time",
        cooldown_seconds=86400  # 24 hours
    )
    economy.add_faucet(login_faucet)
    
    # 30% from achievements/quests (one-time and daily)
    quest_faucet = Faucet(
        id="daily_quest",
        name="Daily Quest",
        resource_id="gold",
        amount_per_trigger=int(daily_target * 0.3),
        trigger_type="achievement",
        daily_limit=1
    )
    economy.add_faucet(quest_faucet)
    
    # Design sinks
    # Standard shop (daily purchases)
    standard_sink = Sink(
        id="standard_shop",
        name="Card Pack",
        resource_id="gold",
        amount_per_use=standard_item_cost,
        use_type="purchase",
        output_type="item",
        output_value=1.0,
        is_required=False,
        has_free_alternative=True
    )
    economy.add_sink(standard_sink)
    
    # Premium shop (weekly target)
    premium_sink = Sink(
        id="premium_shop",
        name="Premium Card Pack",
        resource_id="gold",
        amount_per_use=int(premium_cost),
        use_type="purchase",
        output_type="item",
        output_value=3.0,
        is_required=False,
        has_free_alternative=True
    )
    economy.add_sink(premium_sink)
    
    # Entry fee sink (optional content)
    entry_sink = Sink(
        id="special_mode",
        name="Special Event Entry",
        resource_id="gold",
        amount_per_use=int(daily_target * 0.2),
        use_type="entry_fee",
        output_type="access",
        output_value=0.5,
        is_required=False,
        has_free_alternative=True  # Limited free entries
    )
    economy.add_sink(entry_sink)
    
    return economy
```

---

## Chapter 2: Currency Systems Design

### 2.1 Dual Currency Architecture

```python
"""
DUAL CURRENCY SYSTEMS

Standard F2P pattern:
- Soft currency: Earned through play (abundant, grindable)
- Hard currency: Purchased with real money (scarce, premium)

WHY TWO CURRENCIES?

1. Price Anchoring
   - Soft currency sets "fair" value through play time
   - Hard currency enables "pay to skip grind"
   - Players mentally compare real money to time value

2. Monetization Clarity
   - Soft: Free player experience
   - Hard: Revenue generation
   - Clear separation prevents P2W perception

3. Economic Control
   - Soft inflation doesn't affect real money value
   - Can give bonus soft currency freely
   - Hard currency maintains stable value

4. Conversion Psychology
   - Hard → Soft conversion option
   - Creates "value" perception for hard currency
   - Never allow Soft → Hard (breaks monetization)
"""

@dataclass
class CurrencyTier:
    """Configuration for a currency tier."""
    id: str
    name: str
    symbol: str
    is_premium: bool
    
    # Earning
    can_earn_free: bool
    free_earn_rate_per_hour: float  # Units per hour of play
    
    # Purchasing
    can_purchase: bool
    base_price_per_100: float  # USD per 100 units
    bonus_tiers: Dict[int, float] = field(default_factory=dict)  # {spend_usd: bonus_percent}
    
    # Conversion
    converts_to: Optional[str] = None  # Other currency ID
    conversion_rate: float = 0.0  # Units of other per 1 of this


class DualCurrencySystem:
    """Manage dual currency economy."""
    
    def __init__(self):
        self.soft_currency = CurrencyTier(
            id="gold",
            name="Gold",
            symbol="🪙",
            is_premium=False,
            can_earn_free=True,
            free_earn_rate_per_hour=150.0,  # 150 gold per hour
            can_purchase=False,  # Cannot buy soft directly
            base_price_per_100=0.0,
            converts_to=None
        )
        
        self.hard_currency = CurrencyTier(
            id="gems",
            name="Gems",
            symbol="💎",
            is_premium=True,
            can_earn_free=True,  # Small amount from achievements
            free_earn_rate_per_hour=5.0,  # ~5 gems per hour (slow)
            can_purchase=True,
            base_price_per_100=0.99,  # $0.99 per 100 gems
            bonus_tiers={
                4.99: 0.10,   # 10% bonus at $5
                9.99: 0.15,   # 15% bonus at $10
                24.99: 0.25,  # 25% bonus at $25
                49.99: 0.40,  # 40% bonus at $50
                99.99: 0.60   # 60% bonus at $100
            },
            converts_to="gold",
            conversion_rate=10.0  # 1 gem = 10 gold
        )
        
        # Price catalog
        self.price_catalog: Dict[str, Dict[str, int]] = {}
    
    def add_item(self, item_id: str, gold_price: int = None, gem_price: int = None):
        """Add item to price catalog.
        
        Items can be purchasable with either currency (different prices)
        or exclusively with one.
        """
        self.price_catalog[item_id] = {}
        
        if gold_price is not None:
            self.price_catalog[item_id]["gold"] = gold_price
        
        if gem_price is not None:
            self.price_catalog[item_id]["gems"] = gem_price
    
    def calculate_gem_package_value(self, usd_price: float) -> Dict[str, any]:
        """Calculate gems received for a purchase.
        
        Args:
            usd_price: Purchase amount in USD
        
        Returns:
            Package details including bonus
        """
        base_gems = int((usd_price / self.hard_currency.base_price_per_100) * 100)
        
        # Find applicable bonus tier
        bonus_percent = 0.0
        for threshold, bonus in sorted(self.hard_currency.bonus_tiers.items()):
            if usd_price >= threshold:
                bonus_percent = bonus
        
        bonus_gems = int(base_gems * bonus_percent)
        total_gems = base_gems + bonus_gems
        
        # Calculate value in gold equivalent
        gold_equivalent = total_gems * self.hard_currency.conversion_rate
        
        # Calculate time-to-earn equivalent
        hours_equivalent = gold_equivalent / self.soft_currency.free_earn_rate_per_hour
        
        return {
            "usd_price": usd_price,
            "base_gems": base_gems,
            "bonus_percent": bonus_percent,
            "bonus_gems": bonus_gems,
            "total_gems": total_gems,
            "gold_equivalent": gold_equivalent,
            "hours_to_earn_equivalent": hours_equivalent,
            "value_per_dollar": total_gems / usd_price
        }
    
    def design_gem_packages(self) -> List[Dict[str, any]]:
        """Design standard gem purchase packages."""
        price_points = [0.99, 4.99, 9.99, 24.99, 49.99, 99.99]
        
        packages = []
        for price in price_points:
            package = self.calculate_gem_package_value(price)
            
            # Add marketing labels
            if price == price_points[0]:
                package["label"] = "Starter"
            elif price == 4.99:
                package["label"] = "Popular"
            elif price == 9.99:
                package["label"] = "Best Value"  # Often the sweet spot
            elif price == 24.99:
                package["label"] = "Value Pack"
            elif price == 49.99:
                package["label"] = "Premium"
            else:
                package["label"] = "Ultimate"
            
            packages.append(package)
        
        return packages
    
    def item_value_analysis(self, item_id: str) -> Dict[str, any]:
        """Analyze value proposition of an item."""
        if item_id not in self.price_catalog:
            return {"error": "Item not found"}
        
        prices = self.price_catalog[item_id]
        analysis = {"item_id": item_id}
        
        if "gold" in prices:
            gold_price = prices["gold"]
            hours_to_earn = gold_price / self.soft_currency.free_earn_rate_per_hour
            analysis["gold_price"] = gold_price
            analysis["hours_to_earn_gold"] = hours_to_earn
        
        if "gems" in prices:
            gem_price = prices["gems"]
            # Calculate real money equivalent (at base rate)
            usd_equivalent = gem_price * self.hard_currency.base_price_per_100 / 100
            analysis["gem_price"] = gem_price
            analysis["usd_equivalent"] = usd_equivalent
            
            # Calculate hours if earned free
            hours_free = gem_price / self.hard_currency.free_earn_rate_per_hour
            analysis["hours_to_earn_gems_free"] = hours_free
        
        # Compare currencies if both available
        if "gold" in prices and "gems" in prices:
            # What's the implied conversion rate?
            implied_rate = prices["gold"] / prices["gems"]
            official_rate = self.hard_currency.conversion_rate
            
            if implied_rate > official_rate * 1.1:
                analysis["better_currency"] = "gems"
                analysis["savings_percent"] = (implied_rate - official_rate) / implied_rate
            elif implied_rate < official_rate * 0.9:
                analysis["better_currency"] = "gold"
                analysis["savings_percent"] = (official_rate - implied_rate) / official_rate
            else:
                analysis["better_currency"] = "equivalent"
                analysis["savings_percent"] = 0
        
        return analysis


class CurrencyPriceDesigner:
    """Design item prices across currency tiers."""
    
    def __init__(self, soft_rate: float, hard_rate: float, 
                 hard_to_soft_rate: float):
        """
        Args:
            soft_rate: Soft currency earned per hour
            hard_rate: Hard currency earned per hour (free)
            hard_to_soft_rate: Conversion rate (hard -> soft)
        """
        self.soft_rate = soft_rate
        self.hard_rate = hard_rate
        self.conversion = hard_to_soft_rate
    
    def price_by_time_value(self, hours_of_play: float,
                           premium_multiplier: float = 1.0) -> Dict[str, int]:
        """Calculate prices based on time-to-earn.
        
        Args:
            hours_of_play: How many hours of play should earn this
            premium_multiplier: >1 for premium items that should feel special
        
        Returns:
            {currency: price} for each currency
        """
        base_soft = self.soft_rate * hours_of_play * premium_multiplier
        base_hard = self.hard_rate * hours_of_play * premium_multiplier
        
        return {
            "soft": int(base_soft),
            "hard": int(base_hard),
            # Also include "either/or" hybrid option
            "soft_only": int(base_soft * 1.2),  # 20% premium for soft-only
            "hard_only": int(base_hard * 0.9)   # 10% discount for hard
        }
    
    def design_rarity_pricing(self, 
                             rarities: List[str] = ["common", "uncommon", "rare", "epic", "legendary"]
                            ) -> Dict[str, Dict[str, int]]:
        """Design prices scaling with rarity.
        
        Each rarity tier approximately doubles in value.
        """
        prices = {}
        
        base_hours = 0.25  # 15 minutes of play for common
        
        for i, rarity in enumerate(rarities):
            multiplier = 2 ** i  # 1, 2, 4, 8, 16
            hours = base_hours * multiplier
            
            prices[rarity] = self.price_by_time_value(hours, premium_multiplier=1.0)
        
        return prices
```

---

## Chapter 3: Price Psychology and Anchoring

### 3.1 Psychological Pricing Principles

```python
"""
PRICE PSYCHOLOGY IN GAMES

Players don't evaluate prices rationally.
Understanding cognitive biases enables ethical pricing that feels fair.

KEY PRINCIPLES:

1. ANCHORING
   First price seen becomes reference point.
   Show high-value items first to anchor expectations.

2. DECOY EFFECT
   Add option that makes target option look better.
   Medium pack looks great when small pack is bad value.

3. CHARM PRICING
   $9.99 feels significantly less than $10.00.
   Left-digit effect is powerful.

4. BUNDLE PSYCHOLOGY
   Bundles feel like deals even when fairly priced.
   "Get 5 items for price of 4!"

5. LOSS AVERSION
   People hate losing more than they like gaining.
   Limited time offers trigger fear of missing out.
   USE SPARINGLY - this is the dark pattern zone.

ETHICAL BOUNDARIES:
- Don't exploit vulnerable populations
- Clear communication of what you get
- No hidden costs or subscriptions
- Real value for money spent
- Free path to all non-cosmetic content
"""

from enum import Enum

class PriceEndingStrategy(Enum):
    """Common price ending strategies."""
    CHARM = "charm"           # $X.99 - feels cheaper
    ROUND = "round"          # $X.00 - feels premium/quality
    HALF = "half"            # $X.50 - compromise
    NINETY_FIVE = "95"       # $X.95 - slightly more premium than .99


class PricingPsychology:
    """Apply psychological pricing principles."""
    
    @staticmethod
    def apply_charm_pricing(base_price: float, 
                           strategy: PriceEndingStrategy = PriceEndingStrategy.CHARM
                          ) -> float:
        """Apply psychological price ending.
        
        Args:
            base_price: Raw calculated price
            strategy: Ending strategy to apply
        
        Returns:
            Psychologically optimized price
        """
        dollar_part = int(base_price)
        
        if strategy == PriceEndingStrategy.CHARM:
            return dollar_part + 0.99 if base_price > dollar_part else max(0.99, dollar_part - 0.01)
        elif strategy == PriceEndingStrategy.ROUND:
            return round(base_price)
        elif strategy == PriceEndingStrategy.HALF:
            return dollar_part + 0.50
        elif strategy == PriceEndingStrategy.NINETY_FIVE:
            return dollar_part + 0.95
        
        return base_price
    
    @staticmethod
    def design_price_ladder(min_price: float, max_price: float, 
                           num_tiers: int = 5) -> List[float]:
        """Design price ladder with psychological spacing.
        
        Prices should feel meaningfully different but not arbitrary.
        
        Args:
            min_price: Lowest price point
            max_price: Highest price point
            num_tiers: Number of price tiers
        
        Returns:
            List of psychologically spaced prices
        """
        # Use logarithmic spacing (feels more linear psychologically)
        import math
        
        log_min = math.log(min_price)
        log_max = math.log(max_price)
        log_step = (log_max - log_min) / (num_tiers - 1)
        
        prices = []
        for i in range(num_tiers):
            raw_price = math.exp(log_min + i * log_step)
            # Snap to standard price points
            prices.append(PricingPsychology._snap_to_standard(raw_price))
        
        return prices
    
    @staticmethod
    def _snap_to_standard(price: float) -> float:
        """Snap price to standard game price point."""
        standard_points = [0.99, 1.99, 2.99, 4.99, 6.99, 9.99, 
                         14.99, 19.99, 24.99, 29.99, 39.99, 
                         49.99, 59.99, 79.99, 99.99]
        
        # Find closest standard point
        closest = min(standard_points, key=lambda x: abs(x - price))
        return closest
    
    @staticmethod
    def design_decoy_pricing(target_package: Dict[str, any],
                            num_options: int = 3) -> List[Dict[str, any]]:
        """Design decoy pricing to make target option attractive.
        
        Args:
            target_package: The package we want to promote
                {price, value, items, ...}
            num_options: Total options to present (including target)
        
        Returns:
            List of packages including decoy
        """
        target_price = target_package["price"]
        target_value = target_package["value"]
        value_per_dollar = target_value / target_price
        
        packages = []
        
        # Option 1: Small package (poor value - makes target look good)
        small_price = target_price * 0.4
        small_value = target_value * 0.25  # Only 25% value at 40% price
        packages.append({
            "label": "Starter",
            "price": PricingPsychology._snap_to_standard(small_price),
            "value": small_value,
            "value_per_dollar": small_value / small_price,
            "role": "decoy_small"
        })
        
        # Option 2: Target (good value)
        packages.append({
            **target_package,
            "label": "Best Value",  # Highlight this
            "role": "target"
        })
        
        if num_options >= 3:
            # Option 3: Large package (good value but high commitment)
            large_price = target_price * 2.5
            large_value = target_value * 3.0  # Better per-dollar but more total
            packages.append({
                "label": "Premium",
                "price": PricingPsychology._snap_to_standard(large_price),
                "value": large_value,
                "value_per_dollar": large_value / large_price,
                "role": "upsell"
            })
        
        return packages
    
    @staticmethod
    def anchor_price_display(items: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Order items to create anchoring effect.
        
        Show highest value items first to anchor expectations high.
        Then mid-tier seems reasonable by comparison.
        
        Args:
            items: List of {name, price, value, ...}
        
        Returns:
            Reordered list with anchoring metadata
        """
        # Sort by price descending (highest first)
        sorted_items = sorted(items, key=lambda x: x["price"], reverse=True)
        
        # Add display metadata
        for i, item in enumerate(sorted_items):
            item["display_order"] = i
            if i == 0:
                item["anchor_role"] = "anchor_high"
            elif i == len(sorted_items) - 1:
                item["anchor_role"] = "anchor_low"
            else:
                item["anchor_role"] = "target_zone"
        
        return sorted_items


class BundleDesigner:
    """Design bundles that feel like great value."""
    
    @staticmethod
    def calculate_bundle_value(items: List[Dict[str, any]],
                              discount_percent: float = 0.20) -> Dict[str, any]:
        """Calculate bundle pricing.
        
        Args:
            items: List of {name, individual_price, ...}
            discount_percent: Discount from sum of individual prices
        
        Returns:
            Bundle configuration
        """
        total_individual = sum(item["individual_price"] for item in items)
        bundle_price = total_individual * (1 - discount_percent)
        savings = total_individual - bundle_price
        
        return {
            "items": items,
            "individual_total": total_individual,
            "bundle_price": PricingPsychology._snap_to_standard(bundle_price),
            "savings": savings,
            "savings_percent": discount_percent,
            "item_count": len(items),
            "marketing_copy": f"Save {int(discount_percent * 100)}%! Get {len(items)} items for the price of {len(items) - 1}!"
        }
    
    @staticmethod
    def design_starter_bundle(game_type: str = "deckbuilder") -> Dict[str, any]:
        """Design compelling new player bundle.
        
        Starter bundles should:
        - Provide excellent value
        - Include exclusive cosmetic (FOMO lite)
        - Accelerate early progression
        - Not provide competitive advantage
        """
        if game_type == "deckbuilder":
            items = [
                {"name": "Exclusive Card Back", "individual_price": 4.99, "type": "cosmetic"},
                {"name": "5 Card Packs", "individual_price": 7.50, "type": "content"},
                {"name": "500 Gold", "individual_price": 0.00, "type": "currency"},  # Bonus
                {"name": "Starter Deck Sleeve", "individual_price": 2.99, "type": "cosmetic"}
            ]
            
            # Price at $9.99 despite ~$15 "value"
            return {
                "name": "Starter Bundle",
                "items": items,
                "price": 9.99,
                "individual_value": sum(i["individual_price"] for i in items),
                "discount_percent": 0.35,
                "availability": "First 7 days only",
                "limit": 1,  # One per account
                "marketing": "Begin your journey with style!"
            }
        
        return {}
```

---

## Chapter 4: Supply and Demand in Closed Systems

### 4.1 Artificial Scarcity Design

```python
"""
ARTIFICIAL SCARCITY IN GAMES

Unlike real economies, games can control supply perfectly.
Scarcity creates value, but too much frustrates players.

SCARCITY MECHANISMS:

1. TIME-LIMITED
   - Seasonal items
   - Battle pass rewards
   - Event exclusives
   - Creates urgency without paywall

2. QUANTITY-LIMITED  
   - "Only 1000 available"
   - Numbered editions
   - Works for premium cosmetics
   - DANGER: Can feel manipulative

3. PROGRESSION-GATED
   - Locked behind achievements
   - Requires skill/time investment
   - Most ethical form of scarcity
   - Creates prestige

4. RANDOM DROP
   - Low probability from packs/boxes
   - DANGER ZONE: This is gambling
   - Many jurisdictions regulate this
   - Consider alternatives

ETHICAL GUIDELINES:
- All gameplay content achievable free
- Scarcity only for cosmetics
- Clear communication of availability
- No pressure tactics on children
- Odds disclosure for random elements
"""

from datetime import datetime, timedelta

@dataclass
class ScarcityConfig:
    """Configuration for item scarcity."""
    scarcity_type: str  # "time", "quantity", "progression", "random"
    
    # Time-limited
    available_from: Optional[datetime] = None
    available_until: Optional[datetime] = None
    recurring: bool = False  # Comes back next year?
    
    # Quantity-limited
    max_quantity: Optional[int] = None
    quantity_sold: int = 0
    
    # Progression-gated
    required_achievement: Optional[str] = None
    required_level: Optional[int] = None
    
    # Random (USE WITH CAUTION)
    drop_rate: Optional[float] = None
    pity_system: bool = True  # Guarantee after N tries
    pity_threshold: int = 100


class ScarcityManager:
    """Manage artificial scarcity ethically."""
    
    def __init__(self):
        self.items: Dict[str, ScarcityConfig] = {}
        self.player_pity_counters: Dict[str, Dict[str, int]] = {}
    
    def register_item(self, item_id: str, config: ScarcityConfig):
        """Register item with scarcity configuration."""
        self.items[item_id] = config
    
    def is_available(self, item_id: str, player_id: str = None) -> Tuple[bool, str]:
        """Check if item is currently available.
        
        Args:
            item_id: Item to check
            player_id: Player checking (for progression gates)
        
        Returns:
            (is_available, reason_if_not)
        """
        config = self.items.get(item_id)
        if not config:
            return False, "Item not found"
        
        now = datetime.now()
        
        # Time check
        if config.available_from and now < config.available_from:
            return False, f"Available starting {config.available_from}"
        
        if config.available_until and now > config.available_until:
            if config.recurring:
                return False, "Returns next season"
            return False, "No longer available"
        
        # Quantity check
        if config.max_quantity and config.quantity_sold >= config.max_quantity:
            return False, "Sold out"
        
        # Progression check (would need player data)
        if config.required_level:
            # player_level = get_player_level(player_id)
            # if player_level < config.required_level:
            #     return False, f"Requires level {config.required_level}"
            pass
        
        return True, "Available"
    
    def calculate_random_drop(self, item_id: str, player_id: str) -> Tuple[bool, Dict]:
        """Calculate random drop with pity system.
        
        IMPORTANT: This implements a pity system to prevent
        extreme bad luck. Players are guaranteed the item
        after a maximum number of attempts.
        
        Args:
            item_id: Item being rolled for
            player_id: Player rolling
        
        Returns:
            (dropped, details)
        """
        config = self.items.get(item_id)
        if not config or not config.drop_rate:
            return False, {"error": "Not a random drop item"}
        
        # Get pity counter
        if player_id not in self.player_pity_counters:
            self.player_pity_counters[player_id] = {}
        
        pity_count = self.player_pity_counters[player_id].get(item_id, 0)
        
        # Check pity threshold first
        if config.pity_system and pity_count >= config.pity_threshold:
            self.player_pity_counters[player_id][item_id] = 0
            return True, {
                "method": "pity",
                "attempts": pity_count + 1,
                "message": "Pity system activated!"
            }
        
        # Roll with increasing probability (soft pity)
        base_rate = config.drop_rate
        if config.pity_system:
            # Increase rate as pity counter grows
            # Soft pity starts at 75% of threshold
            soft_pity_start = int(config.pity_threshold * 0.75)
            if pity_count >= soft_pity_start:
                # Linear increase to 100% at threshold
                progress = (pity_count - soft_pity_start) / (config.pity_threshold - soft_pity_start)
                adjusted_rate = base_rate + (1 - base_rate) * progress
            else:
                adjusted_rate = base_rate
        else:
            adjusted_rate = base_rate
        
        # Roll
        import random
        rolled = random.random() < adjusted_rate
        
        if rolled:
            self.player_pity_counters[player_id][item_id] = 0
            return True, {
                "method": "random",
                "rate": adjusted_rate,
                "attempts": pity_count + 1
            }
        else:
            self.player_pity_counters[player_id][item_id] = pity_count + 1
            return False, {
                "pity_progress": pity_count + 1,
                "pity_threshold": config.pity_threshold,
                "next_rate": adjusted_rate
            }
    
    def design_seasonal_calendar(self, year: int) -> List[Dict]:
        """Design seasonal content calendar.
        
        Seasons provide natural scarcity cycles without pressure.
        """
        seasons = [
            {
                "name": "Spring Awakening",
                "start": datetime(year, 3, 20),
                "end": datetime(year, 6, 20),
                "theme": "renewal",
                "exclusive_items": ["spring_card_back", "blossom_effects"],
                "returns_next_year": True
            },
            {
                "name": "Summer Solstice",
                "start": datetime(year, 6, 21),
                "end": datetime(year, 9, 21),
                "theme": "celebration",
                "exclusive_items": ["summer_card_back", "sun_effects"],
                "returns_next_year": True
            },
            {
                "name": "Autumn Harvest",
                "start": datetime(year, 9, 22),
                "end": datetime(year, 12, 20),
                "theme": "abundance",
                "exclusive_items": ["autumn_card_back", "leaf_effects"],
                "returns_next_year": True
            },
            {
                "name": "Winter's Rest",
                "start": datetime(year, 12, 21),
                "end": datetime(year + 1, 3, 19),
                "theme": "reflection",
                "exclusive_items": ["winter_card_back", "frost_effects"],
                "returns_next_year": True
            }
        ]
        
        return seasons


class PrestigeSystem:
    """Design prestige items earned through skill/dedication.
    
    The most ethical form of scarcity: earn through play.
    """
    
    @staticmethod
    def design_ranked_rewards(ranks: List[str]) -> Dict[str, Dict]:
        """Design ranked season rewards.
        
        Higher ranks get better cosmetics.
        Everyone can earn something.
        """
        rewards = {}
        
        for i, rank in enumerate(ranks):
            tier = i + 1
            rewards[rank] = {
                "card_back": f"{rank.lower()}_season_back" if tier >= 2 else None,
                "card_sleeve": f"{rank.lower()}_sleeve" if tier >= 4 else None,
                "profile_frame": f"{rank.lower()}_frame" if tier >= 3 else None,
                "title": f"{rank} Warrior" if tier >= 5 else None,
                "currency_bonus": tier * 500,  # Scaling currency
                "exclusive": tier >= len(ranks) - 1  # Top 2 ranks get exclusive
            }
        
        return rewards
    
    @staticmethod
    def design_achievement_cosmetics(achievements: List[Dict]) -> List[Dict]:
        """Design cosmetics unlocked via achievements.
        
        These have prestige because they require skill/dedication.
        """
        cosmetics = []
        
        for achievement in achievements:
            difficulty = achievement.get("difficulty", "medium")
            
            # Harder achievements get cooler rewards
            reward_tier = {
                "easy": {"type": "icon", "rarity": "common"},
                "medium": {"type": "card_back", "rarity": "uncommon"},
                "hard": {"type": "effect", "rarity": "rare"},
                "legendary": {"type": "full_set", "rarity": "legendary"}
            }.get(difficulty, {"type": "icon", "rarity": "common"})
            
            cosmetics.append({
                "achievement_id": achievement["id"],
                "achievement_name": achievement["name"],
                "cosmetic_type": reward_tier["type"],
                "cosmetic_rarity": reward_tier["rarity"],
                "cosmetic_id": f"ach_{achievement['id']}_reward",
                "display_name": f"{achievement['name']} Trophy"
            })
        
        return cosmetics
```

---

This concludes Part I of the Encyclopedia of Game Economy and Monetization Systems (Chapters 1-4). The full document continues with:

- **Part II**: Player Segmentation (archetypes, metrics, LTV, cohorts)
- **Part III**: Monetization Models (premium, battle pass, cosmetics, subscription)
- **Part IV**: Progression Economics (XP curves, unlocks, prestige, seasons)
- **Part V**: Balance and Fairness (P2W prevention, free experience, integrity)
- **Part VI**: Implementation Systems (simulation, A/B testing, dynamic pricing)
- **Part VII**: Analytics and Optimization (KPIs, funnels, forecasting, churn)
- **Part VIII**: Regional and Legal (pricing, compliance, refunds, platforms)

---

*Continue to Part II: Player Segmentation...*
