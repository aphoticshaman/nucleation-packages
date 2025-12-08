# ENCYCLOPEDIA OF CARD GAME SYSTEMS
## Volume III: Deckbuilder Mechanics, Probability Theory, and Balance Engineering
### Technical Reference for AI-Automated Card Game Development
### Version 1.0 | November 2025

---

# PREFACE

This encyclopedia provides exhaustive technical documentation for designing, implementing, and balancing digital card games—specifically roguelike deckbuilders. Written at research-laboratory standards, it covers the mathematical foundations of card game probability, computational approaches to balance, and production-ready implementation patterns.

The goal: enable AI systems to generate mechanically sound, balanced, and engaging card content at scale while maintaining the emergent complexity that makes deckbuilders compelling.

---

# TABLE OF CONTENTS

## PART I: MATHEMATICAL FOUNDATIONS OF CARD GAMES
- Chapter 1: Probability Theory for Card Games
- Chapter 2: Combinatorics of Deck Composition
- Chapter 3: Markov Chain Analysis of Game States
- Chapter 4: Information Theory and Hidden Information

## PART II: CARD DESIGN THEORY
- Chapter 5: Card Anatomy and Effect Taxonomy
- Chapter 6: Resource Systems and Economy
- Chapter 7: Synergy Networks and Combo Theory
- Chapter 8: Power Curves and Card Evaluation

## PART III: DECK MECHANICS
- Chapter 9: Draw Systems and Card Flow
- Chapter 10: Discard and Exile Mechanics
- Chapter 11: Deck Manipulation and Tutoring
- Chapter 12: Multi-Zone Architecture

## PART IV: COMBAT SYSTEMS
- Chapter 13: Turn Structure and Priority
- Chapter 14: Damage Calculation Systems
- Chapter 15: Status Effects and Buffs
- Chapter 16: Targeting and Area Effects

## PART V: BALANCE ENGINEERING
- Chapter 17: Mathematical Balance Frameworks
- Chapter 18: Automated Balance Testing
- Chapter 19: Monte Carlo Simulation Methods
- Chapter 20: Genetic Algorithm Tuning

## PART VI: CONTENT GENERATION
- Chapter 21: Procedural Card Generation
- Chapter 22: Effect Composition Systems
- Chapter 23: Flavor Text and Theming
- Chapter 24: Rarity and Distribution

## PART VII: ROGUELIKE INTEGRATION
- Chapter 25: Run Structure and Progression
- Chapter 26: Reward Systems and Choices
- Chapter 27: Difficulty Scaling
- Chapter 28: Meta-Progression Systems

## PART VIII: IMPLEMENTATION PATTERNS
- Chapter 29: Card Definition Languages
- Chapter 30: Effect Resolution Engines
- Chapter 31: AI Opponents
- Chapter 32: Replay and Determinism

---

# PART I: MATHEMATICAL FOUNDATIONS OF CARD GAMES

---

## Chapter 1: Probability Theory for Card Games

### 1.1 Fundamental Probability Concepts

Card games are fundamentally probability machines. Understanding the mathematics enables both design and AI generation of mechanically sound content.

```python
"""
PROBABILITY FOUNDATIONS FOR CARD GAMES

Key concepts:
- Sample space: All possible outcomes
- Event: Subset of sample space
- Probability: Measure function P: Events → [0,1]

For card games:
- Sample space often = all possible deck orderings
- Events = specific cards in hand, specific sequences
- Probability affected by deck composition, draw mechanics
"""

import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from functools import lru_cache
from fractions import Fraction
import numpy as np


def factorial(n: int) -> int:
    """Factorial with memoization for performance."""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n,k) = n! / (k!(n-k)!)
    
    Number of ways to choose k items from n items without replacement.
    
    Properties:
    - C(n,0) = C(n,n) = 1
    - C(n,k) = C(n,n-k) (symmetry)
    - C(n,k) = C(n-1,k-1) + C(n-1,k) (Pascal's identity)
    
    Applications in card games:
    - Probability of specific opening hands
    - Number of possible deck combinations
    - Probability of drawing combo pieces
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use symmetry for efficiency
    k = min(k, n - k)
    
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def multinomial(n: int, groups: List[int]) -> int:
    """Multinomial coefficient: n! / (k1! * k2! * ... * km!)
    
    Number of ways to partition n items into groups of sizes k1, k2, ..., km.
    
    Applications:
    - Probability of specific distributions of card types
    - Number of ways to arrange cards with duplicates
    """
    if sum(groups) != n:
        raise ValueError("Groups must sum to n")
    
    result = factorial(n)
    for k in groups:
        result //= factorial(k)
    return result


class HypergeometricDistribution:
    """Hypergeometric distribution for card drawing.
    
    Models drawing without replacement from a finite population.
    
    Parameters:
    - N: Population size (deck size)
    - K: Number of success states in population (copies of desired card)
    - n: Number of draws (hand size)
    - k: Number of observed successes (copies drawn)
    
    P(X = k) = C(K,k) * C(N-K, n-k) / C(N,n)
    
    This is THE fundamental distribution for card game probability.
    """
    
    def __init__(self, population: int, successes_in_population: int, draws: int):
        """
        Args:
            population: Total cards in deck (N)
            successes_in_population: Copies of desired card (K)
            draws: Cards drawn (n)
        """
        self.N = population
        self.K = successes_in_population
        self.n = draws
        
        # Validate
        if self.K > self.N:
            raise ValueError("Cannot have more successes than population")
        if self.n > self.N:
            raise ValueError("Cannot draw more than population")
    
    def pmf(self, k: int) -> float:
        """Probability mass function: P(X = k)
        
        Probability of drawing exactly k copies of the desired card.
        """
        if k < 0 or k > min(self.K, self.n):
            return 0.0
        if k > self.K or (self.n - k) > (self.N - self.K):
            return 0.0
        
        numerator = binomial(self.K, k) * binomial(self.N - self.K, self.n - k)
        denominator = binomial(self.N, self.n)
        
        return numerator / denominator
    
    def cdf(self, k: int) -> float:
        """Cumulative distribution function: P(X <= k)
        
        Probability of drawing at most k copies.
        """
        return sum(self.pmf(i) for i in range(k + 1))
    
    def sf(self, k: int) -> float:
        """Survival function: P(X > k) = 1 - P(X <= k)
        
        Probability of drawing more than k copies.
        """
        return 1.0 - self.cdf(k)
    
    def at_least(self, k: int) -> float:
        """P(X >= k): Probability of drawing at least k copies.
        
        This is the most common question in card games:
        "What's the probability I draw at least one copy?"
        """
        return 1.0 - self.cdf(k - 1)
    
    def expected_value(self) -> float:
        """E[X] = n * K / N
        
        Expected number of copies drawn.
        """
        return self.n * self.K / self.N
    
    def variance(self) -> float:
        """Var(X) = n * K * (N-K) * (N-n) / (N² * (N-1))"""
        N, K, n = self.N, self.K, self.n
        return n * K * (N - K) * (N - n) / (N * N * (N - 1))
    
    def std_dev(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.variance())
    
    def mode(self) -> int:
        """Most likely value of X."""
        # Mode is floor or ceiling of (n+1)(K+1)/(N+2)
        m = (self.n + 1) * (self.K + 1) / (self.N + 2)
        candidates = [int(m), int(m) + 1]
        return max(candidates, key=self.pmf)


class DeckProbabilityCalculator:
    """Calculate probabilities for deck-based card games.
    
    Handles common questions:
    - Opening hand probabilities
    - Probability of drawing specific cards by turn N
    - Combo piece acquisition rates
    - Consistency metrics
    """
    
    def __init__(self, deck_size: int):
        self.deck_size = deck_size
    
    def opening_hand_probability(self, copies: int, hand_size: int, 
                                  at_least: int = 1) -> float:
        """Probability of having at least N copies in opening hand.
        
        Args:
            copies: Number of copies of card in deck
            hand_size: Opening hand size
            at_least: Minimum copies needed
        
        Returns:
            Probability of drawing at least 'at_least' copies
        
        Example:
            # 3 copies of key card, 5-card hand, need at least 1
            prob = calc.opening_hand_probability(3, 5, 1)
        """
        dist = HypergeometricDistribution(self.deck_size, copies, hand_size)
        return dist.at_least(at_least)
    
    def probability_by_turn(self, copies: int, hand_size: int, 
                           turn: int, draws_per_turn: int = 1,
                           at_least: int = 1) -> float:
        """Probability of having card by specific turn.
        
        Accounts for opening hand + draws on subsequent turns.
        
        Args:
            copies: Copies in deck
            hand_size: Opening hand size
            turn: Target turn (1 = first turn)
            draws_per_turn: Cards drawn each turn
            at_least: Minimum copies needed
        
        Returns:
            Probability of having at least 'at_least' by turn start
        """
        # Total cards seen by start of turn N
        # = hand_size + (turn - 1) * draws_per_turn
        total_cards_seen = hand_size + (turn - 1) * draws_per_turn
        total_cards_seen = min(total_cards_seen, self.deck_size)
        
        dist = HypergeometricDistribution(self.deck_size, copies, total_cards_seen)
        return dist.at_least(at_least)
    
    def mulligan_probability(self, copies: int, hand_size: int,
                            mulligan_count: int = 1,
                            at_least: int = 1,
                            full_mulligan: bool = True) -> float:
        """Probability after mulligans.
        
        Args:
            copies: Copies in deck
            hand_size: Hand size
            mulligan_count: Number of mulligans allowed
            at_least: Minimum copies needed
            full_mulligan: True if full hand replaced, False if partial (London)
        
        Returns:
            Probability of having at least 'at_least' after optimal mulligans
        
        Note: Assumes mulligan only if condition not met (optimal play).
        """
        # Probability of hitting in single hand
        p_hit = self.opening_hand_probability(copies, hand_size, at_least)
        
        if full_mulligan:
            # Each mulligan is independent
            # P(miss all) = (1 - p_hit)^(mulligan_count + 1)
            # P(hit at least once) = 1 - P(miss all)
            p_miss_all = (1 - p_hit) ** (mulligan_count + 1)
            return 1 - p_miss_all
        else:
            # London mulligan: more complex, depends on strategy
            # Simplified: assume similar improvement factor
            return 1 - (1 - p_hit) ** (mulligan_count + 1)
    
    def combo_probability(self, piece_copies: List[int], hand_size: int,
                         pieces_needed: int = None) -> float:
        """Probability of assembling a combo.
        
        Args:
            piece_copies: List of copies for each piece [3, 2, 4] for 3 different pieces
            hand_size: Hand size
            pieces_needed: How many unique pieces needed (default: all)
        
        Returns:
            Probability of having at least one of each required piece
        
        This is complex because draws are not independent.
        Uses inclusion-exclusion for exact calculation.
        """
        if pieces_needed is None:
            pieces_needed = len(piece_copies)
        
        if pieces_needed > len(piece_copies):
            return 0.0
        
        if pieces_needed == 1:
            # At least one of any piece
            return 1 - self._miss_all_pieces(piece_copies, hand_size)
        
        # For multiple pieces, use Monte Carlo for accuracy
        return self._monte_carlo_combo(piece_copies, hand_size, pieces_needed)
    
    def _miss_all_pieces(self, piece_copies: List[int], hand_size: int) -> float:
        """Probability of missing all combo pieces."""
        non_piece_cards = self.deck_size - sum(piece_copies)
        if non_piece_cards < hand_size:
            return 0.0  # Must draw at least one piece
        
        return binomial(non_piece_cards, hand_size) / binomial(self.deck_size, hand_size)
    
    def _monte_carlo_combo(self, piece_copies: List[int], hand_size: int,
                          pieces_needed: int, simulations: int = 100000) -> float:
        """Monte Carlo simulation for combo probability."""
        # Build deck representation
        deck = []
        for i, copies in enumerate(piece_copies):
            deck.extend([i] * copies)
        # Fill rest with non-pieces
        non_pieces = self.deck_size - len(deck)
        deck.extend([-1] * non_pieces)
        
        deck = np.array(deck)
        hits = 0
        
        rng = np.random.default_rng(42)
        
        for _ in range(simulations):
            # Shuffle and draw
            shuffled = rng.permutation(deck)
            hand = shuffled[:hand_size]
            
            # Count unique pieces
            pieces_in_hand = set(hand[hand >= 0])
            if len(pieces_in_hand) >= pieces_needed:
                hits += 1
        
        return hits / simulations
    
    def expected_turns_to_draw(self, copies: int, hand_size: int,
                               draws_per_turn: int = 1) -> float:
        """Expected number of turns until drawing at least one copy.
        
        Uses geometric distribution approximation.
        """
        # This is complex because probability changes each turn
        # Use simulation for accuracy
        simulations = 50000
        total_turns = 0
        
        rng = np.random.default_rng(42)
        
        for _ in range(simulations):
            # Build deck
            deck = [1] * copies + [0] * (self.deck_size - copies)
            rng.shuffle(deck)
            
            # Draw opening hand
            hand = deck[:hand_size]
            deck = deck[hand_size:]
            
            if 1 in hand:
                total_turns += 1
                continue
            
            turn = 1
            while deck and 1 not in hand:
                # Draw for turn
                draws = min(draws_per_turn, len(deck))
                hand.extend(deck[:draws])
                deck = deck[draws:]
                turn += 1
            
            total_turns += turn
        
        return total_turns / simulations


class DeckConsistencyMetrics:
    """Metrics for evaluating deck consistency.
    
    Consistency = probability of executing game plan reliably.
    Higher consistency = more competitive viability.
    """
    
    def __init__(self, deck_composition: Dict[str, int], deck_size: int = 40):
        """
        Args:
            deck_composition: {card_name: copies} mapping
            deck_size: Total deck size
        """
        self.composition = deck_composition
        self.deck_size = deck_size
        self.calc = DeckProbabilityCalculator(deck_size)
        
        # Validate
        total_cards = sum(deck_composition.values())
        if total_cards != deck_size:
            raise ValueError(f"Deck has {total_cards} cards, expected {deck_size}")
    
    def redundancy_factor(self, card_names: List[str]) -> float:
        """Measure redundancy of a card type.
        
        Higher = more copies of functionally similar cards.
        
        Args:
            card_names: Cards that serve similar function
        
        Returns:
            Total copies / deck_size (as ratio)
        """
        total_copies = sum(self.composition.get(name, 0) for name in card_names)
        return total_copies / self.deck_size
    
    def access_probability(self, card_names: List[str], 
                          hand_size: int = 5) -> float:
        """Probability of accessing at least one of the named cards.
        
        Args:
            card_names: Cards to check
            hand_size: Opening hand size
        
        Returns:
            Probability of having at least one in opening hand
        """
        total_copies = sum(self.composition.get(name, 0) for name in card_names)
        return self.calc.opening_hand_probability(total_copies, hand_size, 1)
    
    def deck_consistency_score(self, key_cards: List[str],
                              hand_size: int = 5,
                              target_turn: int = 3) -> float:
        """Overall consistency score for deck.
        
        Measures probability of accessing key cards by target turn.
        
        Args:
            key_cards: Cards essential to game plan
            hand_size: Opening hand size
            target_turn: Turn by which key cards should be available
        
        Returns:
            Score from 0 to 1 (higher = more consistent)
        """
        total_copies = sum(self.composition.get(name, 0) for name in key_cards)
        return self.calc.probability_by_turn(total_copies, hand_size, target_turn, 1, 1)
    
    def curve_analysis(self, cost_mapping: Dict[str, int]) -> Dict[int, float]:
        """Analyze mana/energy curve of deck.
        
        Args:
            cost_mapping: {card_name: cost} for each card
        
        Returns:
            {cost: proportion} showing deck's cost distribution
        """
        cost_counts: Dict[int, int] = {}
        
        for card_name, copies in self.composition.items():
            cost = cost_mapping.get(card_name, 0)
            cost_counts[cost] = cost_counts.get(cost, 0) + copies
        
        return {cost: count / self.deck_size for cost, count in sorted(cost_counts.items())}
    
    def expected_hand_cost(self, cost_mapping: Dict[str, int],
                          hand_size: int = 5) -> Tuple[float, float]:
        """Expected total cost of opening hand.
        
        Args:
            cost_mapping: {card_name: cost}
            hand_size: Hand size
        
        Returns:
            (expected_total_cost, std_deviation)
        """
        # Build cost array
        costs = []
        for card_name, copies in self.composition.items():
            cost = cost_mapping.get(card_name, 0)
            costs.extend([cost] * copies)
        
        costs = np.array(costs)
        
        # Expected value of sum of draws
        # E[sum] = hand_size * E[single_draw] = hand_size * mean(costs)
        mean_cost = np.mean(costs)
        expected_total = hand_size * mean_cost
        
        # Variance is more complex for sampling without replacement
        # Var(sum) = hand_size * (N-hand_size)/(N-1) * Var(single)
        var_single = np.var(costs)
        N = self.deck_size
        n = hand_size
        variance_total = n * (N - n) / (N - 1) * var_single
        std_dev = math.sqrt(variance_total)
        
        return expected_total, std_dev
```

### 1.2 Conditional Probability and Bayesian Reasoning

```python
"""
CONDITIONAL PROBABILITY IN CARD GAMES

Critical for:
- Updating beliefs based on opponent's plays
- Optimal decision-making with partial information
- AI opponent modeling
"""

class BayesianCardTracker:
    """Track probabilities of hidden cards using Bayesian updating.
    
    As opponent plays cards, we update beliefs about their deck/hand.
    """
    
    def __init__(self, known_card_pool: Dict[str, int], 
                 opponent_deck_size: int = 40):
        """
        Args:
            known_card_pool: All possible cards opponent could have
            opponent_deck_size: Expected deck size
        """
        self.card_pool = known_card_pool
        self.deck_size = opponent_deck_size
        
        # Prior: uniform over possible deck compositions
        # We track observed cards to update
        self.observed_cards: Dict[str, int] = {}
        self.cards_in_opponent_hand: int = 0
        self.cards_drawn_total: int = 0
    
    def observe_play(self, card_name: str, count: int = 1):
        """Update beliefs when opponent plays a card."""
        self.observed_cards[card_name] = self.observed_cards.get(card_name, 0) + count
        self.cards_in_opponent_hand = max(0, self.cards_in_opponent_hand - count)
    
    def observe_draw(self, count: int = 1):
        """Update when opponent draws cards."""
        self.cards_drawn_total += count
        self.cards_in_opponent_hand += count
    
    def probability_in_hand(self, card_name: str, copies_in_deck: int = None) -> float:
        """Estimate probability opponent has specific card in hand.
        
        Uses hypergeometric with observed information.
        
        Args:
            card_name: Card to check
            copies_in_deck: Known/assumed copies (default: max from pool)
        
        Returns:
            Estimated probability card is in opponent's hand
        """
        if copies_in_deck is None:
            copies_in_deck = self.card_pool.get(card_name, 0)
        
        # Subtract already observed copies
        remaining_copies = copies_in_deck - self.observed_cards.get(card_name, 0)
        remaining_copies = max(0, remaining_copies)
        
        if remaining_copies == 0:
            return 0.0
        
        # Remaining deck size
        total_observed = sum(self.observed_cards.values())
        remaining_deck = self.deck_size - total_observed
        
        if remaining_deck <= 0:
            return 0.0
        
        # Probability at least one copy is in hand
        dist = HypergeometricDistribution(
            remaining_deck, 
            remaining_copies,
            self.cards_in_opponent_hand
        )
        
        return dist.at_least(1)
    
    def probability_in_deck(self, card_name: str, copies_in_deck: int = None) -> float:
        """Probability card is still in opponent's deck (not hand or played)."""
        if copies_in_deck is None:
            copies_in_deck = self.card_pool.get(card_name, 0)
        
        remaining_copies = copies_in_deck - self.observed_cards.get(card_name, 0)
        remaining_copies = max(0, remaining_copies)
        
        if remaining_copies == 0:
            return 0.0
        
        # They have it somewhere (hand or deck)
        # Probability it's in deck = 1 - P(in hand)
        return 1.0 - self.probability_in_hand(card_name, copies_in_deck)
    
    def most_likely_hand(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return most likely cards in opponent's hand.
        
        Useful for AI decision-making.
        """
        probabilities = []
        
        for card_name, max_copies in self.card_pool.items():
            prob = self.probability_in_hand(card_name, max_copies)
            if prob > 0.01:  # Threshold for relevance
                probabilities.append((card_name, prob))
        
        probabilities.sort(key=lambda x: x[1], reverse=True)
        return probabilities[:top_n]


class ConditionalProbabilityEngine:
    """Calculate conditional probabilities for game decisions.
    
    P(A|B) = P(A and B) / P(B)
    
    Used for questions like:
    - "Given I have card X, what's P(drawing card Y)?"
    - "If opponent played card A, what's P(they have card B)?"
    """
    
    def __init__(self, deck_size: int):
        self.deck_size = deck_size
    
    def prob_draw_given_hand(self, target_copies: int, 
                            target_in_hand: int,
                            hand_size: int,
                            deck_remaining: int = None) -> float:
        """P(draw target next | target in hand)
        
        Given we have some copies of a card in hand,
        what's probability of drawing another?
        
        Args:
            target_copies: Total copies in deck originally
            target_in_hand: Copies already in our hand
            hand_size: Current hand size
            deck_remaining: Cards left in deck
        
        Returns:
            Probability next draw is the target card
        """
        if deck_remaining is None:
            deck_remaining = self.deck_size - hand_size
        
        copies_remaining = target_copies - target_in_hand
        
        if copies_remaining <= 0 or deck_remaining <= 0:
            return 0.0
        
        return copies_remaining / deck_remaining
    
    def prob_opponent_has_given_no_play(self, card_copies: int,
                                        turns_without_play: int,
                                        estimated_hand_size: int,
                                        play_if_have_prob: float = 0.8) -> float:
        """Bayesian update for opponent not playing a card.
        
        If opponent hasn't played card X for N turns, 
        what's probability they have it?
        
        Uses Bayes' theorem:
        P(have|no_play) = P(no_play|have) * P(have) / P(no_play)
        
        Args:
            card_copies: Assumed copies in opponent's deck
            turns_without_play: Turns since card could have been played
            estimated_hand_size: Current hand estimate
            play_if_have_prob: Assumed probability they'd play if they had it
        
        Returns:
            Updated probability they have the card
        """
        # Prior: probability they have at least one copy
        dist = HypergeometricDistribution(self.deck_size, card_copies, estimated_hand_size)
        prior_have = dist.at_least(1)
        
        # P(no_play | have) = (1 - play_if_have)^turns
        prob_no_play_given_have = (1 - play_if_have_prob) ** turns_without_play
        
        # P(no_play | don't have) = 1
        prob_no_play_given_not_have = 1.0
        
        # P(no_play) = P(no_play|have)*P(have) + P(no_play|~have)*P(~have)
        prob_no_play = (prob_no_play_given_have * prior_have + 
                       prob_no_play_given_not_have * (1 - prior_have))
        
        # Bayes update
        if prob_no_play < 1e-10:
            return prior_have
        
        posterior_have = prob_no_play_given_have * prior_have / prob_no_play
        
        return posterior_have
```

### 1.3 Expected Value and Decision Theory

```python
"""
EXPECTED VALUE CALCULATIONS FOR CARD GAMES

EV is the fundamental metric for decision quality.
EV = Σ P(outcome) × Value(outcome)

In card games:
- Value = damage dealt, resources gained, board state change
- Outcomes weighted by probability of draws, opponent responses
"""

@dataclass
class GameOutcome:
    """Represents a possible game outcome with probability and value."""
    probability: float
    value: float
    description: str = ""
    
    @property
    def ev_contribution(self) -> float:
        return self.probability * self.value


class ExpectedValueCalculator:
    """Calculate expected values for card game decisions."""
    
    def __init__(self):
        self.outcomes: List[GameOutcome] = []
    
    def add_outcome(self, probability: float, value: float, description: str = ""):
        """Add possible outcome."""
        self.outcomes.append(GameOutcome(probability, value, description))
    
    def clear(self):
        """Reset outcomes."""
        self.outcomes = []
    
    def expected_value(self) -> float:
        """Calculate total expected value."""
        return sum(o.ev_contribution for o in self.outcomes)
    
    def variance(self) -> float:
        """Calculate variance of outcomes."""
        ev = self.expected_value()
        return sum(o.probability * (o.value - ev) ** 2 for o in self.outcomes)
    
    def std_deviation(self) -> float:
        """Standard deviation of outcomes."""
        return math.sqrt(self.variance())
    
    def best_case(self) -> GameOutcome:
        """Best possible outcome."""
        return max(self.outcomes, key=lambda o: o.value)
    
    def worst_case(self) -> GameOutcome:
        """Worst possible outcome."""
        return min(self.outcomes, key=lambda o: o.value)
    
    def probability_above(self, threshold: float) -> float:
        """Probability of outcome above threshold."""
        return sum(o.probability for o in self.outcomes if o.value > threshold)
    
    def conditional_ev_above(self, threshold: float) -> float:
        """EV given outcome is above threshold."""
        good_outcomes = [o for o in self.outcomes if o.value > threshold]
        total_prob = sum(o.probability for o in good_outcomes)
        if total_prob < 1e-10:
            return 0.0
        return sum(o.probability * o.value for o in good_outcomes) / total_prob


class CardPlayEvaluator:
    """Evaluate expected value of playing specific cards.
    
    Considers:
    - Immediate effect value
    - Probability of follow-up draws
    - Opponent response modeling
    - Long-term game state impact
    """
    
    def __init__(self, deck_state: 'DeckState', game_state: 'GameState'):
        self.deck_state = deck_state
        self.game_state = game_state
    
    def evaluate_play(self, card: 'Card') -> Tuple[float, Dict]:
        """Evaluate expected value of playing a card.
        
        Returns:
            (expected_value, breakdown_dict)
        """
        ev_calc = ExpectedValueCalculator()
        breakdown = {}
        
        # Immediate effect value
        immediate_value = self._evaluate_immediate_effect(card)
        ev_calc.add_outcome(1.0, immediate_value, "immediate_effect")
        breakdown['immediate'] = immediate_value
        
        # Draw-dependent effects (if card draws)
        if card.draws_cards:
            draw_ev = self._evaluate_draw_effects(card)
            breakdown['draw_effects'] = draw_ev
            # This modifies ev_calc internally
        
        # Synergy with hand
        synergy_value = self._evaluate_synergy(card)
        ev_calc.add_outcome(1.0, synergy_value, "synergy")
        breakdown['synergy'] = synergy_value
        
        # Tempo/resource considerations
        tempo_value = self._evaluate_tempo(card)
        ev_calc.add_outcome(1.0, tempo_value, "tempo")
        breakdown['tempo'] = tempo_value
        
        return ev_calc.expected_value(), breakdown
    
    def _evaluate_immediate_effect(self, card: 'Card') -> float:
        """Value of card's immediate effect."""
        value = 0.0
        
        # Damage
        if hasattr(card, 'damage'):
            value += card.damage * self._damage_value_multiplier()
        
        # Block/defense
        if hasattr(card, 'block'):
            value += card.block * self._block_value_multiplier()
        
        # Resource gain
        if hasattr(card, 'energy_gain'):
            value += card.energy_gain * 2.0  # Energy is valuable
        
        return value
    
    def _evaluate_draw_effects(self, card: 'Card') -> float:
        """Expected value of cards drawn."""
        if not hasattr(card, 'draw_count'):
            return 0.0
        
        draws = card.draw_count
        deck_ev = self._average_card_ev()
        
        return draws * deck_ev
    
    def _evaluate_synergy(self, card: 'Card') -> float:
        """Value from synergy with current hand/deck."""
        synergy = 0.0
        
        # Check for keyword synergies
        # Placeholder - would check actual game mechanics
        
        return synergy
    
    def _evaluate_tempo(self, card: 'Card') -> float:
        """Tempo value (efficiency of cost vs effect)."""
        if not hasattr(card, 'cost') or card.cost == 0:
            return 0.0
        
        effect_value = self._evaluate_immediate_effect(card)
        cost = card.cost
        
        # Tempo = getting more value than expected for cost
        expected_value_per_cost = 2.5  # Baseline assumption
        tempo = effect_value - (cost * expected_value_per_cost)
        
        return tempo
    
    def _damage_value_multiplier(self) -> float:
        """Context-dependent damage value.
        
        Damage is worth more when enemy is low HP.
        """
        if self.game_state.enemy_hp <= 20:
            return 1.5  # Lethal range
        return 1.0
    
    def _block_value_multiplier(self) -> float:
        """Context-dependent block value.
        
        Block is worth more when we're low HP or facing big attack.
        """
        if self.game_state.player_hp <= 20:
            return 1.5
        if self.game_state.incoming_damage > 15:
            return 1.3
        return 1.0
    
    def _average_card_ev(self) -> float:
        """Average EV of drawing a random card from deck."""
        # Would calculate based on actual deck composition
        return 3.0  # Placeholder


@dataclass
class DeckState:
    """Current state of player's deck."""
    cards_in_deck: List['Card']
    cards_in_discard: List['Card']
    cards_in_hand: List['Card']
    cards_exhausted: List['Card']


@dataclass
class GameState:
    """Current game state for evaluation."""
    player_hp: int
    player_max_hp: int
    player_energy: int
    enemy_hp: int
    enemy_max_hp: int
    incoming_damage: int
    turn_number: int
    status_effects: Dict[str, int]
```

---

## Chapter 2: Combinatorics of Deck Composition

### 2.1 Deck Building Combinatorics

```python
"""
COMBINATORICS OF DECK BUILDING

Understanding the mathematical space of possible decks is essential for:
- Balancing card pools
- Designing draft/collection systems
- Generating varied AI opponents
- Measuring meta diversity
"""

class DeckCombinatorics:
    """Calculate combinatorial properties of deck building."""
    
    def __init__(self, card_pool_size: int, max_copies: int = 3, deck_size: int = 40):
        """
        Args:
            card_pool_size: Total unique cards available
            max_copies: Maximum copies of any single card
            deck_size: Required deck size
        """
        self.pool_size = card_pool_size
        self.max_copies = max_copies
        self.deck_size = deck_size
    
    def possible_decks_unlimited(self) -> int:
        """Number of possible decks with no copy limit.
        
        Stars and bars: choosing deck_size cards from pool_size with replacement.
        = C(pool_size + deck_size - 1, deck_size)
        """
        return binomial(self.pool_size + self.deck_size - 1, self.deck_size)
    
    def possible_decks_with_limit(self) -> int:
        """Number of possible decks with copy limit.
        
        This is more complex - uses generating functions.
        
        Each card can appear 0, 1, 2, ..., max_copies times.
        Generating function per card: 1 + x + x² + ... + x^max_copies
        
        We need coefficient of x^deck_size in (1 + x + ... + x^k)^n
        where k = max_copies, n = pool_size
        """
        # For exact calculation with limits, use dynamic programming
        return self._count_decks_dp()
    
    def _count_decks_dp(self) -> int:
        """Count valid decks using dynamic programming.
        
        dp[i][j] = number of ways to select j cards from first i unique cards
        """
        # This can be huge - return approximation for large values
        if self.pool_size > 100 or self.deck_size > 100:
            return self._approximate_deck_count()
        
        # dp[j] = ways to reach exactly j cards
        dp = [0] * (self.deck_size + 1)
        dp[0] = 1
        
        for card in range(self.pool_size):
            # Iterate backwards to avoid using same card multiple times in one pass
            new_dp = dp.copy()
            for copies in range(1, self.max_copies + 1):
                for j in range(copies, self.deck_size + 1):
                    new_dp[j] += dp[j - copies]
            dp = new_dp
        
        return dp[self.deck_size]
    
    def _approximate_deck_count(self) -> float:
        """Approximate number of valid decks for large values.
        
        Uses normal approximation to multinomial.
        """
        # Expected cards per unique card = deck_size / pool_size
        avg_per_card = self.deck_size / self.pool_size
        
        if avg_per_card > self.max_copies:
            # Not enough variety - need more unique cards
            return 0
        
        # Approximation using entropy
        # Number of ways ≈ exp(n * H(p)) where p is distribution
        import math
        
        # Assume roughly uniform distribution up to max_copies
        # This is a rough approximation
        effective_choices = min(self.max_copies + 1, self.deck_size + 1)
        log_count = self.pool_size * math.log(effective_choices)
        
        return math.exp(min(log_count, 700))  # Prevent overflow
    
    def deck_diversity_metric(self, deck_a: List[Tuple[str, int]], 
                             deck_b: List[Tuple[str, int]]) -> float:
        """Measure how different two decks are.
        
        Returns value from 0 (identical) to 1 (completely different).
        
        Uses Jaccard-like distance on multisets.
        """
        # Convert to multiset representation
        set_a = {}
        set_b = {}
        
        for card, count in deck_a:
            set_a[card] = set_a.get(card, 0) + count
        for card, count in deck_b:
            set_b[card] = set_b.get(card, 0) + count
        
        # Calculate intersection and union sizes
        all_cards = set(set_a.keys()) | set(set_b.keys())
        
        intersection = 0
        union = 0
        
        for card in all_cards:
            count_a = set_a.get(card, 0)
            count_b = set_b.get(card, 0)
            
            intersection += min(count_a, count_b)
            union += max(count_a, count_b)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        return 1.0 - jaccard
    
    def archetype_coverage(self, card_pool: List['Card'],
                          archetype_tags: Dict[str, List[str]]) -> Dict[str, float]:
        """Measure how well card pool supports different archetypes.
        
        Args:
            card_pool: Available cards
            archetype_tags: {archetype_name: [relevant_card_tags]}
        
        Returns:
            {archetype_name: coverage_score}
        """
        coverage = {}
        
        for archetype, tags in archetype_tags.items():
            # Count cards that match archetype
            matching_cards = []
            for card in card_pool:
                card_tags = getattr(card, 'tags', [])
                if any(tag in card_tags for tag in tags):
                    matching_cards.append(card)
            
            # Coverage = can we build a viable deck?
            # Need at least deck_size / 2 relevant cards
            min_needed = self.deck_size // 2
            total_copies = sum(getattr(c, 'max_copies', self.max_copies) 
                              for c in matching_cards)
            
            if total_copies >= min_needed:
                # Score based on variety
                variety = len(matching_cards) / (self.deck_size / self.max_copies)
                coverage[archetype] = min(1.0, variety)
            else:
                coverage[archetype] = total_copies / min_needed
        
        return coverage


class DraftMathematics:
    """Mathematics of draft formats (pick one of N cards)."""
    
    def __init__(self, pack_size: int = 15, picks_per_pack: int = 15,
                 num_packs: int = 3):
        self.pack_size = pack_size
        self.picks_per_pack = picks_per_pack
        self.num_packs = num_packs
    
    def pick_probability(self, card_rarity_frequency: float,
                        pick_number: int) -> float:
        """Probability of specific card being available at pick N.
        
        Args:
            card_rarity_frequency: P(card appears in pack)
            pick_number: Which pick (1 = first pick)
        
        Returns:
            Probability card is available
        """
        # Simplified model: each previous pick has some chance of taking it
        # Assume card is taken if available with probability take_rate
        take_rate = 0.3  # Baseline take rate
        
        # P(available at pick N) = P(in pack) × P(not taken in picks 1..N-1)
        p_in_pack = card_rarity_frequency
        p_survive_picks = (1 - take_rate) ** (pick_number - 1)
        
        return p_in_pack * p_survive_picks
    
    def expected_card_quality(self, pick_number: int,
                             quality_distribution: List[float]) -> float:
        """Expected quality of best remaining card at pick N.
        
        Args:
            pick_number: Which pick
            quality_distribution: Quality scores of cards in pack
        
        Returns:
            Expected max quality available
        """
        # Order statistics: expected value of (pack_size - pick + 1)th highest
        # from quality_distribution
        
        cards_remaining = self.pack_size - pick_number + 1
        sorted_qualities = sorted(quality_distribution, reverse=True)
        
        if cards_remaining <= 0:
            return 0.0
        
        # Expected best of remaining (simplified)
        remaining = sorted_qualities[:cards_remaining]
        return max(remaining) if remaining else 0.0
    
    def signal_reading(self, passed_cards: List['Card'],
                      pack_number: int) -> Dict[str, float]:
        """Infer what colors/archetypes are open based on passed cards.
        
        Draft signal reading mathematics.
        
        Args:
            passed_cards: Cards passed to us
            pack_number: Which pack (affects interpretation)
        
        Returns:
            {archetype: openness_score}
        """
        # Late picks of powerful cards indicate archetype is open
        openness = {}
        
        for card in passed_cards:
            archetypes = getattr(card, 'archetypes', [])
            power = getattr(card, 'power_level', 1.0)
            pick = getattr(card, 'pick_number', 8)
            
            # Powerful card passed late = strong signal
            signal_strength = power * (pick / self.pack_size)
            
            for arch in archetypes:
                openness[arch] = openness.get(arch, 0) + signal_strength
        
        # Normalize
        if openness:
            max_signal = max(openness.values())
            if max_signal > 0:
                openness = {k: v/max_signal for k, v in openness.items()}
        
        return openness
```

### 2.2 Shuffle Theory and Randomization

```python
"""
SHUFFLE THEORY FOR CARD GAMES

Proper randomization is critical for fair play.
Understanding shuffle mathematics helps with:
- Ensuring actual randomness
- Detecting shuffle exploits
- Implementing efficient shuffles
"""

class ShuffleTheory:
    """Mathematical analysis of shuffling."""
    
    @staticmethod
    def riffle_shuffle_analysis(deck_size: int, 
                               num_shuffles: int) -> float:
        """Analyze mixing quality of riffle shuffles.
        
        A perfect riffle shuffle is deterministic!
        Real shuffles introduce randomness via imperfect cuts.
        
        Returns "mixing time" estimate (shuffles needed for randomness).
        """
        # Gilbert-Shannon-Reeds model
        # Mixing time ≈ (3/2) log₂(n)
        import math
        return 1.5 * math.log2(deck_size)
    
    @staticmethod
    def recommended_shuffles(deck_size: int) -> int:
        """Recommended number of shuffles for adequate randomization.
        
        Based on mathematical analysis of convergence to uniform distribution.
        """
        import math
        # Standard result: 7 shuffles for 52 cards
        # Scales as O(log n)
        base_shuffles = 7
        base_deck = 52
        
        scale_factor = math.log2(deck_size) / math.log2(base_deck)
        return max(5, int(base_shuffles * scale_factor + 0.5))
    
    @staticmethod
    def mana_weave_analysis() -> str:
        """Analysis of "mana weaving" (illegal pile shuffling).
        
        Common misconception that distributing lands evenly helps.
        """
        return """
MANA WEAVING ANALYSIS:

Claim: Spreading lands evenly through deck before shuffling improves draws.
Reality: This is either:
  a) Ineffective (if followed by sufficient shuffling)
  b) Cheating (if insufficient shuffling preserves pattern)

Mathematical proof:
- A sufficiently shuffled deck is uniformly random
- The starting configuration is irrelevant after mixing time
- If weaving "helps", the deck isn't properly shuffled

Proper solution for mana screw:
- Adjust deck composition (correct land/spell ratio)
- Use mulligan rules appropriately
- Accept variance as part of the game
"""
    
    @staticmethod
    def clump_probability(deck_size: int, card_copies: int,
                         clump_size: int) -> float:
        """Probability of N copies appearing consecutively in shuffled deck.
        
        Players often feel lands "clump" - this calculates actual probability.
        
        Args:
            deck_size: Total cards
            card_copies: Copies of the card type
            clump_size: Consecutive copies to check for
        
        Returns:
            Probability of at least one clump of that size
        """
        if clump_size > card_copies:
            return 0.0
        
        # Use runs theory
        # Approximate using inclusion-exclusion on possible positions
        
        # Number of possible starting positions for a clump
        positions = deck_size - clump_size + 1
        
        # Probability a specific position has clump
        # = P(all clump_size cards are the type) for consecutive positions
        # This is complex - use approximation
        
        p_single = 1.0
        remaining = deck_size
        copies_left = card_copies
        
        for i in range(clump_size):
            if remaining <= 0:
                p_single = 0
                break
            p_single *= copies_left / remaining
            copies_left -= 1
            remaining -= 1
        
        # Approximate P(at least one clump) using Poisson approximation
        expected_clumps = positions * p_single
        p_at_least_one = 1 - math.exp(-expected_clumps)
        
        return min(1.0, p_at_least_one)


class DeterministicShuffler:
    """Seedable shuffle for replay/testing.
    
    Critical for:
    - Deterministic replays
    - Bug reproduction
    - Automated testing
    - Seeded runs in roguelikes
    """
    
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.initial_seed = seed
    
    def shuffle(self, items: List) -> List:
        """Fisher-Yates shuffle with seeded RNG.
        
        O(n) time, O(1) extra space (in-place).
        Produces uniform random permutation.
        """
        result = items.copy()
        n = len(result)
        
        for i in range(n - 1, 0, -1):
            j = self.rng.integers(0, i + 1)
            result[i], result[j] = result[j], result[i]
        
        return result
    
    def shuffle_top(self, items: List, count: int) -> List:
        """Shuffle only top N cards (for effects like "shuffle into top 5")."""
        result = items.copy()
        
        top = result[:count]
        n = len(top)
        
        for i in range(n - 1, 0, -1):
            j = self.rng.integers(0, i + 1)
            top[i], top[j] = top[j], top[i]
        
        return top + result[count:]
    
    def insert_random(self, items: List, new_item) -> List:
        """Insert item at random position."""
        result = items.copy()
        position = self.rng.integers(0, len(result) + 1)
        result.insert(position, new_item)
        return result
    
    def get_state(self) -> dict:
        """Get RNG state for save/load."""
        return {
            'initial_seed': self.initial_seed,
            'bit_generator_state': self.rng.bit_generator.state
        }
    
    def set_state(self, state: dict):
        """Restore RNG state."""
        self.rng.bit_generator.state = state['bit_generator_state']
```

---

## Chapter 3: Markov Chain Analysis of Game States

### 3.1 Game State as Markov Process

```python
"""
MARKOV CHAIN ANALYSIS FOR CARD GAMES

A game can be modeled as transitions between states.
State = (deck, hand, discard, HP, energy, buffs, enemy state, ...)

This enables:
- Win probability estimation
- Optimal play calculation
- Game length analysis
- Balance verification
"""

from typing import TypeVar, Generic
from collections import defaultdict

State = TypeVar('State')

class MarkovGameModel(Generic[State]):
    """Model game as Markov chain for analysis."""
    
    def __init__(self):
        # Transition matrix: P[s1][s2] = probability of s1 -> s2
        self.transitions: Dict[State, Dict[State, float]] = defaultdict(dict)
        
        # Terminal states (win/lose)
        self.terminal_states: Set[State] = set()
        self.winning_states: Set[State] = set()
    
    def add_transition(self, from_state: State, to_state: State, 
                      probability: float):
        """Add state transition."""
        self.transitions[from_state][to_state] = probability
    
    def mark_terminal(self, state: State, is_win: bool):
        """Mark state as terminal (game over)."""
        self.terminal_states.add(state)
        if is_win:
            self.winning_states.add(state)
    
    def win_probability(self, start_state: State, 
                       max_iterations: int = 1000) -> float:
        """Calculate probability of winning from a state.
        
        Uses value iteration to solve:
        V(s) = 1 if s is winning terminal
        V(s) = 0 if s is losing terminal
        V(s) = Σ P(s,s') × V(s') otherwise
        """
        # Initialize values
        values: Dict[State, float] = {}
        
        for state in self.terminal_states:
            values[state] = 1.0 if state in self.winning_states else 0.0
        
        # Value iteration
        for _ in range(max_iterations):
            max_change = 0.0
            
            for state, transitions in self.transitions.items():
                if state in self.terminal_states:
                    continue
                
                new_value = 0.0
                for next_state, prob in transitions.items():
                    next_value = values.get(next_state, 0.5)  # Default 50%
                    new_value += prob * next_value
                
                old_value = values.get(state, 0.5)
                max_change = max(max_change, abs(new_value - old_value))
                values[state] = new_value
            
            if max_change < 1e-6:
                break
        
        return values.get(start_state, 0.5)
    
    def expected_game_length(self, start_state: State) -> float:
        """Expected number of transitions until terminal state.
        
        Solves: T(s) = 0 if terminal, else 1 + Σ P(s,s') × T(s')
        """
        lengths: Dict[State, float] = {}
        
        for state in self.terminal_states:
            lengths[state] = 0.0
        
        # Iterate until convergence
        for _ in range(1000):
            max_change = 0.0
            
            for state, transitions in self.transitions.items():
                if state in self.terminal_states:
                    continue
                
                new_length = 1.0
                for next_state, prob in transitions.items():
                    next_length = lengths.get(next_state, 10.0)  # Default
                    new_length += prob * next_length
                
                old_length = lengths.get(state, 10.0)
                max_change = max(max_change, abs(new_length - old_length))
                lengths[state] = new_length
            
            if max_change < 1e-6:
                break
        
        return lengths.get(start_state, 10.0)
    
    def steady_state_distribution(self) -> Dict[State, float]:
        """Find steady-state distribution (for cyclic games).
        
        π such that π = π × P (left eigenvector with eigenvalue 1)
        
        For card games, useful for:
        - Analyzing infinite game scenarios
        - Understanding meta equilibria
        """
        # Convert to matrix form and solve
        # Simplified: use power iteration
        
        states = list(set(self.transitions.keys()) | 
                     set(s for t in self.transitions.values() for s in t.keys()))
        state_idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        
        # Build transition matrix
        P = np.zeros((n, n))
        for from_state, transitions in self.transitions.items():
            i = state_idx[from_state]
            for to_state, prob in transitions.items():
                j = state_idx[to_state]
                P[i, j] = prob
        
        # Ensure rows sum to 1 (add self-loops for terminal states)
        for i in range(n):
            row_sum = P[i].sum()
            if row_sum < 1e-10:
                P[i, i] = 1.0  # Self-loop
            elif abs(row_sum - 1.0) > 1e-10:
                P[i] /= row_sum
        
        # Power iteration for stationary distribution
        pi = np.ones(n) / n
        for _ in range(1000):
            new_pi = pi @ P
            if np.allclose(pi, new_pi, atol=1e-10):
                break
            pi = new_pi
        
        return {states[i]: pi[i] for i in range(n)}


class SimplifiedCombatModel:
    """Simplified Markov model for combat analysis.
    
    State = (player_hp, enemy_hp)
    Useful for quick balance checks.
    """
    
    def __init__(self, player_max_hp: int, enemy_max_hp: int):
        self.player_max_hp = player_max_hp
        self.enemy_max_hp = enemy_max_hp
        
        # Average damage dealt/received per turn (will be set)
        self.avg_player_damage = 15.0
        self.player_damage_std = 5.0
        self.avg_enemy_damage = 10.0
        self.enemy_damage_std = 3.0
    
    def set_damage_profiles(self, player_avg: float, player_std: float,
                           enemy_avg: float, enemy_std: float):
        """Set damage distribution parameters."""
        self.avg_player_damage = player_avg
        self.player_damage_std = player_std
        self.avg_enemy_damage = enemy_avg
        self.enemy_damage_std = enemy_std
    
    def simulate_combat(self, num_simulations: int = 10000
                       ) -> Dict[str, float]:
        """Monte Carlo simulation of combat.
        
        Returns:
            {
                'win_rate': probability of player winning,
                'avg_turns': average combat length,
                'avg_remaining_hp': average HP if player wins
            }
        """
        rng = np.random.default_rng(42)
        
        wins = 0
        total_turns = 0
        remaining_hp_sum = 0
        
        for _ in range(num_simulations):
            player_hp = self.player_max_hp
            enemy_hp = self.enemy_max_hp
            turns = 0
            
            while player_hp > 0 and enemy_hp > 0:
                turns += 1
                
                # Player attacks
                damage = max(0, rng.normal(self.avg_player_damage, 
                                          self.player_damage_std))
                enemy_hp -= damage
                
                if enemy_hp <= 0:
                    break
                
                # Enemy attacks
                damage = max(0, rng.normal(self.avg_enemy_damage,
                                          self.enemy_damage_std))
                player_hp -= damage
            
            total_turns += turns
            
            if player_hp > 0:
                wins += 1
                remaining_hp_sum += player_hp
        
        return {
            'win_rate': wins / num_simulations,
            'avg_turns': total_turns / num_simulations,
            'avg_remaining_hp': remaining_hp_sum / wins if wins > 0 else 0
        }
    
    def required_damage_for_target_winrate(self, target_winrate: float,
                                           tolerance: float = 0.01) -> float:
        """Binary search for player damage to achieve target win rate.
        
        Useful for balance tuning.
        """
        low, high = 1.0, 100.0
        
        while high - low > 0.1:
            mid = (low + high) / 2
            self.avg_player_damage = mid
            
            result = self.simulate_combat(5000)
            actual_winrate = result['win_rate']
            
            if abs(actual_winrate - target_winrate) < tolerance:
                return mid
            elif actual_winrate < target_winrate:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2
```

### 3.2 Absorbing Markov Chains

```python
"""
ABSORBING MARKOV CHAIN ANALYSIS

Card games always end (absorbing states = win/lose).
Absorbing chain analysis gives:
- P(ending in state X | starting from Y)
- Expected steps to absorption
- Variance of game length
"""

class AbsorbingMarkovChain:
    """Analysis tools for absorbing Markov chains.
    
    Standard form: P = [[Q, R], [0, I]]
    - Q: transitions between transient states
    - R: transitions from transient to absorbing
    - I: identity (absorbing states stay)
    
    Fundamental matrix: N = (I - Q)^(-1)
    - N[i,j] = expected visits to state j starting from i
    """
    
    def __init__(self, transient_states: List[State], 
                 absorbing_states: List[State]):
        self.transient = transient_states
        self.absorbing = absorbing_states
        
        self.t_idx = {s: i for i, s in enumerate(transient_states)}
        self.a_idx = {s: i for i, s in enumerate(absorbing_states)}
        
        self.Q = np.zeros((len(transient_states), len(transient_states)))
        self.R = np.zeros((len(transient_states), len(absorbing_states)))
    
    def set_transition(self, from_state: State, to_state: State, prob: float):
        """Set transition probability."""
        if from_state not in self.t_idx:
            raise ValueError(f"From state {from_state} not transient")
        
        i = self.t_idx[from_state]
        
        if to_state in self.t_idx:
            j = self.t_idx[to_state]
            self.Q[i, j] = prob
        elif to_state in self.a_idx:
            j = self.a_idx[to_state]
            self.R[i, j] = prob
        else:
            raise ValueError(f"To state {to_state} not recognized")
    
    def fundamental_matrix(self) -> np.ndarray:
        """Compute fundamental matrix N = (I - Q)^(-1).
        
        N[i,j] = expected number of times in state j before absorption,
                 starting from state i.
        """
        n = len(self.transient)
        I = np.eye(n)
        
        try:
            N = np.linalg.inv(I - self.Q)
        except np.linalg.LinAlgError:
            # Near-singular, use pseudo-inverse
            N = np.linalg.pinv(I - self.Q)
        
        return N
    
    def absorption_probabilities(self) -> np.ndarray:
        """B = N × R gives absorption probabilities.
        
        B[i,j] = probability of being absorbed in state j,
                 starting from transient state i.
        """
        N = self.fundamental_matrix()
        return N @ self.R
    
    def expected_steps_to_absorption(self) -> np.ndarray:
        """Expected number of steps until absorption from each transient state.
        
        t = N × 1 (sum of expected visits to all transient states)
        """
        N = self.fundamental_matrix()
        return N.sum(axis=1)
    
    def variance_of_absorption_time(self) -> np.ndarray:
        """Variance in number of steps to absorption.
        
        var = (2N - I) × t - t² (element-wise square)
        """
        N = self.fundamental_matrix()
        t = self.expected_steps_to_absorption()
        
        n = len(self.transient)
        I = np.eye(n)
        
        # (2N - I) × t
        temp = (2 * N - I) @ t
        
        return temp - t ** 2
    
    def analyze(self) -> Dict[str, Any]:
        """Complete analysis of the chain."""
        N = self.fundamental_matrix()
        B = self.absorption_probabilities()
        t = self.expected_steps_to_absorption()
        var = self.variance_of_absorption_time()
        
        return {
            'fundamental_matrix': N,
            'absorption_probabilities': {
                self.transient[i]: {
                    self.absorbing[j]: B[i, j] 
                    for j in range(len(self.absorbing))
                }
                for i in range(len(self.transient))
            },
            'expected_steps': {
                self.transient[i]: t[i] 
                for i in range(len(self.transient))
            },
            'variance': {
                self.transient[i]: var[i]
                for i in range(len(self.transient))
            }
        }
```

---

## Chapter 4: Information Theory and Hidden Information

### 4.1 Information Content in Card Games

```python
"""
INFORMATION THEORY FOR CARD GAMES

Information = reduction in uncertainty
Entropy = measure of uncertainty

In card games:
- Hidden information creates strategic depth
- Perfect information = less interesting (often solved)
- Too much hidden info = feels random
- Sweet spot enables skill expression through inference
"""

class InformationTheory:
    """Information-theoretic analysis of card game states."""
    
    @staticmethod
    def entropy(probabilities: List[float]) -> float:
        """Shannon entropy: H = -Σ p_i log₂(p_i)
        
        Measures uncertainty in bits.
        
        Examples:
        - Fair coin: H = 1 bit
        - Fair die: H = log₂(6) ≈ 2.58 bits
        - Certainty (p=1): H = 0 bits
        """
        h = 0.0
        for p in probabilities:
            if p > 1e-10:  # Avoid log(0)
                h -= p * math.log2(p)
        return h
    
    @staticmethod
    def conditional_entropy(joint_probs: np.ndarray) -> float:
        """Conditional entropy H(Y|X).
        
        Uncertainty in Y given knowledge of X.
        
        joint_probs[i,j] = P(X=i, Y=j)
        """
        # H(Y|X) = H(X,Y) - H(X)
        # H(X,Y) = -Σᵢⱼ p(i,j) log p(i,j)
        # H(X) = -Σᵢ p(x_i) log p(x_i)
        
        joint_flat = joint_probs.flatten()
        h_xy = InformationTheory.entropy(joint_flat)
        
        p_x = joint_probs.sum(axis=1)
        h_x = InformationTheory.entropy(p_x)
        
        return h_xy - h_x
    
    @staticmethod
    def mutual_information(joint_probs: np.ndarray) -> float:
        """Mutual information I(X;Y).
        
        Information shared between X and Y.
        I(X;Y) = H(X) + H(Y) - H(X,Y) = H(Y) - H(Y|X)
        
        For card games: measures how much knowing X tells you about Y.
        """
        p_x = joint_probs.sum(axis=1)
        p_y = joint_probs.sum(axis=0)
        
        h_x = InformationTheory.entropy(p_x)
        h_y = InformationTheory.entropy(p_y)
        
        joint_flat = joint_probs.flatten()
        h_xy = InformationTheory.entropy(joint_flat)
        
        return h_x + h_y - h_xy


class HiddenInformationAnalyzer:
    """Analyze hidden information in game state."""
    
    def __init__(self, deck_size: int):
        self.deck_size = deck_size
    
    def deck_uncertainty(self, known_cards: List[str], 
                        card_pool: Dict[str, int]) -> float:
        """Entropy of opponent's deck composition.
        
        Args:
            known_cards: Cards we've seen opponent play
            card_pool: All possible cards {name: max_copies}
        
        Returns:
            Entropy in bits
        """
        # Count remaining possible cards
        remaining = {}
        for card, max_copies in card_pool.items():
            played = known_cards.count(card)
            remaining[card] = max_copies - played
        
        # Total remaining slots
        cards_seen = len(known_cards)
        slots = self.deck_size - cards_seen
        
        if slots <= 0:
            return 0.0  # Deck fully known
        
        # Calculate entropy of possible deck compositions
        # Simplified: assume uniform distribution over remaining
        total_remaining = sum(remaining.values())
        
        if total_remaining <= 0:
            return 0.0
        
        # Entropy of drawing from remaining pool
        probs = [count / total_remaining for count in remaining.values() if count > 0]
        
        # Multiply by number of slots for total uncertainty
        single_draw_entropy = InformationTheory.entropy(probs)
        
        # Approximate: slots × single_draw_entropy (ignoring correlations)
        return slots * single_draw_entropy
    
    def hand_uncertainty(self, hand_size: int, 
                        deck_composition: Dict[str, int]) -> float:
        """Entropy of opponent's current hand.
        
        What cards could they have?
        """
        total_cards = sum(deck_composition.values())
        
        if hand_size >= total_cards:
            return 0.0  # They have everything
        
        # Number of possible hands = C(total_cards, hand_size)
        # Entropy ≈ log₂(number of hands)
        num_hands = binomial(total_cards, hand_size)
        
        return math.log2(num_hands) if num_hands > 0 else 0.0
    
    def information_gain_from_play(self, played_card: str,
                                  prior_belief: Dict[str, float],
                                  play_probability: Dict[str, float]) -> float:
        """Information gained from observing a card play.
        
        Uses Bayesian update and measures reduction in entropy.
        
        Args:
            played_card: Card that was played
            prior_belief: P(card in deck) for each card before observation
            play_probability: P(play card | have card)
        
        Returns:
            Information gain in bits
        """
        # Prior entropy
        prior_probs = list(prior_belief.values())
        prior_entropy = InformationTheory.entropy(prior_probs)
        
        # Posterior entropy (after observing play)
        # Bayesian update: P(have card | played card) ∝ P(play | have) × P(have)
        posterior = {}
        for card, prior in prior_belief.items():
            if card == played_card:
                posterior[card] = 1.0  # Definitely have it
            else:
                # Update other cards based on the play
                # Simplified: assume play doesn't change other beliefs much
                posterior[card] = prior
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v/total for k, v in posterior.items()}
        
        posterior_entropy = InformationTheory.entropy(list(posterior.values()))
        
        return prior_entropy - posterior_entropy
    
    def optimal_information_play(self, hand: List[str],
                                opponent_uncertainty: Dict[str, float]
                               ) -> Tuple[str, float]:
        """Find play that reveals least information to opponent.
        
        From opponent's perspective, what play is most expected/boring?
        
        Args:
            hand: Our cards
            opponent_uncertainty: P(we have card) from opponent's view
        
        Returns:
            (best_card, information_revealed)
        """
        min_info = float('inf')
        best_card = hand[0] if hand else None
        
        for card in hand:
            # Information revealed = -log₂(P(play))
            # Play most expected card = reveal least
            prob = opponent_uncertainty.get(card, 0.5)
            
            if prob > 1e-10:
                info = -math.log2(prob)
            else:
                info = float('inf')  # Surprising play
            
            if info < min_info:
                min_info = info
                best_card = card
        
        return best_card, min_info
```

---

This concludes Part I of the Encyclopedia of Card Game Systems (Chapters 1-4). The full document continues with:

- **Part II**: Card Design Theory (anatomy, resources, synergy, power curves)
- **Part III**: Deck Mechanics (draw, discard, manipulation, zones)
- **Part IV**: Combat Systems (turns, damage, status, targeting)
- **Part V**: Balance Engineering (frameworks, testing, Monte Carlo, genetic tuning)
- **Part VI**: Content Generation (procedural cards, effects, flavor)
- **Part VII**: Roguelike Integration (runs, rewards, difficulty, meta)
- **Part VIII**: Implementation Patterns (definition languages, effect engines, AI)

---

*Continue to Part II: Card Design Theory...*
