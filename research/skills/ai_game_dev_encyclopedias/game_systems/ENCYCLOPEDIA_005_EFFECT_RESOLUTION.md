# ENCYCLOPEDIA OF EFFECT RESOLUTION ENGINES
## Volume V: Card Effect Implementation, State Machines, and Deterministic Game Logic
### Technical Reference for AI-Automated Card Game Development
### Version 1.0 | November 2025

---

# PREFACE

This encyclopedia provides exhaustive technical documentation for implementing robust card effect systems—the computational engine that transforms card text into game behavior. Written at research-laboratory standards, it covers effect parsing, resolution stacks, state management, and deterministic replay systems.

The goal: enable AI systems to generate mechanically correct card effects that resolve unambiguously in all edge cases while maintaining clean separation between effect definition (data) and execution (code).

---

# TABLE OF CONTENTS

## PART I: EFFECT SYSTEM ARCHITECTURE
- Chapter 1: Effect Definition Languages
- Chapter 2: Effect Type Taxonomy
- Chapter 3: Trigger Systems and Timing
- Chapter 4: Stack-Based Resolution

## PART II: STATE MANAGEMENT
- Chapter 5: Game State Representation
- Chapter 6: State Transitions and Validation
- Chapter 7: Undo/Redo Systems
- Chapter 8: Deterministic Replay

## PART III: TARGETING SYSTEMS
- Chapter 9: Target Selection Models
- Chapter 10: Area of Effect Patterns
- Chapter 11: Filter and Predicate Systems
- Chapter 12: Random Targeting and Seeding

## PART IV: DAMAGE AND COMBAT
- Chapter 13: Damage Calculation Pipeline
- Chapter 14: Damage Types and Modifiers
- Chapter 15: Combat Phase Implementation
- Chapter 16: Death and Cleanup Processing

## PART V: STATUS EFFECTS
- Chapter 17: Buff/Debuff Architecture
- Chapter 18: Duration and Decay Systems
- Chapter 19: Stacking Rules
- Chapter 20: Effect Interactions and Priority

## PART VI: RESOURCE SYSTEMS
- Chapter 21: Resource Pool Management
- Chapter 22: Cost Payment Systems
- Chapter 23: Resource Generation Effects
- Chapter 24: Resource Modification Chains

## PART VII: CARD MECHANICS
- Chapter 25: Draw and Discard Systems
- Chapter 26: Card Transformation Effects
- Chapter 27: Zone Movement Rules
- Chapter 28: Card Creation and Copying

## PART VIII: ADVANCED PATTERNS
- Chapter 29: AI-Emittable Effect DSL
- Chapter 30: Effect Validation Systems
- Chapter 31: Performance Optimization
- Chapter 32: Multiplayer Synchronization

---

# PART I: EFFECT SYSTEM ARCHITECTURE

---

## Chapter 1: Effect Definition Languages

### 1.1 Domain-Specific Languages for Card Effects

```python
"""
EFFECT DEFINITION LANGUAGES

Card effects must be defined in a format that is:
1. Human-readable (designers can understand)
2. Machine-parseable (engine can execute)
3. AI-generatable (LLMs can emit valid effects)
4. Validatable (can verify correctness before runtime)
5. Serializable (can save/load/transmit)

APPROACHES:

1. JSON/YAML Declarative
   + Easy to generate
   + Easy to validate
   - Limited expressiveness
   - Verbose for complex effects

2. Custom DSL
   + Natural syntax
   + Concise
   - Requires parser
   - AI needs examples

3. Code-as-Data (Lisp-like)
   + Fully expressive
   + Homoiconic
   - Unfamiliar to most
   - Security concerns

4. Hybrid (JSON + Expression Language)
   + Balance of simplicity and power
   + Safe subset for AI
   - Two syntaxes to learn
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum
import json
import re


class EffectType(Enum):
    """Categories of card effects."""
    DAMAGE = "damage"
    HEAL = "heal"
    BLOCK = "block"
    DRAW = "draw"
    DISCARD = "discard"
    ENERGY = "energy"
    STATUS = "status"
    SUMMON = "summon"
    TRANSFORM = "transform"
    MOVE_CARD = "move_card"
    MODIFY = "modify"
    TRIGGER = "trigger"
    CONDITIONAL = "conditional"
    COMPOSITE = "composite"


class TargetType(Enum):
    """Valid targeting modes."""
    SELF = "self"                   # The card's owner
    SINGLE_ENEMY = "single_enemy"   # One enemy (player chooses)
    ALL_ENEMIES = "all_enemies"     # Every enemy
    RANDOM_ENEMY = "random_enemy"   # Random enemy
    SINGLE_ALLY = "single_ally"     # One friendly unit
    ALL_ALLIES = "all_allies"       # All friendly units
    ALL = "all"                     # Everything
    NONE = "none"                   # No target (self-targeting only)
    CARD_IN_HAND = "card_in_hand"   # A card in hand
    CARD_IN_DECK = "card_in_deck"   # A card in deck
    CARD_IN_DISCARD = "card_in_discard"  # A card in discard


class TriggerTiming(Enum):
    """When effects can trigger."""
    ON_PLAY = "on_play"             # When card is played
    ON_DRAW = "on_draw"             # When card is drawn
    ON_DISCARD = "on_discard"       # When card is discarded
    ON_EXHAUST = "on_exhaust"       # When card is exhausted
    START_OF_TURN = "start_of_turn"
    END_OF_TURN = "end_of_turn"
    START_OF_COMBAT = "start_of_combat"
    END_OF_COMBAT = "end_of_combat"
    ON_DAMAGE_DEALT = "on_damage_dealt"
    ON_DAMAGE_TAKEN = "on_damage_taken"
    ON_DEATH = "on_death"
    ON_KILL = "on_kill"
    ON_BLOCK = "on_block"
    ON_STATUS_APPLIED = "on_status_applied"


@dataclass
class EffectDefinition:
    """Base structure for effect definitions.
    
    This is the core data format that AI will emit and the engine will execute.
    """
    effect_type: EffectType
    target: TargetType = TargetType.SELF
    
    # Common parameters (effect-type dependent)
    value: Union[int, str] = 0  # Static value or expression
    status_id: Optional[str] = None
    duration: int = 0
    
    # Conditions
    condition: Optional[str] = None  # Expression that must be true
    
    # Modifiers
    multiplier: float = 1.0
    additive_bonus: int = 0
    
    # Chaining
    then: Optional['EffectDefinition'] = None  # Effect to apply after
    
    # Metadata
    description_override: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        result = {
            "type": self.effect_type.value,
            "target": self.target.value,
            "value": self.value
        }
        
        if self.status_id:
            result["status"] = self.status_id
        if self.duration:
            result["duration"] = self.duration
        if self.condition:
            result["if"] = self.condition
        if self.multiplier != 1.0:
            result["multiplier"] = self.multiplier
        if self.additive_bonus:
            result["bonus"] = self.additive_bonus
        if self.then:
            result["then"] = self.then.to_dict()
        if self.description_override:
            result["description"] = self.description_override
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EffectDefinition':
        """Deserialize from dictionary."""
        effect = cls(
            effect_type=EffectType(data["type"]),
            target=TargetType(data.get("target", "self")),
            value=data.get("value", 0),
            status_id=data.get("status"),
            duration=data.get("duration", 0),
            condition=data.get("if"),
            multiplier=data.get("multiplier", 1.0),
            additive_bonus=data.get("bonus", 0)
        )
        
        if "then" in data:
            effect.then = cls.from_dict(data["then"])
        
        effect.description_override = data.get("description")
        
        return effect


class EffectDSL:
    """Domain-Specific Language for effect definitions.
    
    Provides a fluent API for building effects programmatically
    and a parser for text-based definitions.
    """
    
    @staticmethod
    def damage(value: Union[int, str], target: TargetType = TargetType.SINGLE_ENEMY
              ) -> EffectDefinition:
        """Create a damage effect."""
        return EffectDefinition(
            effect_type=EffectType.DAMAGE,
            target=target,
            value=value
        )
    
    @staticmethod
    def heal(value: Union[int, str], target: TargetType = TargetType.SELF
            ) -> EffectDefinition:
        """Create a healing effect."""
        return EffectDefinition(
            effect_type=EffectType.HEAL,
            target=target,
            value=value
        )
    
    @staticmethod
    def block(value: Union[int, str], target: TargetType = TargetType.SELF
             ) -> EffectDefinition:
        """Create a block effect."""
        return EffectDefinition(
            effect_type=EffectType.BLOCK,
            target=target,
            value=value
        )
    
    @staticmethod
    def draw(count: int = 1) -> EffectDefinition:
        """Create a draw effect."""
        return EffectDefinition(
            effect_type=EffectType.DRAW,
            target=TargetType.SELF,
            value=count
        )
    
    @staticmethod
    def apply_status(status_id: str, stacks: int = 1, 
                    duration: int = 0,
                    target: TargetType = TargetType.SELF) -> EffectDefinition:
        """Create a status application effect."""
        return EffectDefinition(
            effect_type=EffectType.STATUS,
            target=target,
            value=stacks,
            status_id=status_id,
            duration=duration
        )
    
    @staticmethod
    def gain_energy(amount: int = 1) -> EffectDefinition:
        """Create an energy gain effect."""
        return EffectDefinition(
            effect_type=EffectType.ENERGY,
            target=TargetType.SELF,
            value=amount
        )
    
    @staticmethod
    def discard(count: int = 1, random: bool = False) -> EffectDefinition:
        """Create a discard effect."""
        return EffectDefinition(
            effect_type=EffectType.DISCARD,
            target=TargetType.CARD_IN_HAND if not random else TargetType.NONE,
            value=count
        )
    
    @staticmethod
    def conditional(condition: str, if_true: EffectDefinition,
                   if_false: Optional[EffectDefinition] = None) -> EffectDefinition:
        """Create a conditional effect."""
        effect = EffectDefinition(
            effect_type=EffectType.CONDITIONAL,
            target=TargetType.NONE,
            condition=condition
        )
        effect.then = if_true
        # if_false would need additional handling
        return effect
    
    @staticmethod
    def composite(*effects: EffectDefinition) -> EffectDefinition:
        """Create a composite effect that applies multiple effects."""
        if not effects:
            raise ValueError("Composite effect requires at least one effect")
        
        # Chain effects together
        result = effects[0]
        current = result
        
        for effect in effects[1:]:
            current.then = effect
            current = effect
        
        return result


class EffectParser:
    """Parse effect definitions from various formats."""
    
    # Simple effect patterns for AI generation
    SIMPLE_PATTERNS = {
        r"deal (\d+) damage": lambda m: EffectDSL.damage(int(m.group(1))),
        r"deal (\d+) damage to all enemies": lambda m: EffectDSL.damage(int(m.group(1)), TargetType.ALL_ENEMIES),
        r"gain (\d+) block": lambda m: EffectDSL.block(int(m.group(1))),
        r"draw (\d+) cards?": lambda m: EffectDSL.draw(int(m.group(1))),
        r"draw a card": lambda m: EffectDSL.draw(1),
        r"heal (\d+) hp": lambda m: EffectDSL.heal(int(m.group(1))),
        r"gain (\d+) energy": lambda m: EffectDSL.gain_energy(int(m.group(1))),
        r"apply (\d+) (\w+)": lambda m: EffectDSL.apply_status(m.group(2).lower(), int(m.group(1))),
        r"apply (\w+)": lambda m: EffectDSL.apply_status(m.group(1).lower(), 1),
    }
    
    @classmethod
    def parse_simple(cls, text: str) -> Optional[EffectDefinition]:
        """Parse simple effect from natural language.
        
        Useful for AI-generated card text.
        """
        text = text.lower().strip()
        
        for pattern, builder in cls.SIMPLE_PATTERNS.items():
            match = re.match(pattern, text)
            if match:
                return builder(match)
        
        return None
    
    @classmethod
    def parse_json(cls, json_str: str) -> EffectDefinition:
        """Parse effect from JSON."""
        data = json.loads(json_str)
        return EffectDefinition.from_dict(data)
    
    @classmethod
    def parse_yaml(cls, yaml_str: str) -> EffectDefinition:
        """Parse effect from YAML."""
        import yaml
        data = yaml.safe_load(yaml_str)
        return EffectDefinition.from_dict(data)
    
    @classmethod
    def parse_compound(cls, text: str) -> List[EffectDefinition]:
        """Parse compound effect (multiple effects in one text).
        
        Handles patterns like:
        "Deal 6 damage. Draw 1 card."
        "Deal 3 damage to ALL enemies. Gain 2 Block."
        """
        effects = []
        
        # Split on sentence endings
        sentences = re.split(r'[.;]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                effect = cls.parse_simple(sentence)
                if effect:
                    effects.append(effect)
        
        return effects


class EffectValidator:
    """Validate effect definitions for correctness."""
    
    VALID_STATUSES = {
        "vulnerable", "weak", "frail", "strength", "dexterity",
        "poison", "burn", "bleed", "regeneration", "thorns",
        "artifact", "intangible", "barricade", "rage", "vigor"
    }
    
    @classmethod
    def validate(cls, effect: EffectDefinition) -> List[str]:
        """Validate an effect definition.
        
        Returns list of errors (empty if valid).
        """
        errors = []
        
        # Type-specific validation
        if effect.effect_type == EffectType.DAMAGE:
            errors.extend(cls._validate_damage(effect))
        elif effect.effect_type == EffectType.STATUS:
            errors.extend(cls._validate_status(effect))
        elif effect.effect_type == EffectType.DRAW:
            errors.extend(cls._validate_draw(effect))
        
        # Target validation
        if effect.effect_type in [EffectType.DAMAGE] and effect.target == TargetType.SELF:
            errors.append("Damage cannot target self")
        
        # Value validation
        if isinstance(effect.value, int) and effect.value < 0:
            errors.append(f"Negative value not allowed: {effect.value}")
        
        # Recursively validate chained effects
        if effect.then:
            errors.extend(cls.validate(effect.then))
        
        return errors
    
    @classmethod
    def _validate_damage(cls, effect: EffectDefinition) -> List[str]:
        """Validate damage-specific rules."""
        errors = []
        
        if isinstance(effect.value, int) and effect.value > 999:
            errors.append(f"Damage value suspiciously high: {effect.value}")
        
        return errors
    
    @classmethod
    def _validate_status(cls, effect: EffectDefinition) -> List[str]:
        """Validate status-specific rules."""
        errors = []
        
        if not effect.status_id:
            errors.append("Status effect requires status_id")
        elif effect.status_id not in cls.VALID_STATUSES:
            errors.append(f"Unknown status: {effect.status_id}")
        
        return errors
    
    @classmethod
    def _validate_draw(cls, effect: EffectDefinition) -> List[str]:
        """Validate draw-specific rules."""
        errors = []
        
        if isinstance(effect.value, int) and effect.value > 10:
            errors.append(f"Draw count suspiciously high: {effect.value}")
        
        return errors
```

### 1.2 Expression Language for Dynamic Values

```python
"""
EXPRESSION LANGUAGE FOR EFFECTS

Static values are simple but limiting.
An expression language enables dynamic effects:

- "damage = player_strength * 2"
- "draw = min(enemy_count, 3)"
- "block = cards_in_hand * 3"

The expression language must be:
1. Safe (no arbitrary code execution)
2. Deterministic (same input = same output)
3. Simple (AI can generate valid expressions)
4. Efficient (evaluated many times per game)
"""

import operator
import ast
from typing import Dict, Any, Callable


class SafeExpressionEvaluator:
    """Safe expression evaluation for card effects.
    
    Supports a limited set of operations:
    - Arithmetic: +, -, *, /, //, %, **
    - Comparison: ==, !=, <, >, <=, >=
    - Logic: and, or, not
    - Functions: min, max, abs, len, sum
    - Variables: from provided context
    """
    
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.Gt: operator.gt,
        ast.LtE: operator.le,
        ast.GtE: operator.ge,
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    ALLOWED_FUNCTIONS = {
        'min': min,
        'max': max,
        'abs': abs,
        'len': len,
        'sum': sum,
        'int': int,
        'float': float,
        'round': round,
        'clamp': lambda x, lo, hi: max(lo, min(hi, x)),
    }
    
    def __init__(self, context: Dict[str, Any] = None):
        """
        Args:
            context: Variable bindings available in expressions
        """
        self.context = context or {}
    
    def update_context(self, **kwargs):
        """Update context variables."""
        self.context.update(kwargs)
    
    def evaluate(self, expression: str) -> Any:
        """Safely evaluate an expression.
        
        Args:
            expression: Expression string to evaluate
        
        Returns:
            Result of evaluation
        
        Raises:
            ValueError: If expression contains disallowed constructs
        """
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        
        return self._eval_node(tree.body)
    
    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate an AST node."""
        
        # Numeric literals
        if isinstance(node, ast.Constant):
            return node.value
        
        # For older Python versions
        if isinstance(node, ast.Num):
            return node.n
        
        # Variable reference
        if isinstance(node, ast.Name):
            if node.id in self.context:
                return self.context[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        
        # Binary operations
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed operator: {op_type.__name__}")
            
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.ALLOWED_OPERATORS[op_type](left, right)
        
        # Comparison operations
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            
            for op, comparator in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in self.ALLOWED_OPERATORS:
                    raise ValueError(f"Disallowed comparison: {op_type.__name__}")
                
                right = self._eval_node(comparator)
                if not self.ALLOWED_OPERATORS[op_type](left, right):
                    return False
                left = right
            
            return True
        
        # Boolean operations
        if isinstance(node, ast.BoolOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed boolean op: {op_type.__name__}")
            
            values = [self._eval_node(v) for v in node.values]
            
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        
        # Unary operations
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed unary op: {op_type.__name__}")
            
            operand = self._eval_node(node.operand)
            return self.ALLOWED_OPERATORS[op_type](operand)
        
        # Function calls
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named function calls allowed")
            
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Disallowed function: {func_name}")
            
            args = [self._eval_node(arg) for arg in node.args]
            return self.ALLOWED_FUNCTIONS[func_name](*args)
        
        # Ternary conditional
        if isinstance(node, ast.IfExp):
            test = self._eval_node(node.test)
            if test:
                return self._eval_node(node.body)
            else:
                return self._eval_node(node.orelse)
        
        # List/tuple literals
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elt) for elt in node.elts]
        
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")
    
    def validate_expression(self, expression: str) -> List[str]:
        """Validate an expression without evaluating.
        
        Returns list of errors (empty if valid).
        """
        errors = []
        
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            return [f"Syntax error: {e}"]
        
        # Walk the tree and check each node
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # Note: we can't validate variables without context
                pass
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self.ALLOWED_FUNCTIONS:
                        errors.append(f"Unknown function: {node.func.id}")
                else:
                    errors.append("Only named function calls allowed")
            elif isinstance(node, ast.BinOp):
                if type(node.op) not in self.ALLOWED_OPERATORS:
                    errors.append(f"Disallowed operator: {type(node.op).__name__}")
        
        return errors


class EffectContext:
    """Context provider for effect evaluation.
    
    Provides all variables that can be referenced in effect expressions.
    """
    
    def __init__(self, game_state: 'GameState'):
        self.game_state = game_state
    
    def get_context(self, source: 'Entity' = None, 
                   target: 'Entity' = None) -> Dict[str, Any]:
        """Build context dictionary for expression evaluation.
        
        Args:
            source: Entity using the effect (e.g., player)
            target: Entity being targeted (e.g., enemy)
        
        Returns:
            Dictionary of all available variables
        """
        ctx = {}
        
        # Game state
        ctx['turn'] = self.game_state.turn_number
        ctx['round'] = self.game_state.round_number
        
        # Source entity (player)
        if source:
            ctx['hp'] = source.current_hp
            ctx['max_hp'] = source.max_hp
            ctx['missing_hp'] = source.max_hp - source.current_hp
            ctx['hp_percent'] = source.current_hp / source.max_hp if source.max_hp > 0 else 0
            ctx['block'] = source.block
            ctx['energy'] = source.energy
            ctx['strength'] = source.get_status_stacks('strength')
            ctx['dexterity'] = source.get_status_stacks('dexterity')
            
            # Hand/deck
            ctx['cards_in_hand'] = len(self.game_state.hand)
            ctx['cards_in_deck'] = len(self.game_state.deck)
            ctx['cards_in_discard'] = len(self.game_state.discard)
            ctx['cards_played_this_turn'] = self.game_state.cards_played_this_turn
            ctx['attacks_played_this_turn'] = self.game_state.attacks_played_this_turn
            ctx['skills_played_this_turn'] = self.game_state.skills_played_this_turn
        
        # Target entity (enemy)
        if target:
            ctx['target_hp'] = target.current_hp
            ctx['target_max_hp'] = target.max_hp
            ctx['target_block'] = target.block
            ctx['target_intent_damage'] = target.intent.damage if target.intent else 0
            ctx['target_poison'] = target.get_status_stacks('poison')
            ctx['target_vulnerable'] = target.has_status('vulnerable')
            ctx['target_weak'] = target.has_status('weak')
        
        # Enemies
        ctx['enemy_count'] = len(self.game_state.enemies)
        ctx['total_enemy_hp'] = sum(e.current_hp for e in self.game_state.enemies)
        
        # Combat stats
        ctx['damage_dealt_this_turn'] = self.game_state.damage_dealt_this_turn
        ctx['damage_taken_this_turn'] = self.game_state.damage_taken_this_turn
        ctx['times_blocked_this_turn'] = self.game_state.times_blocked_this_turn
        
        return ctx


class DynamicValue:
    """Wrapper for values that can be static or dynamic."""
    
    def __init__(self, value: Union[int, str]):
        """
        Args:
            value: Either static int or expression string
        """
        self.raw_value = value
        self.is_dynamic = isinstance(value, str)
        
        if self.is_dynamic:
            self.evaluator = SafeExpressionEvaluator()
            # Pre-validate
            errors = self.evaluator.validate_expression(value)
            if errors:
                raise ValueError(f"Invalid expression: {errors}")
    
    def resolve(self, context: Dict[str, Any]) -> int:
        """Resolve the value given a context.
        
        Args:
            context: Variable bindings
        
        Returns:
            Resolved integer value
        """
        if not self.is_dynamic:
            return self.raw_value
        
        self.evaluator.context = context
        result = self.evaluator.evaluate(self.raw_value)
        
        # Ensure integer result
        return int(result)
    
    def __repr__(self):
        if self.is_dynamic:
            return f"DynamicValue('{self.raw_value}')"
        return f"DynamicValue({self.raw_value})"


# Example: Card with dynamic damage
class ExampleCards:
    """Example card definitions using the effect system."""
    
    STRIKE = {
        "name": "Strike",
        "type": "attack",
        "cost": 1,
        "effects": [
            {"type": "damage", "target": "single_enemy", "value": 6}
        ],
        "description": "Deal 6 damage."
    }
    
    HEAVY_BLADE = {
        "name": "Heavy Blade",
        "type": "attack", 
        "cost": 2,
        "effects": [
            {"type": "damage", "target": "single_enemy", "value": "14 + strength * 2"}
        ],
        "description": "Deal 14 damage. Strength affects this card 3 times."
    }
    
    INFLAME = {
        "name": "Inflame",
        "type": "skill",
        "cost": 1,
        "effects": [
            {"type": "status", "target": "self", "status": "strength", "value": 2}
        ],
        "description": "Gain 2 Strength."
    }
    
    WHIRLWIND = {
        "name": "Whirlwind",
        "type": "attack",
        "cost": "X",  # Special: uses all energy
        "effects": [
            {
                "type": "damage",
                "target": "all_enemies",
                "value": "5 * energy"  # Scales with energy spent
            }
        ],
        "description": "Deal 5 damage to ALL enemies X times."
    }
    
    BODY_SLAM = {
        "name": "Body Slam",
        "type": "attack",
        "cost": 1,
        "effects": [
            {"type": "damage", "target": "single_enemy", "value": "block"}
        ],
        "description": "Deal damage equal to your current Block."
    }
    
    FEED = {
        "name": "Feed",
        "type": "attack",
        "cost": 1,
        "effects": [
            {"type": "damage", "target": "single_enemy", "value": 10},
            {
                "type": "conditional",
                "if": "target_hp <= 0",
                "then": {"type": "modify", "stat": "max_hp", "value": 3}
            }
        ],
        "description": "Deal 10 damage. If this kills an enemy, gain 3 Max HP."
    }
```

---

## Chapter 2: Effect Type Taxonomy

### 2.1 Complete Effect Classification

```python
"""
EFFECT TYPE TAXONOMY

A comprehensive classification of all possible card effects.
This taxonomy enables:
1. AI to understand the design space
2. Engine to handle all cases
3. Designers to communicate precisely
4. Validation to catch impossible combinations

HIERARCHY:

- RESOURCE EFFECTS
  - Damage (reduce HP)
  - Heal (restore HP)
  - Block (temporary shield)
  - Energy (action resource)
  - Gold (currency)

- CARD EFFECTS
  - Draw (add to hand)
  - Discard (remove from hand)
  - Exhaust (remove from run)
  - Create (generate new card)
  - Transform (replace card)
  - Upgrade (improve card)
  - Copy (duplicate card)

- STATUS EFFECTS
  - Buff (positive status)
  - Debuff (negative status)
  - Channel (ongoing effect)
  - Stance (mode change)

- TARGETING EFFECTS
  - Redirect (change target)
  - Spread (hit additional targets)
  - Chain (bounce between targets)

- META EFFECTS
  - Trigger (cause other effects)
  - Delay (effect happens later)
  - Cancel (negate effect)
  - Modify (change parameters)

- SPECIAL EFFECTS
  - Scry (look at cards)
  - Retain (keep in hand)
  - Ethereal (exhaust if not played)
  - Innate (start in hand)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum, auto


class EffectCategory(Enum):
    """High-level effect categories."""
    RESOURCE = auto()
    CARD = auto()
    STATUS = auto()
    TARGETING = auto()
    META = auto()
    SPECIAL = auto()


@dataclass
class EffectMetadata:
    """Metadata about an effect type."""
    name: str
    category: EffectCategory
    description: str
    
    # Targeting
    requires_target: bool
    valid_targets: List[TargetType]
    
    # Parameters
    required_params: List[str]
    optional_params: List[str]
    
    # Behavior
    can_be_modified: bool = True  # Can strength/etc affect it?
    can_be_blocked: bool = False  # Can block prevent it?
    can_be_dodged: bool = False   # Can be avoided?
    triggers_on_damage: bool = False  # Triggers damage reactions?
    
    # Validation
    min_value: Optional[int] = None
    max_value: Optional[int] = None


class EffectRegistry:
    """Registry of all effect types with metadata."""
    
    _effects: Dict[str, EffectMetadata] = {}
    
    @classmethod
    def register(cls, effect_type: str, metadata: EffectMetadata):
        """Register an effect type."""
        cls._effects[effect_type] = metadata
    
    @classmethod
    def get(cls, effect_type: str) -> Optional[EffectMetadata]:
        """Get metadata for an effect type."""
        return cls._effects.get(effect_type)
    
    @classmethod
    def all_of_category(cls, category: EffectCategory) -> List[str]:
        """Get all effect types in a category."""
        return [name for name, meta in cls._effects.items() 
                if meta.category == category]
    
    @classmethod
    def validate_effect(cls, effect: Dict[str, Any]) -> List[str]:
        """Validate an effect definition against registry."""
        errors = []
        
        effect_type = effect.get("type")
        if not effect_type:
            return ["Effect missing 'type' field"]
        
        metadata = cls.get(effect_type)
        if not metadata:
            return [f"Unknown effect type: {effect_type}"]
        
        # Check required parameters
        for param in metadata.required_params:
            if param not in effect:
                errors.append(f"Missing required parameter: {param}")
        
        # Check target
        if metadata.requires_target:
            target = effect.get("target")
            if not target:
                errors.append("Effect requires target")
            elif target not in [t.value for t in metadata.valid_targets]:
                errors.append(f"Invalid target type: {target}")
        
        # Check value bounds
        value = effect.get("value")
        if value is not None and isinstance(value, int):
            if metadata.min_value is not None and value < metadata.min_value:
                errors.append(f"Value {value} below minimum {metadata.min_value}")
            if metadata.max_value is not None and value > metadata.max_value:
                errors.append(f"Value {value} above maximum {metadata.max_value}")
        
        return errors


# Register all standard effects
def _register_standard_effects():
    """Register all standard effect types."""
    
    # === RESOURCE EFFECTS ===
    
    EffectRegistry.register("damage", EffectMetadata(
        name="Damage",
        category=EffectCategory.RESOURCE,
        description="Deal damage to a target, reduced by block",
        requires_target=True,
        valid_targets=[TargetType.SINGLE_ENEMY, TargetType.ALL_ENEMIES, 
                      TargetType.RANDOM_ENEMY],
        required_params=["value"],
        optional_params=["damage_type", "piercing", "lifesteal"],
        can_be_blocked=True,
        can_be_dodged=True,
        triggers_on_damage=True,
        min_value=0,
        max_value=999
    ))
    
    EffectRegistry.register("heal", EffectMetadata(
        name="Heal",
        category=EffectCategory.RESOURCE,
        description="Restore HP to a target",
        requires_target=True,
        valid_targets=[TargetType.SELF, TargetType.SINGLE_ALLY],
        required_params=["value"],
        optional_params=["overheal"],
        min_value=0,
        max_value=999
    ))
    
    EffectRegistry.register("block", EffectMetadata(
        name="Block",
        category=EffectCategory.RESOURCE,
        description="Gain temporary damage reduction",
        requires_target=True,
        valid_targets=[TargetType.SELF, TargetType.SINGLE_ALLY, TargetType.ALL_ALLIES],
        required_params=["value"],
        optional_params=["retain"],  # Does block persist between turns?
        min_value=0,
        max_value=999
    ))
    
    EffectRegistry.register("energy", EffectMetadata(
        name="Energy",
        category=EffectCategory.RESOURCE,
        description="Modify energy (action resource)",
        requires_target=False,
        valid_targets=[TargetType.SELF],
        required_params=["value"],
        optional_params=["temporary"],  # Only for this turn?
        min_value=-10,
        max_value=10
    ))
    
    # === CARD EFFECTS ===
    
    EffectRegistry.register("draw", EffectMetadata(
        name="Draw",
        category=EffectCategory.CARD,
        description="Draw cards from deck",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=["value"],
        optional_params=["filter"],  # Draw specific type?
        min_value=1,
        max_value=10
    ))
    
    EffectRegistry.register("discard", EffectMetadata(
        name="Discard",
        category=EffectCategory.CARD,
        description="Discard cards from hand",
        requires_target=False,
        valid_targets=[TargetType.CARD_IN_HAND, TargetType.NONE],  # NONE = random
        required_params=["value"],
        optional_params=["random", "filter"],
        min_value=1,
        max_value=10
    ))
    
    EffectRegistry.register("exhaust", EffectMetadata(
        name="Exhaust",
        category=EffectCategory.CARD,
        description="Remove card from combat permanently",
        requires_target=False,
        valid_targets=[TargetType.CARD_IN_HAND, TargetType.CARD_IN_DISCARD],
        required_params=[],
        optional_params=["value", "random", "filter"]
    ))
    
    EffectRegistry.register("create", EffectMetadata(
        name="Create",
        category=EffectCategory.CARD,
        description="Create a new card",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=["card_id"],  # What card to create
        optional_params=["destination", "upgraded", "count"],
    ))
    
    EffectRegistry.register("transform", EffectMetadata(
        name="Transform",
        category=EffectCategory.CARD,
        description="Replace a card with another",
        requires_target=False,
        valid_targets=[TargetType.CARD_IN_HAND, TargetType.CARD_IN_DECK],
        required_params=["into"],  # Transform into what
        optional_params=["random_from_pool"]
    ))
    
    # === STATUS EFFECTS ===
    
    EffectRegistry.register("status", EffectMetadata(
        name="Apply Status",
        category=EffectCategory.STATUS,
        description="Apply a status effect",
        requires_target=True,
        valid_targets=[TargetType.SELF, TargetType.SINGLE_ENEMY, 
                      TargetType.ALL_ENEMIES, TargetType.SINGLE_ALLY],
        required_params=["status"],
        optional_params=["value", "duration"]
    ))
    
    EffectRegistry.register("remove_status", EffectMetadata(
        name="Remove Status",
        category=EffectCategory.STATUS,
        description="Remove a status effect",
        requires_target=True,
        valid_targets=[TargetType.SELF, TargetType.SINGLE_ENEMY],
        required_params=["status"],
        optional_params=["value"]  # Remove specific amount
    ))
    
    EffectRegistry.register("transfer_status", EffectMetadata(
        name="Transfer Status",
        category=EffectCategory.STATUS,
        description="Move status from one entity to another",
        requires_target=True,
        valid_targets=[TargetType.SINGLE_ENEMY],
        required_params=["status", "from"],
        optional_params=["value"]
    ))
    
    # === META EFFECTS ===
    
    EffectRegistry.register("trigger", EffectMetadata(
        name="Trigger",
        category=EffectCategory.META,
        description="Trigger another effect or event",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=["event"],
        optional_params=["count"]
    ))
    
    EffectRegistry.register("conditional", EffectMetadata(
        name="Conditional",
        category=EffectCategory.META,
        description="Execute effect if condition met",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=["if", "then"],
        optional_params=["else"]
    ))
    
    EffectRegistry.register("repeat", EffectMetadata(
        name="Repeat",
        category=EffectCategory.META,
        description="Execute effect multiple times",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=["count", "effect"],
        optional_params=[]
    ))
    
    EffectRegistry.register("delay", EffectMetadata(
        name="Delay",
        category=EffectCategory.META,
        description="Schedule effect for later",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=["timing", "effect"],
        optional_params=[]
    ))
    
    # === SPECIAL EFFECTS ===
    
    EffectRegistry.register("scry", EffectMetadata(
        name="Scry",
        category=EffectCategory.SPECIAL,
        description="Look at top cards of deck",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=["value"],
        optional_params=["may_discard"]
    ))
    
    EffectRegistry.register("retain", EffectMetadata(
        name="Retain",
        category=EffectCategory.SPECIAL,
        description="Card doesn't discard at end of turn",
        requires_target=False,
        valid_targets=[TargetType.NONE],
        required_params=[],
        optional_params=["permanent"]
    ))
    
    EffectRegistry.register("upgrade_card", EffectMetadata(
        name="Upgrade Card",
        category=EffectCategory.SPECIAL,
        description="Upgrade a card",
        requires_target=False,
        valid_targets=[TargetType.CARD_IN_HAND, TargetType.CARD_IN_DECK],
        required_params=[],
        optional_params=["random", "count"]
    ))


# Initialize registry
_register_standard_effects()
```

---

## Chapter 3: Trigger Systems and Timing

### 3.1 Event-Driven Effect Triggers

```python
"""
TRIGGER SYSTEMS

Effects don't just happen when cards are played.
Many effects trigger in response to game events:

- "When you play an Attack, gain 1 Strength"
- "At the start of turn, draw 1 card"
- "Whenever you take damage, gain 1 energy"

A robust trigger system enables emergent gameplay through
unexpected synergies while remaining deterministic.

DESIGN PRINCIPLES:
1. Explicit ordering (no ambiguity)
2. Infinite loop prevention
3. Consistent evaluation timing
4. Clear trigger chaining rules
"""

from typing import List, Set, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq


class GameEvent(Enum):
    """All possible game events that can trigger effects."""
    
    # Turn structure
    COMBAT_START = auto()
    COMBAT_END = auto()
    TURN_START = auto()
    TURN_END = auto()
    ROUND_START = auto()
    ROUND_END = auto()
    
    # Card events
    CARD_DRAWN = auto()
    CARD_PLAYED = auto()
    CARD_DISCARDED = auto()
    CARD_EXHAUSTED = auto()
    CARD_CREATED = auto()
    CARD_TRANSFORMED = auto()
    CARD_UPGRADED = auto()
    
    # Combat events
    ATTACK_PLAYED = auto()
    SKILL_PLAYED = auto()
    POWER_PLAYED = auto()
    
    # Damage events
    DAMAGE_DEALT = auto()
    DAMAGE_TAKEN = auto()
    UNBLOCKED_DAMAGE_TAKEN = auto()
    HP_LOSS = auto()  # Different from damage (e.g., costs)
    
    # Defense events
    BLOCK_GAINED = auto()
    BLOCK_LOST = auto()
    BLOCK_BROKEN = auto()
    
    # Status events
    STATUS_APPLIED = auto()
    STATUS_REMOVED = auto()
    STATUS_TRIGGERED = auto()
    
    # Entity events
    ENTITY_DIED = auto()
    ENTITY_HEALED = auto()
    
    # Resource events
    ENERGY_GAINED = auto()
    ENERGY_SPENT = auto()
    GOLD_GAINED = auto()
    GOLD_SPENT = auto()
    
    # Special
    LETHAL_DAMAGE = auto()  # Damage that would kill
    OVERKILL = auto()       # Damage beyond lethal


@dataclass
class EventData:
    """Data associated with a game event."""
    event_type: GameEvent
    source: Optional[Any] = None      # What caused the event
    target: Optional[Any] = None      # What was affected
    value: int = 0                    # Associated value (damage amount, etc.)
    card: Optional[Any] = None        # Associated card if applicable
    status: Optional[str] = None      # Associated status if applicable
    
    # For modification
    modified_value: Optional[int] = None
    cancelled: bool = False
    
    # Metadata
    timestamp: int = 0
    chain_depth: int = 0  # How many triggers deep


@dataclass
class TriggerDefinition:
    """Definition of a trigger (when to activate an effect)."""
    
    trigger_id: str
    event: GameEvent
    
    # Source of the trigger
    owner: Any  # Entity that owns this trigger (relic, power, card, etc.)
    source_type: str  # "relic", "power", "card_in_play", "status"
    
    # Conditions
    condition: Optional[str] = None  # Expression to evaluate
    
    # Filtering
    source_filter: Optional[Callable] = None  # Filter on event source
    target_filter: Optional[Callable] = None  # Filter on event target
    
    # Effect to execute
    effect: 'EffectDefinition' = None
    
    # Execution parameters
    priority: int = 100  # Higher = earlier (default 100)
    limit_per_turn: Optional[int] = None
    limit_per_combat: Optional[int] = None
    
    # State
    times_triggered_this_turn: int = 0
    times_triggered_this_combat: int = 0
    
    def can_trigger(self, event_data: EventData, context: Dict) -> bool:
        """Check if trigger can activate for this event."""
        
        # Check limits
        if self.limit_per_turn and self.times_triggered_this_turn >= self.limit_per_turn:
            return False
        if self.limit_per_combat and self.times_triggered_this_combat >= self.limit_per_combat:
            return False
        
        # Check filters
        if self.source_filter and not self.source_filter(event_data.source):
            return False
        if self.target_filter and not self.target_filter(event_data.target):
            return False
        
        # Check condition
        if self.condition:
            evaluator = SafeExpressionEvaluator(context)
            try:
                if not evaluator.evaluate(self.condition):
                    return False
            except:
                return False
        
        return True
    
    def reset_turn_limit(self):
        """Reset turn-based trigger limit."""
        self.times_triggered_this_turn = 0
    
    def reset_combat_limit(self):
        """Reset combat-based trigger limit."""
        self.times_triggered_this_combat = 0
        self.times_triggered_this_turn = 0


class TriggerManager:
    """Manages all triggers and event processing.
    
    Responsible for:
    - Registering triggers
    - Processing events
    - Maintaining trigger order
    - Preventing infinite loops
    """
    
    MAX_CHAIN_DEPTH = 50  # Prevent infinite trigger chains
    
    def __init__(self):
        # Triggers organized by event type for fast lookup
        self._triggers: Dict[GameEvent, List[TriggerDefinition]] = {}
        for event in GameEvent:
            self._triggers[event] = []
        
        # Event queue for processing
        self._event_queue: List[EventData] = []
        self._processing: bool = False
        
        # Chain depth tracking
        self._current_chain_depth: int = 0
        
        # Event history for debugging/replay
        self._event_history: List[EventData] = []
    
    def register_trigger(self, trigger: TriggerDefinition):
        """Register a new trigger."""
        self._triggers[trigger.event].append(trigger)
        
        # Maintain priority order
        self._triggers[trigger.event].sort(key=lambda t: -t.priority)
    
    def unregister_trigger(self, trigger_id: str):
        """Remove a trigger by ID."""
        for event_type in self._triggers:
            self._triggers[event_type] = [
                t for t in self._triggers[event_type] 
                if t.trigger_id != trigger_id
            ]
    
    def unregister_by_owner(self, owner: Any):
        """Remove all triggers owned by an entity."""
        for event_type in self._triggers:
            self._triggers[event_type] = [
                t for t in self._triggers[event_type]
                if t.owner != owner
            ]
    
    def fire_event(self, event_data: EventData, game_state: 'GameState') -> EventData:
        """Fire an event and process all triggers.
        
        Args:
            event_data: The event to fire
            game_state: Current game state
        
        Returns:
            Potentially modified event data
        """
        # Track chain depth
        event_data.chain_depth = self._current_chain_depth
        
        # Check for infinite loop
        if self._current_chain_depth >= self.MAX_CHAIN_DEPTH:
            # Log warning, don't process
            print(f"WARNING: Max trigger chain depth reached at {event_data.event_type}")
            return event_data
        
        # Record event
        self._event_history.append(event_data)
        
        # Get applicable triggers
        triggers = self._triggers.get(event_data.event_type, [])
        
        # Build context for condition evaluation
        context = EffectContext(game_state).get_context(
            source=event_data.source,
            target=event_data.target
        )
        context['event_value'] = event_data.value
        
        # Process triggers in priority order
        self._current_chain_depth += 1
        
        try:
            for trigger in triggers:
                if event_data.cancelled:
                    break  # Event was cancelled
                
                if trigger.can_trigger(event_data, context):
                    # Execute the trigger's effect
                    self._execute_trigger(trigger, event_data, game_state)
                    
                    # Update trigger counts
                    trigger.times_triggered_this_turn += 1
                    trigger.times_triggered_this_combat += 1
        finally:
            self._current_chain_depth -= 1
        
        return event_data
    
    def _execute_trigger(self, trigger: TriggerDefinition, 
                        event_data: EventData, game_state: 'GameState'):
        """Execute a trigger's effect."""
        if trigger.effect:
            # The effect execution would go here
            # This would call into the EffectResolver
            pass
    
    def process_pending_events(self, game_state: 'GameState'):
        """Process any pending events in the queue."""
        while self._event_queue:
            event = self._event_queue.pop(0)
            self.fire_event(event, game_state)
    
    def reset_turn(self):
        """Reset turn-based trigger limits."""
        for triggers in self._triggers.values():
            for trigger in triggers:
                trigger.reset_turn_limit()
    
    def reset_combat(self):
        """Reset combat-based trigger limits."""
        for triggers in self._triggers.values():
            for trigger in triggers:
                trigger.reset_combat_limit()
        self._event_history.clear()


class TimingPhase(Enum):
    """Distinct timing phases within a turn."""
    
    # Turn start phases
    TURN_START_BEGIN = auto()      # Before anything
    TURN_START_DRAW = auto()       # Draw phase
    TURN_START_ENERGY = auto()     # Energy gain
    TURN_START_TRIGGERS = auto()   # Start of turn triggers
    TURN_START_END = auto()        # After start triggers
    
    # Main phase
    MAIN_PHASE = auto()            # Player can act
    
    # Turn end phases
    TURN_END_BEGIN = auto()        # Before cleanup
    TURN_END_TRIGGERS = auto()     # End of turn triggers
    TURN_END_DISCARD = auto()      # Discard hand
    TURN_END_STATUS = auto()       # Status tick-down
    TURN_END_BLOCK_DECAY = auto()  # Remove block
    TURN_END_END = auto()          # Final cleanup
    
    # Enemy turn
    ENEMY_TURN = auto()


class TurnManager:
    """Manages turn phases and timing."""
    
    def __init__(self, trigger_manager: TriggerManager):
        self.trigger_manager = trigger_manager
        self.current_phase = TimingPhase.MAIN_PHASE
        self._phase_handlers: Dict[TimingPhase, List[Callable]] = {}
    
    def register_phase_handler(self, phase: TimingPhase, handler: Callable):
        """Register a handler for a specific phase."""
        if phase not in self._phase_handlers:
            self._phase_handlers[phase] = []
        self._phase_handlers[phase].append(handler)
    
    def execute_turn_start(self, game_state: 'GameState'):
        """Execute all turn start phases."""
        phases = [
            TimingPhase.TURN_START_BEGIN,
            TimingPhase.TURN_START_DRAW,
            TimingPhase.TURN_START_ENERGY,
            TimingPhase.TURN_START_TRIGGERS,
            TimingPhase.TURN_START_END
        ]
        
        for phase in phases:
            self._execute_phase(phase, game_state)
        
        self.current_phase = TimingPhase.MAIN_PHASE
    
    def execute_turn_end(self, game_state: 'GameState'):
        """Execute all turn end phases."""
        phases = [
            TimingPhase.TURN_END_BEGIN,
            TimingPhase.TURN_END_TRIGGERS,
            TimingPhase.TURN_END_DISCARD,
            TimingPhase.TURN_END_STATUS,
            TimingPhase.TURN_END_BLOCK_DECAY,
            TimingPhase.TURN_END_END
        ]
        
        for phase in phases:
            self._execute_phase(phase, game_state)
        
        # Reset turn-based triggers
        self.trigger_manager.reset_turn()
    
    def _execute_phase(self, phase: TimingPhase, game_state: 'GameState'):
        """Execute a single phase."""
        self.current_phase = phase
        
        # Fire phase-specific event
        event_mapping = {
            TimingPhase.TURN_START_BEGIN: GameEvent.TURN_START,
            TimingPhase.TURN_END_BEGIN: GameEvent.TURN_END,
        }
        
        if phase in event_mapping:
            event_data = EventData(event_type=event_mapping[phase])
            self.trigger_manager.fire_event(event_data, game_state)
        
        # Execute registered handlers
        handlers = self._phase_handlers.get(phase, [])
        for handler in handlers:
            handler(game_state)
```

---

## Chapter 4: Stack-Based Resolution

### 4.1 Effect Stack Implementation

```python
"""
STACK-BASED EFFECT RESOLUTION

Like Magic: The Gathering's stack, effects resolve in LIFO order.
This enables:
- Responses to effects before they resolve
- Predictable ordering
- Counter-play opportunities

STACK RULES:
1. Effects are pushed to the stack
2. Each player can add more effects in response
3. When no one adds more, stack resolves top-to-bottom
4. Later effects resolve first (LIFO)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, List, Callable
from enum import Enum, auto


class StackItemType(Enum):
    """Types of items that can be on the stack."""
    CARD_PLAY = auto()      # Card being played
    TRIGGERED_EFFECT = auto()  # Effect from trigger
    ACTIVATED_ABILITY = auto()  # Manually activated ability
    STATUS_EFFECT = auto()  # Status applying damage/effect


@dataclass
class StackItem:
    """Item on the effect resolution stack."""
    
    item_type: StackItemType
    effect: 'EffectDefinition'
    source: Any  # What created this (card, relic, status, etc.)
    
    # Targeting
    targets: List[Any] = field(default_factory=list)
    
    # State
    resolved: bool = False
    cancelled: bool = False
    
    # For modification
    damage_modifier: int = 0
    block_modifier: int = 0
    
    # Metadata
    stack_order: int = 0  # When it was added to stack


class EffectStack:
    """LIFO stack for effect resolution."""
    
    def __init__(self, trigger_manager: TriggerManager):
        self._stack: Deque[StackItem] = deque()
        self._next_order: int = 0
        self.trigger_manager = trigger_manager
        
        # Hooks for external systems to observe/modify
        self._pre_resolve_hooks: List[Callable] = []
        self._post_resolve_hooks: List[Callable] = []
    
    def push(self, item: StackItem):
        """Add item to top of stack."""
        item.stack_order = self._next_order
        self._next_order += 1
        self._stack.append(item)
    
    def peek(self) -> Optional[StackItem]:
        """View top of stack without removing."""
        return self._stack[-1] if self._stack else None
    
    def pop(self) -> Optional[StackItem]:
        """Remove and return top of stack."""
        return self._stack.pop() if self._stack else None
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self._stack) == 0
    
    def cancel_item(self, stack_order: int):
        """Cancel a specific item on the stack."""
        for item in self._stack:
            if item.stack_order == stack_order:
                item.cancelled = True
                break
    
    def modify_item_damage(self, stack_order: int, modifier: int):
        """Modify damage of a stack item."""
        for item in self._stack:
            if item.stack_order == stack_order:
                item.damage_modifier += modifier
                break
    
    def resolve_all(self, game_state: 'GameState', resolver: 'EffectResolver'):
        """Resolve entire stack."""
        while not self.is_empty():
            item = self.peek()
            
            # Run pre-resolve hooks (can cancel or modify)
            for hook in self._pre_resolve_hooks:
                hook(item, game_state)
            
            # Pop and resolve if not cancelled
            item = self.pop()
            
            if not item.cancelled:
                resolver.resolve_stack_item(item, game_state)
            
            # Run post-resolve hooks
            for hook in self._post_resolve_hooks:
                hook(item, game_state)
            
            # Process any triggered effects
            # (they'll be added to the stack)
            self.trigger_manager.process_pending_events(game_state)
    
    def add_pre_resolve_hook(self, hook: Callable):
        """Add hook called before each resolution."""
        self._pre_resolve_hooks.append(hook)
    
    def add_post_resolve_hook(self, hook: Callable):
        """Add hook called after each resolution."""
        self._post_resolve_hooks.append(hook)
    
    def get_stack_preview(self) -> List[Dict]:
        """Get preview of current stack state for UI."""
        return [
            {
                "order": item.stack_order,
                "type": item.item_type.name,
                "source": str(item.source),
                "targets": [str(t) for t in item.targets],
                "cancelled": item.cancelled
            }
            for item in reversed(self._stack)  # Top-first for display
        ]


class EffectResolver:
    """Resolves effects from the stack.
    
    This is the core execution engine that turns effect definitions
    into game state changes.
    """
    
    def __init__(self, trigger_manager: TriggerManager):
        self.trigger_manager = trigger_manager
        self._effect_handlers: Dict[EffectType, Callable] = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for each effect type."""
        self._effect_handlers = {
            EffectType.DAMAGE: self._resolve_damage,
            EffectType.HEAL: self._resolve_heal,
            EffectType.BLOCK: self._resolve_block,
            EffectType.DRAW: self._resolve_draw,
            EffectType.DISCARD: self._resolve_discard,
            EffectType.ENERGY: self._resolve_energy,
            EffectType.STATUS: self._resolve_status,
            EffectType.CONDITIONAL: self._resolve_conditional,
        }
    
    def resolve_stack_item(self, item: StackItem, game_state: 'GameState'):
        """Resolve a single stack item."""
        effect = item.effect
        
        # Apply modifiers
        if item.damage_modifier and effect.effect_type == EffectType.DAMAGE:
            if isinstance(effect.value, int):
                effect.value += item.damage_modifier
        
        # Resolve the effect
        handler = self._effect_handlers.get(effect.effect_type)
        if handler:
            handler(effect, item.targets, game_state, item.source)
        
        # Resolve chained effects
        if effect.then:
            self.resolve_effect(effect.then, item.targets, game_state, item.source)
    
    def resolve_effect(self, effect: EffectDefinition, 
                      targets: List[Any], game_state: 'GameState',
                      source: Any = None):
        """Resolve an effect directly (not from stack)."""
        handler = self._effect_handlers.get(effect.effect_type)
        if handler:
            handler(effect, targets, game_state, source)
        
        if effect.then:
            self.resolve_effect(effect.then, targets, game_state, source)
    
    def _resolve_damage(self, effect: EffectDefinition, targets: List[Any],
                       game_state: 'GameState', source: Any):
        """Resolve damage effect."""
        # Calculate damage value
        context = EffectContext(game_state).get_context(source=source)
        
        if isinstance(effect.value, str):
            value = DynamicValue(effect.value).resolve(context)
        else:
            value = effect.value
        
        # Apply multiplier and bonus
        value = int(value * effect.multiplier + effect.additive_bonus)
        
        # Apply strength if from player
        if hasattr(source, 'get_status_stacks'):
            value += source.get_status_stacks('strength')
        
        # Deal damage to each target
        for target in targets:
            actual_damage = self._apply_damage_to_target(value, target, game_state, source)
            
            # Fire damage event
            event = EventData(
                event_type=GameEvent.DAMAGE_DEALT,
                source=source,
                target=target,
                value=actual_damage
            )
            self.trigger_manager.fire_event(event, game_state)
    
    def _apply_damage_to_target(self, damage: int, target: Any,
                               game_state: 'GameState', source: Any) -> int:
        """Apply damage to a single target."""
        # Check for vulnerable
        if hasattr(target, 'has_status') and target.has_status('vulnerable'):
            damage = int(damage * 1.5)
        
        # Check for weak on source
        if hasattr(source, 'has_status') and source.has_status('weak'):
            damage = int(damage * 0.75)
        
        # Apply block first
        blocked = min(damage, target.block)
        damage -= blocked
        target.block -= blocked
        
        if blocked > 0:
            event = EventData(
                event_type=GameEvent.BLOCK_LOST,
                target=target,
                value=blocked
            )
            self.trigger_manager.fire_event(event, game_state)
        
        # Apply remaining damage to HP
        if damage > 0:
            target.current_hp -= damage
            
            event = EventData(
                event_type=GameEvent.DAMAGE_TAKEN,
                source=source,
                target=target,
                value=damage
            )
            self.trigger_manager.fire_event(event, game_state)
            
            # Check for death
            if target.current_hp <= 0:
                target.current_hp = 0
                event = EventData(
                    event_type=GameEvent.ENTITY_DIED,
                    target=target,
                    source=source
                )
                self.trigger_manager.fire_event(event, game_state)
        
        return damage + blocked
    
    def _resolve_heal(self, effect: EffectDefinition, targets: List[Any],
                     game_state: 'GameState', source: Any):
        """Resolve healing effect."""
        context = EffectContext(game_state).get_context(source=source)
        
        if isinstance(effect.value, str):
            value = DynamicValue(effect.value).resolve(context)
        else:
            value = effect.value
        
        for target in targets:
            healed = min(value, target.max_hp - target.current_hp)
            target.current_hp += healed
            
            if healed > 0:
                event = EventData(
                    event_type=GameEvent.ENTITY_HEALED,
                    target=target,
                    value=healed
                )
                self.trigger_manager.fire_event(event, game_state)
    
    def _resolve_block(self, effect: EffectDefinition, targets: List[Any],
                      game_state: 'GameState', source: Any):
        """Resolve block gain effect."""
        context = EffectContext(game_state).get_context(source=source)
        
        if isinstance(effect.value, str):
            value = DynamicValue(effect.value).resolve(context)
        else:
            value = effect.value
        
        # Apply dexterity if from player
        if hasattr(source, 'get_status_stacks'):
            value += source.get_status_stacks('dexterity')
        
        value = int(value * effect.multiplier + effect.additive_bonus)
        
        for target in targets:
            target.block += value
            
            event = EventData(
                event_type=GameEvent.BLOCK_GAINED,
                target=target,
                value=value
            )
            self.trigger_manager.fire_event(event, game_state)
    
    def _resolve_draw(self, effect: EffectDefinition, targets: List[Any],
                     game_state: 'GameState', source: Any):
        """Resolve card draw effect."""
        count = effect.value if isinstance(effect.value, int) else 1
        
        for _ in range(count):
            if game_state.deck:
                card = game_state.deck.pop(0)
                game_state.hand.append(card)
                
                event = EventData(
                    event_type=GameEvent.CARD_DRAWN,
                    card=card
                )
                self.trigger_manager.fire_event(event, game_state)
            elif game_state.discard:
                # Shuffle discard into deck
                game_state.deck = game_state.discard.copy()
                game_state.discard.clear()
                game_state.shuffler.shuffle(game_state.deck)
                
                # Try again
                if game_state.deck:
                    card = game_state.deck.pop(0)
                    game_state.hand.append(card)
                    
                    event = EventData(
                        event_type=GameEvent.CARD_DRAWN,
                        card=card
                    )
                    self.trigger_manager.fire_event(event, game_state)
    
    def _resolve_discard(self, effect: EffectDefinition, targets: List[Any],
                        game_state: 'GameState', source: Any):
        """Resolve discard effect."""
        # targets should be cards to discard
        for card in targets:
            if card in game_state.hand:
                game_state.hand.remove(card)
                game_state.discard.append(card)
                
                event = EventData(
                    event_type=GameEvent.CARD_DISCARDED,
                    card=card
                )
                self.trigger_manager.fire_event(event, game_state)
    
    def _resolve_energy(self, effect: EffectDefinition, targets: List[Any],
                       game_state: 'GameState', source: Any):
        """Resolve energy gain/loss effect."""
        value = effect.value if isinstance(effect.value, int) else 1
        
        old_energy = game_state.energy
        game_state.energy = max(0, game_state.energy + value)
        
        change = game_state.energy - old_energy
        
        if change > 0:
            event = EventData(
                event_type=GameEvent.ENERGY_GAINED,
                value=change
            )
            self.trigger_manager.fire_event(event, game_state)
        elif change < 0:
            event = EventData(
                event_type=GameEvent.ENERGY_SPENT,
                value=-change
            )
            self.trigger_manager.fire_event(event, game_state)
    
    def _resolve_status(self, effect: EffectDefinition, targets: List[Any],
                       game_state: 'GameState', source: Any):
        """Resolve status application."""
        status_id = effect.status_id
        stacks = effect.value if isinstance(effect.value, int) else 1
        
        for target in targets:
            target.apply_status(status_id, stacks, effect.duration)
            
            event = EventData(
                event_type=GameEvent.STATUS_APPLIED,
                target=target,
                status=status_id,
                value=stacks
            )
            self.trigger_manager.fire_event(event, game_state)
    
    def _resolve_conditional(self, effect: EffectDefinition, targets: List[Any],
                            game_state: 'GameState', source: Any):
        """Resolve conditional effect."""
        context = EffectContext(game_state).get_context(source=source)
        
        if effect.condition:
            evaluator = SafeExpressionEvaluator(context)
            try:
                condition_met = evaluator.evaluate(effect.condition)
            except:
                condition_met = False
        else:
            condition_met = True
        
        if condition_met and effect.then:
            self.resolve_effect(effect.then, targets, game_state, source)
```

---

This concludes Part I of the Encyclopedia of Effect Resolution Engines (Chapters 1-4). The full document continues with:

- **Part II**: State Management (game state, transitions, undo, replay)
- **Part III**: Targeting Systems (selection, AoE, filters, random)
- **Part IV**: Damage and Combat (calculation, types, phases, death)
- **Part V**: Status Effects (buffs, durations, stacking, priority)
- **Part VI**: Resource Systems (pools, costs, generation, chains)
- **Part VII**: Card Mechanics (draw, transform, zones, creation)
- **Part VIII**: Advanced Patterns (AI DSL, validation, optimization, sync)

---

*Continue to Part II: State Management...*
