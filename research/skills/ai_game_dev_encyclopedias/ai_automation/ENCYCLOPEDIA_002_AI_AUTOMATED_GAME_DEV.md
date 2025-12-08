# ENCYCLOPEDIA OF AI-AUTOMATED GAME DEVELOPMENT
## Volume II: LLM-Driven Content Generation and Pipeline Architecture
### Technical Reference for AI-Human Fusion Development Systems
### Version 1.0 | November 2025

---

# PREFACE

This encyclopedia provides exhaustive technical documentation for building AI-automated game development pipelines where Large Language Models (LLMs) emit raw game files directly. Written at research-laboratory standards, it covers architecture patterns, file format specifications, validation systems, and production deployment strategies.

The goal: enable a "Pod" of AI systems (Claude, GPT-4, Gemini, Grok, DeepSeek) orchestrated by human direction to produce AAA-quality game content at unprecedented speed and scale.

---

# TABLE OF CONTENTS

## PART I: PIPELINE ARCHITECTURE
- Chapter 1: AI-Native Game Development Philosophy
- Chapter 2: File Emission vs SDK Compilation
- Chapter 3: Validation and Safety Layers
- Chapter 4: Multi-Model Orchestration (The Pod)

## PART II: ENGINE-AGNOSTIC FILE FORMATS
- Chapter 5: Scene Description Languages
- Chapter 6: Mesh and Model Formats
- Chapter 7: Material and Shader Definitions
- Chapter 8: Animation Data Structures
- Chapter 9: Audio Asset Specifications

## PART III: GODOT-SPECIFIC INTEGRATION
- Chapter 10: GDScript Emission Patterns
- Chapter 11: TSCN Scene File Format
- Chapter 12: Resource Files (.tres)
- Chapter 13: Project Configuration
- Chapter 14: Headless Export and CI/CD

## PART IV: WEB-NATIVE GAME ARCHITECTURE
- Chapter 15: Browser Runtime Considerations
- Chapter 16: Three.js/PixiJS/Phaser Integration
- Chapter 17: WebGL and WebGPU Programming
- Chapter 18: Electron/Tauri Desktop Packaging
- Chapter 19: Steamworks.js Integration

## PART V: CONTENT GENERATION SYSTEMS
- Chapter 20: Card Definition Languages
- Chapter 21: Encounter and Enemy Specification
- Chapter 22: Dialog and Narrative Trees
- Chapter 23: Procedural Level Grammar
- Chapter 24: Balance and Economy Models

## PART VI: LLM PROMPTING FOR GAME CONTENT
- Chapter 25: Structured Output Formats
- Chapter 26: JSON Schema Enforcement
- Chapter 27: Error Recovery and Validation
- Chapter 28: Iterative Refinement Loops
- Chapter 29: Multi-Agent Collaboration

## PART VII: TESTING AND QUALITY ASSURANCE
- Chapter 30: Automated Playtesting
- Chapter 31: Balance Simulation
- Chapter 32: Visual Regression Testing
- Chapter 33: Performance Profiling
- Chapter 34: Steam Review Prediction

## PART VIII: DEPLOYMENT AND LIVE OPERATIONS
- Chapter 35: Steam Early Access Launch
- Chapter 36: Update and Patch Pipelines
- Chapter 37: Telemetry and Analytics
- Chapter 38: A/B Testing Infrastructure
- Chapter 39: Community Feedback Integration

---

# PART I: PIPELINE ARCHITECTURE

---

## Chapter 1: AI-Native Game Development Philosophy

### 1.1 The Paradigm Shift

Traditional game development follows a human-centric pipeline:

```
TRADITIONAL PIPELINE:
Human Designer → Game Design Document → Asset Creation → Programming → 
QA Testing → Iteration → Release

Time: 2-5 years
Team: 10-500+ people
Cost: $1M-$500M
```

AI-native development inverts this model:

```
AI-NATIVE PIPELINE:
Human Vision → AI Generation → Validation → Curation → Release

Time: Weeks to months
Team: 1 human + AI Pod
Cost: $5K-$100K
```

### 1.2 Core Principles

```python
"""
AI-NATIVE GAME DEVELOPMENT PRINCIPLES

1. DECLARATIVE OVER IMPERATIVE
   Define WHAT, not HOW. Let AI figure out implementation.
   
   BAD:  "Write a function that iterates through cards and applies damage"
   GOOD: "Card: Fireball, Type: Attack, Damage: 8, Target: Single Enemy"

2. SCHEMA-FIRST DESIGN
   Define strict schemas before generating content.
   AI outputs must validate against schema or reject.
   
3. SEPARATION OF CONCERNS
   - Static engine code (human-maintained)
   - Generated content (AI-emitted, validated)
   - Runtime configuration (neither)

4. DETERMINISTIC VALIDATION
   Every AI output passes through validation layer.
   If validation fails, AI iterates until passing.
   No human intervention for mechanical errors.

5. HUMAN CURATION OVER CORRECTION
   Human role: select best outputs, set direction
   NOT: fix broken outputs, debug AI mistakes

6. FAIL-FAST GENERATION
   Generate many candidates quickly.
   Validate ruthlessly.
   Discard failures without sentiment.

7. VERSION EVERYTHING
   Every AI interaction logged.
   Every output version-controlled.
   Complete audit trail for debugging/improvement.

8. COMPOSITION OVER INHERITANCE
   AI-generated content combines via composition.
   No complex inheritance hierarchies.
   Easy to validate, replace, remix.
"""
```

### 1.3 The Pod Architecture

A "Pod" is a coordinated group of AI systems working on different aspects:

```python
from dataclasses import dataclass
from typing import Dict, List, Callable, Any
from enum import Enum
import json

class AIModel(Enum):
    """Available AI models in the Pod."""
    CLAUDE_OPUS = "claude-opus-4-5-20250514"
    CLAUDE_SONNET = "claude-sonnet-4-5-20250514"
    GPT4 = "gpt-4-turbo"
    GEMINI_PRO = "gemini-1.5-pro"
    GROK = "grok-2"
    DEEPSEEK = "deepseek-chat"
    LOCAL_PHI3 = "phi-3-mini"

@dataclass
class PodMember:
    """Configuration for a Pod member AI."""
    model: AIModel
    specialization: str
    strengths: List[str]
    cost_per_1k_tokens: float
    max_context: int
    supports_images: bool
    supports_code_execution: bool

class GameDevPod:
    """Orchestrates multiple AI models for game development.
    
    Each model has strengths:
    - Claude: Complex reasoning, long context, code generation
    - GPT-4: Creative writing, dialog, narrative
    - Gemini: Multimodal, visual understanding
    - Grok: Real-time data, trend analysis
    - DeepSeek: Cost-effective bulk generation
    - Local Phi-3: Privacy-sensitive, offline, fast iteration
    """
    
    def __init__(self):
        self.members: Dict[str, PodMember] = {
            "architect": PodMember(
                model=AIModel.CLAUDE_OPUS,
                specialization="System architecture, complex algorithms, code review",
                strengths=["reasoning", "code", "math", "long_context"],
                cost_per_1k_tokens=0.015,
                max_context=200000,
                supports_images=True,
                supports_code_execution=True
            ),
            "writer": PodMember(
                model=AIModel.GPT4,
                specialization="Narrative, dialog, flavor text, lore",
                strengths=["creativity", "tone", "character_voice"],
                cost_per_1k_tokens=0.01,
                max_context=128000,
                supports_images=True,
                supports_code_execution=True
            ),
            "artist": PodMember(
                model=AIModel.GEMINI_PRO,
                specialization="Visual description, art direction, asset specs",
                strengths=["multimodal", "visual_understanding", "style_analysis"],
                cost_per_1k_tokens=0.00125,
                max_context=1000000,
                supports_images=True,
                supports_code_execution=True
            ),
            "analyst": PodMember(
                model=AIModel.GROK,
                specialization="Market analysis, trend detection, competitive intel",
                strengths=["real_time", "social_analysis", "trend_prediction"],
                cost_per_1k_tokens=0.005,
                max_context=128000,
                supports_images=True,
                supports_code_execution=False
            ),
            "bulk_generator": PodMember(
                model=AIModel.DEEPSEEK,
                specialization="High-volume content generation, translation",
                strengths=["cost_efficiency", "bulk_output", "consistency"],
                cost_per_1k_tokens=0.0001,
                max_context=64000,
                supports_images=False,
                supports_code_execution=True
            ),
            "local_iterator": PodMember(
                model=AIModel.LOCAL_PHI3,
                specialization="Fast iteration, private data, offline work",
                strengths=["speed", "privacy", "offline", "unlimited_calls"],
                cost_per_1k_tokens=0.0,  # Local = free
                max_context=4096,
                supports_images=False,
                supports_code_execution=False
            )
        }
        
        self.task_routing: Dict[str, str] = {
            "system_design": "architect",
            "algorithm_implementation": "architect",
            "card_mechanics": "architect",
            "balance_formulas": "architect",
            "dialog_writing": "writer",
            "flavor_text": "writer",
            "lore_creation": "writer",
            "character_voice": "writer",
            "art_direction": "artist",
            "visual_specs": "artist",
            "ui_mockups": "artist",
            "market_research": "analyst",
            "competitor_analysis": "analyst",
            "bulk_cards": "bulk_generator",
            "localization": "bulk_generator",
            "quick_iteration": "local_iterator",
            "offline_work": "local_iterator"
        }
    
    def route_task(self, task_type: str) -> PodMember:
        """Route task to appropriate Pod member."""
        member_key = self.task_routing.get(task_type, "architect")
        return self.members[member_key]
    
    def estimate_cost(self, task_type: str, estimated_tokens: int) -> float:
        """Estimate cost for a task."""
        member = self.route_task(task_type)
        return (estimated_tokens / 1000) * member.cost_per_1k_tokens
    
    def parallel_generate(self, task: str, 
                         members: List[str] = None) -> Dict[str, Any]:
        """Generate content in parallel across multiple members.
        
        Used for: diverse perspectives, quality comparison, ensemble voting
        """
        if members is None:
            members = list(self.members.keys())
        
        results = {}
        for member_key in members:
            member = self.members[member_key]
            # In production: async API calls
            results[member_key] = {
                "model": member.model.value,
                "response": f"[{member_key} would generate here]",
                "cost": 0.0
            }
        
        return results
    
    def consensus_generate(self, task: str, 
                          voters: List[str] = None,
                          threshold: float = 0.6) -> Any:
        """Generate with consensus voting across models.
        
        Each model generates independently.
        Results compared/voted on.
        Consensus output returned if threshold met.
        """
        results = self.parallel_generate(task, voters)
        
        # In production: semantic similarity clustering
        # Return most common/agreed-upon output
        return results


class ContentPipeline:
    """Pipeline for AI-generated game content.
    
    Flow:
    1. Schema definition (human)
    2. Prompt construction (human + AI)
    3. Content generation (AI)
    4. Validation (automated)
    5. Curation (human)
    6. Integration (automated)
    7. Testing (automated + human)
    """
    
    def __init__(self, pod: GameDevPod):
        self.pod = pod
        self.schemas: Dict[str, dict] = {}
        self.validators: Dict[str, Callable] = {}
        self.generated_content: List[dict] = []
    
    def register_schema(self, content_type: str, schema: dict):
        """Register JSON schema for content type."""
        self.schemas[content_type] = schema
    
    def register_validator(self, content_type: str, 
                          validator: Callable[[dict], bool]):
        """Register custom validator for content type."""
        self.validators[content_type] = validator
    
    def generate(self, content_type: str, 
                requirements: dict,
                num_candidates: int = 5) -> List[dict]:
        """Generate content candidates.
        
        Args:
            content_type: Type of content (must have registered schema)
            requirements: Specific requirements for this generation
            num_candidates: Number of candidates to generate
        
        Returns:
            List of validated content objects
        """
        if content_type not in self.schemas:
            raise ValueError(f"No schema registered for {content_type}")
        
        schema = self.schemas[content_type]
        validator = self.validators.get(content_type, lambda x: True)
        
        candidates = []
        attempts = 0
        max_attempts = num_candidates * 3
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            
            # Generate candidate
            prompt = self._build_prompt(content_type, schema, requirements)
            member = self.pod.route_task(content_type)
            
            # In production: actual API call
            raw_output = self._call_model(member, prompt)
            
            # Parse and validate
            try:
                parsed = json.loads(raw_output)
                
                # Schema validation
                if not self._validate_schema(parsed, schema):
                    continue
                
                # Custom validation
                if not validator(parsed):
                    continue
                
                candidates.append(parsed)
                
            except json.JSONDecodeError:
                continue  # Malformed output, try again
        
        return candidates
    
    def _build_prompt(self, content_type: str, 
                     schema: dict, 
                     requirements: dict) -> str:
        """Build prompt for content generation."""
        return f"""Generate a {content_type} according to this schema:

SCHEMA:
{json.dumps(schema, indent=2)}

REQUIREMENTS:
{json.dumps(requirements, indent=2)}

OUTPUT INSTRUCTIONS:
- Return ONLY valid JSON matching the schema
- No markdown code blocks
- No explanatory text
- Must pass all schema constraints

JSON OUTPUT:"""
    
    def _call_model(self, member: PodMember, prompt: str) -> str:
        """Call AI model API."""
        # Placeholder - in production, actual API integration
        return "{}"
    
    def _validate_schema(self, data: dict, schema: dict) -> bool:
        """Validate data against JSON schema."""
        # Simplified validation - production would use jsonschema library
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        for field in required:
            if field not in data:
                return False
        
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return False
                if expected_type == "number" and not isinstance(value, (int, float)):
                    return False
                if expected_type == "integer" and not isinstance(value, int):
                    return False
                if expected_type == "array" and not isinstance(value, list):
                    return False
                if expected_type == "object" and not isinstance(value, dict):
                    return False
        
        return True
```

### 1.4 File Emission Architecture

```python
"""
FILE EMISSION ARCHITECTURE

The core innovation: AI emits raw text files that ARE the game.
No compilation step. No SDK. No binary formats.

SUPPORTED FILE TYPES:

1. SCENE FILES (.tscn, .json)
   - Complete scene graph
   - Node hierarchy
   - Component configurations
   - Resource references

2. SCRIPT FILES (.gd, .ts, .js, .py)
   - Game logic
   - AI behavior
   - UI controllers
   - Systems

3. DATA FILES (.json, .yaml, .toml)
   - Card definitions
   - Enemy stats
   - Dialog trees
   - Economy tuning

4. SHADER FILES (.gdshader, .glsl, .wgsl)
   - Visual effects
   - Post-processing
   - Material definitions

5. ASSET DESCRIPTORS (.tres, .json)
   - Material properties
   - Animation curves
   - Particle systems
   - Audio buses

EMISSION PATTERNS:

Pattern 1: ATOMIC EMISSION
- Each file independent
- No cross-file dependencies in single emission
- Easier validation
- More API calls

Pattern 2: BUNDLE EMISSION
- Related files emitted together
- Guaranteed consistency
- Fewer API calls
- Harder to validate partially

Pattern 3: TEMPLATE + DELTA
- Base templates human-authored
- AI emits modifications/additions
- Smaller outputs
- Requires merge logic

Pattern 4: INCREMENTAL REFINEMENT
- AI emits complete file
- Validation feedback
- AI emits corrected file
- Repeat until passing
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import hashlib

@dataclass
class EmittedFile:
    """Represents a file emitted by AI."""
    path: Path
    content: str
    content_type: str
    schema_version: str
    generator_model: str
    generation_id: str
    parent_generation: Optional[str]
    validation_status: str
    
    def content_hash(self) -> str:
        """SHA-256 hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    def save(self, base_dir: Path):
        """Save file to disk."""
        full_path = base_dir / self.path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(self.content)
    
    def to_metadata(self) -> dict:
        """Export metadata for tracking."""
        return {
            "path": str(self.path),
            "content_type": self.content_type,
            "schema_version": self.schema_version,
            "generator_model": self.generator_model,
            "generation_id": self.generation_id,
            "parent_generation": self.parent_generation,
            "validation_status": self.validation_status,
            "content_hash": self.content_hash(),
            "size_bytes": len(self.content.encode())
        }


class FileEmissionEngine:
    """Engine for AI file emission with validation and versioning."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.content_dir = project_root / "content"
        self.staging_dir = project_root / ".staging"
        self.archive_dir = project_root / ".archive"
        
        self.content_dir.mkdir(exist_ok=True)
        self.staging_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        self.validators: Dict[str, callable] = {}
        self.emission_history: List[EmittedFile] = []
    
    def register_validator(self, file_extension: str, 
                          validator: callable):
        """Register validator for file type."""
        self.validators[file_extension] = validator
    
    def emit(self, path: str, content: str, 
            metadata: Dict[str, Any] = None) -> EmittedFile:
        """Emit a file from AI.
        
        1. Create EmittedFile object
        2. Stage in .staging directory
        3. Run validators
        4. If passing, move to content directory
        5. Archive previous version if exists
        """
        file_path = Path(path)
        extension = file_path.suffix
        
        emitted = EmittedFile(
            path=file_path,
            content=content,
            content_type=extension,
            schema_version=metadata.get("schema_version", "1.0") if metadata else "1.0",
            generator_model=metadata.get("model", "unknown") if metadata else "unknown",
            generation_id=self._generate_id(),
            parent_generation=metadata.get("parent_id") if metadata else None,
            validation_status="pending"
        )
        
        # Stage file
        staging_path = self.staging_dir / emitted.generation_id / file_path
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.write_text(content)
        
        # Validate
        validator = self.validators.get(extension)
        if validator:
            is_valid, errors = validator(content, str(file_path))
            if not is_valid:
                emitted.validation_status = f"failed: {errors}"
                self.emission_history.append(emitted)
                return emitted
        
        emitted.validation_status = "passed"
        
        # Archive existing if present
        existing_path = self.content_dir / file_path
        if existing_path.exists():
            archive_path = self.archive_dir / emitted.generation_id / file_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            existing_path.rename(archive_path)
        
        # Move to content
        final_path = self.content_dir / file_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.rename(final_path)
        
        self.emission_history.append(emitted)
        return emitted
    
    def emit_bundle(self, files: Dict[str, str], 
                   metadata: Dict[str, Any] = None) -> List[EmittedFile]:
        """Emit multiple related files as a bundle.
        
        All-or-nothing: if any file fails validation,
        entire bundle is rejected.
        """
        bundle_id = self._generate_id()
        emitted_files = []
        all_valid = True
        
        # First pass: stage and validate all
        for path, content in files.items():
            file_meta = {**(metadata or {}), "bundle_id": bundle_id}
            emitted = EmittedFile(
                path=Path(path),
                content=content,
                content_type=Path(path).suffix,
                schema_version=file_meta.get("schema_version", "1.0"),
                generator_model=file_meta.get("model", "unknown"),
                generation_id=f"{bundle_id}_{Path(path).name}",
                parent_generation=file_meta.get("parent_id"),
                validation_status="pending"
            )
            
            # Stage
            staging_path = self.staging_dir / bundle_id / path
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            staging_path.write_text(content)
            
            # Validate
            validator = self.validators.get(emitted.content_type)
            if validator:
                is_valid, errors = validator(content, path)
                if not is_valid:
                    emitted.validation_status = f"failed: {errors}"
                    all_valid = False
                else:
                    emitted.validation_status = "passed"
            else:
                emitted.validation_status = "passed"
            
            emitted_files.append(emitted)
        
        # Second pass: commit or rollback
        if all_valid:
            for emitted in emitted_files:
                staging_path = self.staging_dir / bundle_id / str(emitted.path)
                final_path = self.content_dir / emitted.path
                
                # Archive existing
                if final_path.exists():
                    archive_path = self.archive_dir / bundle_id / str(emitted.path)
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    final_path.rename(archive_path)
                
                # Move to content
                final_path.parent.mkdir(parents=True, exist_ok=True)
                staging_path.rename(final_path)
        else:
            # Rollback: mark all as failed
            for emitted in emitted_files:
                if emitted.validation_status == "passed":
                    emitted.validation_status = "rollback: bundle failed"
        
        self.emission_history.extend(emitted_files)
        return emitted_files
    
    def _generate_id(self) -> str:
        """Generate unique ID for emission."""
        import uuid
        import time
        timestamp = int(time.time() * 1000)
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"
    
    def get_emission_report(self) -> dict:
        """Generate report of all emissions."""
        return {
            "total_emissions": len(self.emission_history),
            "passed": sum(1 for e in self.emission_history 
                         if e.validation_status == "passed"),
            "failed": sum(1 for e in self.emission_history 
                         if "failed" in e.validation_status),
            "by_type": self._group_by_type(),
            "recent": [e.to_metadata() for e in self.emission_history[-10:]]
        }
    
    def _group_by_type(self) -> Dict[str, int]:
        """Group emissions by content type."""
        counts = {}
        for e in self.emission_history:
            counts[e.content_type] = counts.get(e.content_type, 0) + 1
        return counts
```

---

## Chapter 2: File Emission vs SDK Compilation

### 2.1 The Traditional SDK Model

```
TRADITIONAL SDK MODEL:

Developer → IDE → SDK APIs → Compiler → Binary

Problems:
1. SDK lock-in (Unity, Unreal, etc.)
2. Binary opacity (can't inspect/modify)
3. Version dependencies (SDK updates break projects)
4. Licensing costs (runtime fees, seat licenses)
5. AI unfriendly (can't emit binaries)
```

### 2.2 The Direct Emission Model

```
DIRECT EMISSION MODEL:

AI → Text Files → Interpreter/Runtime → Game

Advantages:
1. Platform agnostic (any engine that reads text)
2. Fully inspectable (version control, diff, review)
3. No compilation (instant iteration)
4. Zero licensing (open formats)
5. AI native (LLMs emit text naturally)

File Format Requirements:
1. Human readable (text, not binary)
2. Self-describing (schema in file or standard)
3. Composable (reference other files)
4. Validatable (deterministic parsing)
5. Versionable (meaningful diffs)
```

### 2.3 Engine Compatibility Matrix

```python
"""
ENGINE COMPATIBILITY FOR DIRECT EMISSION

ENGINE          | FILE FORMATS           | EMISSION FRIENDLY | NOTES
----------------|------------------------|-------------------|------------------
Godot 4.x       | .gd, .tscn, .tres     | ★★★★★             | All text-based
Phaser 3.x      | .js/.ts, .json        | ★★★★★             | Web native
Three.js        | .js/.ts, .gltf        | ★★★★★             | Web native
PixiJS          | .js/.ts, .json        | ★★★★★             | 2D web native
Defold          | .script, .collection   | ★★★★☆             | Lua-based, text
LÖVE            | .lua                   | ★★★★★             | Pure Lua
Pygame          | .py                    | ★★★★★             | Pure Python
Unity           | .cs, .unity (yaml)     | ★★☆☆☆             | Requires compilation
Unreal          | .cpp, .uasset          | ★☆☆☆☆             | Binary assets
GameMaker       | .gml, .yy              | ★★★☆☆             | JSON-based

RECOMMENDED STACK FOR AI-NATIVE DEVELOPMENT:

Option A: Godot (Desktop-First)
- Engine: Godot 4.x
- Scripting: GDScript (text-based, Python-like)
- Scenes: .tscn (text-based scene format)
- Resources: .tres (text-based resource format)
- Export: Headless CI/CD possible

Option B: Web-Native (Cross-Platform)
- Runtime: Phaser 3.x or Three.js
- Scripting: TypeScript
- Data: JSON
- Desktop: Electron or Tauri
- Steam: steamworks.js
"""

@dataclass
class EngineCapability:
    """Engine capability assessment for AI emission."""
    name: str
    text_scene_format: bool
    text_script_format: bool
    headless_export: bool
    hot_reload: bool
    file_watch: bool
    ci_cd_friendly: bool
    steam_export: bool
    mobile_export: bool
    console_export: bool
    
    def emission_score(self) -> float:
        """Calculate overall emission friendliness score."""
        weights = {
            'text_scene_format': 0.25,
            'text_script_format': 0.20,
            'headless_export': 0.15,
            'hot_reload': 0.15,
            'file_watch': 0.10,
            'ci_cd_friendly': 0.15
        }
        
        score = 0.0
        for attr, weight in weights.items():
            if getattr(self, attr):
                score += weight
        
        return score

ENGINES = {
    "godot": EngineCapability(
        name="Godot 4.x",
        text_scene_format=True,
        text_script_format=True,
        headless_export=True,
        hot_reload=True,
        file_watch=True,
        ci_cd_friendly=True,
        steam_export=True,
        mobile_export=True,
        console_export=True  # With official support
    ),
    "phaser": EngineCapability(
        name="Phaser 3.x",
        text_scene_format=True,  # JSON configs
        text_script_format=True,  # JS/TS
        headless_export=True,  # Just bundle
        hot_reload=True,  # Vite/webpack
        file_watch=True,
        ci_cd_friendly=True,
        steam_export=True,  # Via Electron
        mobile_export=True,  # Capacitor/Cordova
        console_export=False
    ),
    "unity": EngineCapability(
        name="Unity",
        text_scene_format=False,  # YAML but brittle
        text_script_format=True,  # C#
        headless_export=True,  # Unity Build Server
        hot_reload=False,  # Requires domain reload
        file_watch=False,  # Manual refresh
        ci_cd_friendly=True,  # With Unity Build Server
        steam_export=True,
        mobile_export=True,
        console_export=True
    )
}
```

### 2.4 Validation Layer Architecture

```python
"""
VALIDATION LAYER ARCHITECTURE

Every AI-emitted file must pass through validation before integration.

VALIDATION LEVELS:

L0: SYNTAX
- File parses without error
- Valid JSON/YAML/GDScript/etc.
- Character encoding correct (UTF-8)

L1: SCHEMA
- Required fields present
- Field types correct
- Value ranges valid
- Enum values recognized

L2: SEMANTIC
- References resolve (other files exist)
- No circular dependencies
- Game logic coherent
- Balance sanity checks

L3: INTEGRATION
- Doesn't conflict with existing content
- Version compatibility
- Performance budgets met
- Memory limits respected

L4: PLAYTEST
- Game runs without crash
- Content is accessible
- No softlocks
- Difficulty appropriate
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import json
import re

class ValidationResult:
    """Result of a validation check."""
    
    def __init__(self, passed: bool, 
                 errors: List[str] = None,
                 warnings: List[str] = None):
        self.passed = passed
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __bool__(self):
        return self.passed
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge two validation results."""
        return ValidationResult(
            passed=self.passed and other.passed,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )


class Validator(ABC):
    """Abstract validator interface."""
    
    @property
    @abstractmethod
    def level(self) -> int:
        """Validation level (0-4)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name for logging."""
        pass
    
    @abstractmethod
    def validate(self, content: str, path: str) -> ValidationResult:
        """Validate content."""
        pass


class JSONSyntaxValidator(Validator):
    """L0: JSON syntax validation."""
    
    @property
    def level(self) -> int:
        return 0
    
    @property
    def name(self) -> str:
        return "JSON Syntax"
    
    def validate(self, content: str, path: str) -> ValidationResult:
        try:
            json.loads(content)
            return ValidationResult(True)
        except json.JSONDecodeError as e:
            return ValidationResult(False, [f"JSON parse error at line {e.lineno}: {e.msg}"])


class GDScriptSyntaxValidator(Validator):
    """L0: GDScript syntax validation.
    
    Basic validation without full Godot parser.
    Catches common errors AI makes.
    """
    
    @property
    def level(self) -> int:
        return 0
    
    @property
    def name(self) -> str:
        return "GDScript Syntax"
    
    def validate(self, content: str, path: str) -> ValidationResult:
        errors = []
        warnings = []
        lines = content.split('\n')
        
        # Check for common issues
        indent_stack = [0]
        in_multiline_string = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Track multiline strings
            triple_quote_count = line.count('"""')
            if triple_quote_count % 2 == 1:
                in_multiline_string = not in_multiline_string
            if in_multiline_string:
                continue
            
            # Check indentation (must be tabs or consistent spaces)
            if line and not line[0].isspace() and not any(
                stripped.startswith(kw) for kw in 
                ['class_name', 'extends', '@', 'const', 'enum', 'signal']
            ):
                # Non-indented line that's not a valid start
                if not stripped.startswith(('func ', 'var ', 'class ', 'static ')):
                    if '=' not in stripped and ':' not in stripped:
                        warnings.append(f"Line {i}: Suspicious unindented line")
            
            # Check for common Python-isms
            if 'self.' in line and 'self' not in line.split('(')[0]:
                errors.append(f"Line {i}: 'self.' is not valid GDScript (use no prefix)")
            
            if ' def ' in line:
                errors.append(f"Line {i}: Use 'func' not 'def' in GDScript")
            
            if 'True' in line or 'False' in line:
                errors.append(f"Line {i}: Use 'true'/'false' not 'True'/'False'")
            
            if 'None' in line:
                errors.append(f"Line {i}: Use 'null' not 'None'")
            
            if 'import ' in line and not line.strip().startswith('#'):
                errors.append(f"Line {i}: GDScript doesn't use 'import'")
            
            # Check for unclosed brackets on single lines
            if stripped.count('(') != stripped.count(')'):
                # Could be multiline, just warn
                warnings.append(f"Line {i}: Unbalanced parentheses")
            
            if stripped.count('[') != stripped.count(']'):
                warnings.append(f"Line {i}: Unbalanced brackets")
            
            if stripped.count('{') != stripped.count('}'):
                warnings.append(f"Line {i}: Unbalanced braces")
        
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class CardSchemaValidator(Validator):
    """L1: Card definition schema validation."""
    
    REQUIRED_FIELDS = ['id', 'name', 'type', 'cost']
    VALID_TYPES = ['attack', 'skill', 'power', 'status', 'curse']
    
    @property
    def level(self) -> int:
        return 1
    
    @property
    def name(self) -> str:
        return "Card Schema"
    
    def validate(self, content: str, path: str) -> ValidationResult:
        errors = []
        warnings = []
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return ValidationResult(False, ["Invalid JSON"])
        
        # Handle both single card and card list
        cards = data if isinstance(data, list) else [data]
        
        for i, card in enumerate(cards):
            prefix = f"Card {i}" if len(cards) > 1 else "Card"
            
            # Check required fields
            for field in self.REQUIRED_FIELDS:
                if field not in card:
                    errors.append(f"{prefix}: Missing required field '{field}'")
            
            # Validate types
            if 'type' in card and card['type'] not in self.VALID_TYPES:
                errors.append(f"{prefix}: Invalid type '{card['type']}'")
            
            # Validate cost
            if 'cost' in card:
                cost = card['cost']
                if not isinstance(cost, int) or cost < 0 or cost > 10:
                    errors.append(f"{prefix}: Cost must be integer 0-10")
            
            # Check for effects
            if 'effects' not in card and 'description' not in card:
                warnings.append(f"{prefix}: No effects or description defined")
            
            # Validate effects structure
            if 'effects' in card:
                for j, effect in enumerate(card['effects']):
                    if 'type' not in effect:
                        errors.append(f"{prefix} Effect {j}: Missing effect type")
                    if 'value' not in effect:
                        warnings.append(f"{prefix} Effect {j}: No value specified")
        
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class ValidationPipeline:
    """Pipeline that runs all validators in order."""
    
    def __init__(self):
        self.validators: Dict[str, List[Validator]] = {}
    
    def register(self, file_pattern: str, validator: Validator):
        """Register validator for file pattern."""
        if file_pattern not in self.validators:
            self.validators[file_pattern] = []
        
        # Insert in level order
        validators = self.validators[file_pattern]
        insert_idx = 0
        for i, v in enumerate(validators):
            if v.level > validator.level:
                break
            insert_idx = i + 1
        validators.insert(insert_idx, validator)
    
    def validate(self, content: str, path: str) -> ValidationResult:
        """Run all applicable validators."""
        result = ValidationResult(True)
        
        # Find matching validators
        for pattern, validators in self.validators.items():
            if self._matches(path, pattern):
                for validator in validators:
                    v_result = validator.validate(content, path)
                    result = result.merge(v_result)
                    
                    # Stop on L0 failure (can't validate higher levels)
                    if not v_result.passed and validator.level == 0:
                        return result
        
        return result
    
    def _matches(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern (simple glob)."""
        if pattern.startswith('*.'):
            return path.endswith(pattern[1:])
        return pattern in path


# Example usage
def create_game_validation_pipeline() -> ValidationPipeline:
    """Create validation pipeline for game content."""
    pipeline = ValidationPipeline()
    
    # JSON files
    pipeline.register('*.json', JSONSyntaxValidator())
    
    # Card definitions
    pipeline.register('cards/*.json', CardSchemaValidator())
    
    # GDScript
    pipeline.register('*.gd', GDScriptSyntaxValidator())
    
    return pipeline
```

---

## Chapter 3: Validation and Safety Layers

### 3.1 Defense in Depth

```python
"""
DEFENSE IN DEPTH FOR AI-GENERATED CODE

AI can generate malicious or buggy code. Multiple layers prevent damage:

LAYER 1: PROMPT ENGINEERING
- Explicit constraints in system prompt
- Output format enforcement
- Refusal training for dangerous patterns

LAYER 2: OUTPUT SANITIZATION
- Strip potentially dangerous constructs
- Sandbox file system access
- Limit network operations

LAYER 3: STATIC ANALYSIS
- AST parsing and analysis
- Pattern matching for dangerous calls
- Complexity limits

LAYER 4: RUNTIME SANDBOXING
- Process isolation
- Resource limits
- Capability restrictions

LAYER 5: MONITORING
- Anomaly detection
- Behavior logging
- Kill switches
"""

import ast
import re
from typing import Set

class CodeSanitizer:
    """Sanitize AI-generated code before execution."""
    
    # Dangerous patterns to block
    DANGEROUS_IMPORTS = {
        'os', 'subprocess', 'sys', 'shutil', 'socket', 
        'requests', 'urllib', 'http', 'ftplib', 'smtplib',
        'pickle', 'marshal', 'shelve',
        'ctypes', 'cffi',
        '__builtins__', 'builtins',
        'code', 'codeop', 'compile',
        'importlib', '__import__'
    }
    
    DANGEROUS_CALLS = {
        'eval', 'exec', 'compile', 'execfile',
        'open', 'file', 'input', 'raw_input',
        '__import__', 'reload',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'type', 'isinstance', 'issubclass'
    }
    
    DANGEROUS_PATTERNS = [
        r'__\w+__',  # Dunder methods
        r'lambda.*:.*exec',  # Lambda with exec
        r'\.read\s*\(',  # File reading
        r'\.write\s*\(',  # File writing
        r'subprocess',  # Process spawning
        r'os\.system',  # Shell commands
        r'socket\.',  # Network access
    ]
    
    def __init__(self, allowed_imports: Set[str] = None):
        self.allowed_imports = allowed_imports or {'math', 'random', 'json', 'typing'}
    
    def sanitize_python(self, code: str) -> Tuple[str, List[str]]:
        """Sanitize Python code.
        
        Returns: (sanitized_code, list_of_issues)
        """
        issues = []
        
        # Check for dangerous patterns via regex
        for pattern in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                issues.append(f"Dangerous pattern: {pattern}")
        
        # Parse AST and analyze
        try:
            tree = ast.parse(code)
            issues.extend(self._analyze_ast(tree))
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            return code, issues
        
        # If issues found, attempt to sanitize
        if issues:
            sanitized = self._remove_dangerous(code)
            return sanitized, issues
        
        return code, []
    
    def _analyze_ast(self, tree: ast.AST) -> List[str]:
        """Analyze AST for dangerous constructs."""
        issues = []
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in self.DANGEROUS_IMPORTS:
                        if alias.name not in self.allowed_imports:
                            issues.append(f"Dangerous import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.DANGEROUS_IMPORTS:
                    if node.module not in self.allowed_imports:
                        issues.append(f"Dangerous import from: {node.module}")
            
            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_CALLS:
                        issues.append(f"Dangerous call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.DANGEROUS_CALLS:
                        issues.append(f"Dangerous method: {node.func.attr}")
            
            # Check for exec/eval in any form
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name) and func.id in ('exec', 'eval'):
                        issues.append(f"Dangerous expression: {func.id}")
        
        return issues
    
    def _remove_dangerous(self, code: str) -> str:
        """Remove dangerous constructs from code."""
        lines = code.split('\n')
        sanitized_lines = []
        
        for line in lines:
            is_dangerous = False
            
            # Check imports
            if line.strip().startswith(('import ', 'from ')):
                for dangerous in self.DANGEROUS_IMPORTS:
                    if dangerous in line:
                        is_dangerous = True
                        break
            
            # Check dangerous calls
            for call in self.DANGEROUS_CALLS:
                if f'{call}(' in line:
                    is_dangerous = True
                    break
            
            if is_dangerous:
                sanitized_lines.append(f"# SANITIZED: {line}")
            else:
                sanitized_lines.append(line)
        
        return '\n'.join(sanitized_lines)
    
    def sanitize_gdscript(self, code: str) -> Tuple[str, List[str]]:
        """Sanitize GDScript code.
        
        GDScript is inherently sandboxed, but some patterns are still risky.
        """
        issues = []
        
        dangerous_gdscript = [
            r'OS\.execute',  # Running external programs
            r'OS\.shell_open',  # Opening URLs/files
            r'File\.new\(\)',  # File access (old API)
            r'FileAccess\.open',  # File access (new API)
            r'DirAccess\.',  # Directory access
            r'HTTPRequest',  # Network requests
            r'JavaScript\.',  # JavaScript bridge
            r'JavaScriptBridge\.',  # JS bridge (Godot 4)
        ]
        
        for pattern in dangerous_gdscript:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Potentially dangerous GDScript: {pattern}")
        
        return code, issues


class ResourceLimiter:
    """Limit resources for AI-generated code execution."""
    
    def __init__(self, 
                 max_memory_mb: int = 256,
                 max_cpu_seconds: float = 5.0,
                 max_output_bytes: int = 1024 * 1024):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        self.max_output_bytes = max_output_bytes
    
    def create_sandbox_config(self) -> dict:
        """Create configuration for sandboxed execution."""
        return {
            'memory_limit': self.max_memory_mb * 1024 * 1024,
            'cpu_limit': self.max_cpu_seconds,
            'output_limit': self.max_output_bytes,
            'network': False,
            'filesystem': 'read_only',
            'allowed_paths': ['/tmp/sandbox'],
            'environment': {
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONUNBUFFERED': '1'
            }
        }
    
    def execute_sandboxed(self, code: str, language: str = 'python') -> dict:
        """Execute code in sandbox.
        
        In production, this would use:
        - Docker containers
        - seccomp/AppArmor
        - cgroups for resource limits
        - nsjail or firejail
        """
        config = self.create_sandbox_config()
        
        # Placeholder - actual implementation depends on platform
        result = {
            'success': False,
            'output': '',
            'error': 'Sandbox execution not implemented',
            'resources': {
                'memory_peak_mb': 0,
                'cpu_seconds': 0,
                'output_bytes': 0
            }
        }
        
        return result
```

### 3.2 Content Safety

```python
"""
CONTENT SAFETY FOR AI-GENERATED GAME CONTENT

Games have specific content concerns:
- Age-appropriate content
- Cultural sensitivity
- Legal compliance (gambling, etc.)
- Brand safety

AI must be guided to generate appropriate content.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class ContentRating(Enum):
    """ESRB-style content ratings."""
    EVERYONE = "E"
    EVERYONE_10 = "E10+"
    TEEN = "T"
    MATURE = "M"
    ADULTS_ONLY = "AO"

@dataclass
class ContentGuidelines:
    """Guidelines for content generation."""
    target_rating: ContentRating
    allowed_themes: List[str]
    forbidden_themes: List[str]
    violence_level: str  # 'none', 'cartoon', 'fantasy', 'realistic'
    language_filter: bool
    cultural_notes: List[str]
    legal_restrictions: List[str]

class ContentModerator:
    """Moderate AI-generated content for appropriateness."""
    
    # Patterns that indicate content issues
    VIOLENCE_PATTERNS = {
        'mild': [r'\b(hit|punch|kick|fight)\b'],
        'moderate': [r'\b(blood|wound|injur)\w*\b'],
        'severe': [r'\b(kill|murder|death|die|dead)\b', r'\b(gore|dismember)\w*\b']
    }
    
    LANGUAGE_PATTERNS = {
        'mild': [r'\b(damn|hell|crap)\b'],
        'moderate': [r'\b(ass|bastard)\b'],  # Simplified
        'severe': []  # Would contain stronger terms
    }
    
    GAMBLING_PATTERNS = [
        r'\b(gambl|bet|wager|casino|slot)\w*\b',
        r'\b(loot\s*box|gacha)\b',
        r'\bpaid\s+random\b'
    ]
    
    def __init__(self, guidelines: ContentGuidelines):
        self.guidelines = guidelines
    
    def moderate(self, content: str, content_type: str) -> dict:
        """Moderate content and return analysis."""
        issues = []
        warnings = []
        
        # Check violence levels
        violence_level = self._check_violence(content)
        if self._violence_exceeds_rating(violence_level):
            issues.append(f"Violence level '{violence_level}' exceeds rating")
        
        # Check language
        if self.guidelines.language_filter:
            language_issues = self._check_language(content)
            issues.extend(language_issues)
        
        # Check gambling (important for mobile/certain regions)
        if self._check_gambling(content):
            warnings.append("Content may contain gambling-related elements")
        
        # Check forbidden themes
        for theme in self.guidelines.forbidden_themes:
            if theme.lower() in content.lower():
                issues.append(f"Forbidden theme detected: {theme}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'analysis': {
                'violence_level': violence_level,
                'word_count': len(content.split()),
                'content_type': content_type
            }
        }
    
    def _check_violence(self, content: str) -> str:
        """Determine violence level in content."""
        content_lower = content.lower()
        
        for level in ['severe', 'moderate', 'mild']:
            for pattern in self.VIOLENCE_PATTERNS.get(level, []):
                if re.search(pattern, content_lower):
                    return level
        
        return 'none'
    
    def _violence_exceeds_rating(self, violence_level: str) -> bool:
        """Check if violence exceeds target rating."""
        rating = self.guidelines.target_rating
        
        limits = {
            ContentRating.EVERYONE: 'none',
            ContentRating.EVERYONE_10: 'mild',
            ContentRating.TEEN: 'moderate',
            ContentRating.MATURE: 'severe',
            ContentRating.ADULTS_ONLY: 'severe'
        }
        
        levels = ['none', 'mild', 'moderate', 'severe']
        max_level = limits.get(rating, 'moderate')
        
        return levels.index(violence_level) > levels.index(max_level)
    
    def _check_language(self, content: str) -> List[str]:
        """Check for inappropriate language."""
        issues = []
        content_lower = content.lower()
        
        for level in ['severe', 'moderate']:
            for pattern in self.LANGUAGE_PATTERNS.get(level, []):
                if re.search(pattern, content_lower):
                    issues.append(f"Language issue ({level})")
                    break
        
        return issues
    
    def _check_gambling(self, content: str) -> bool:
        """Check for gambling-related content."""
        content_lower = content.lower()
        
        for pattern in self.GAMBLING_PATTERNS:
            if re.search(pattern, content_lower):
                return True
        
        return False


# Example guidelines for different game types
DECKBUILDER_GUIDELINES = ContentGuidelines(
    target_rating=ContentRating.TEEN,
    allowed_themes=['fantasy', 'magic', 'adventure', 'psychological', 'tarot', 'mystical'],
    forbidden_themes=['real_world_religion', 'real_politics', 'sexual_content'],
    violence_level='fantasy',
    language_filter=True,
    cultural_notes=['Respect diverse spiritual traditions', 'Avoid cultural appropriation'],
    legal_restrictions=['No real-money gambling mechanics', 'Disclose odds for any random rewards']
)
```

---

## Chapter 4: Multi-Model Orchestration (The Pod)

### 4.1 Orchestration Patterns

```python
"""
POD ORCHESTRATION PATTERNS

Different tasks benefit from different orchestration strategies.

PATTERN 1: SPECIALIST ROUTING
Route each task to the model best suited for it.
Pros: Optimal quality per task
Cons: Requires knowing which model is best

PATTERN 2: ENSEMBLE VOTING
Multiple models generate, best output selected.
Pros: Higher quality through diversity
Cons: Higher cost (N× API calls)

PATTERN 3: PIPELINE HANDOFF
Model A generates, Model B refines, Model C validates.
Pros: Specialized stages, iterative improvement
Cons: Error propagation, latency

PATTERN 4: PARALLEL GENERATION
All models generate in parallel, results merged.
Pros: Fast, comprehensive
Cons: Merge complexity, consistency issues

PATTERN 5: HIERARCHICAL DELEGATION
Orchestrator model delegates subtasks to specialists.
Pros: Flexible, scalable
Cons: Orchestrator overhead, coordination complexity
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
import json

class OrchestrationPattern(Enum):
    SPECIALIST = "specialist"
    ENSEMBLE = "ensemble"
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"

@dataclass
class TaskResult:
    """Result from a Pod task."""
    task_id: str
    model: str
    output: Any
    tokens_used: int
    latency_ms: float
    success: bool
    error: Optional[str] = None

@dataclass
class PodTask:
    """Task to be executed by the Pod."""
    task_id: str
    task_type: str
    prompt: str
    schema: Optional[dict] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferred_model: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: float = 60.0

class PodOrchestrator:
    """Orchestrates tasks across multiple AI models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}  # Model clients
        self.task_history: List[TaskResult] = []
        self.model_stats: Dict[str, Dict[str, float]] = {}
    
    def register_model(self, name: str, client: Any, capabilities: List[str]):
        """Register a model with the orchestrator."""
        self.models[name] = {
            'client': client,
            'capabilities': capabilities
        }
        self.model_stats[name] = {
            'total_tasks': 0,
            'success_rate': 1.0,
            'avg_latency_ms': 0,
            'avg_quality': 0.5
        }
    
    async def execute(self, task: PodTask, 
                     pattern: OrchestrationPattern = OrchestrationPattern.SPECIALIST
                     ) -> TaskResult:
        """Execute task using specified orchestration pattern."""
        
        if pattern == OrchestrationPattern.SPECIALIST:
            return await self._execute_specialist(task)
        elif pattern == OrchestrationPattern.ENSEMBLE:
            return await self._execute_ensemble(task)
        elif pattern == OrchestrationPattern.PIPELINE:
            return await self._execute_pipeline(task)
        elif pattern == OrchestrationPattern.PARALLEL:
            return await self._execute_parallel(task)
        elif pattern == OrchestrationPattern.HIERARCHICAL:
            return await self._execute_hierarchical(task)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    async def _execute_specialist(self, task: PodTask) -> TaskResult:
        """Route to single best model."""
        model_name = self._select_model(task)
        return await self._call_model(model_name, task)
    
    async def _execute_ensemble(self, task: PodTask, 
                               num_models: int = 3) -> TaskResult:
        """Get outputs from multiple models, select best."""
        # Select top N models for this task type
        models = self._rank_models(task.task_type)[:num_models]
        
        # Call all in parallel
        tasks = [self._call_model(m, task) for m in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful = [r for r in results if isinstance(r, TaskResult) and r.success]
        
        if not successful:
            return TaskResult(
                task_id=task.task_id,
                model="ensemble",
                output=None,
                tokens_used=sum(r.tokens_used for r in results if isinstance(r, TaskResult)),
                latency_ms=max(r.latency_ms for r in results if isinstance(r, TaskResult)),
                success=False,
                error="All models failed"
            )
        
        # Select best output (simplified - would use quality scoring)
        best = max(successful, key=lambda r: self._score_output(r, task))
        best.model = f"ensemble({best.model})"
        return best
    
    async def _execute_pipeline(self, task: PodTask) -> TaskResult:
        """Execute through pipeline of models."""
        # Define pipeline stages
        pipeline = [
            ('generator', self._select_model(task)),
            ('refiner', 'claude'),
            ('validator', 'claude')
        ]
        
        current_output = None
        total_tokens = 0
        total_latency = 0
        
        for stage_name, model_name in pipeline:
            stage_task = PodTask(
                task_id=f"{task.task_id}_{stage_name}",
                task_type=f"{task.task_type}_{stage_name}",
                prompt=self._build_stage_prompt(stage_name, task, current_output),
                schema=task.schema,
                constraints=task.constraints
            )
            
            result = await self._call_model(model_name, stage_task)
            total_tokens += result.tokens_used
            total_latency += result.latency_ms
            
            if not result.success:
                return TaskResult(
                    task_id=task.task_id,
                    model=f"pipeline(failed_at:{stage_name})",
                    output=current_output,
                    tokens_used=total_tokens,
                    latency_ms=total_latency,
                    success=False,
                    error=f"Pipeline failed at {stage_name}: {result.error}"
                )
            
            current_output = result.output
        
        return TaskResult(
            task_id=task.task_id,
            model="pipeline",
            output=current_output,
            tokens_used=total_tokens,
            latency_ms=total_latency,
            success=True
        )
    
    async def _execute_parallel(self, task: PodTask) -> TaskResult:
        """Execute on all models in parallel, merge results."""
        tasks = [self._call_model(m, task) for m in self.models.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = [r for r in results if isinstance(r, TaskResult) and r.success]
        
        if not successful:
            return TaskResult(
                task_id=task.task_id,
                model="parallel",
                output=None,
                tokens_used=0,
                latency_ms=0,
                success=False,
                error="All parallel executions failed"
            )
        
        # Merge outputs (task-type specific)
        merged = self._merge_outputs([r.output for r in successful], task.task_type)
        
        return TaskResult(
            task_id=task.task_id,
            model="parallel",
            output=merged,
            tokens_used=sum(r.tokens_used for r in successful),
            latency_ms=max(r.latency_ms for r in successful),
            success=True
        )
    
    async def _execute_hierarchical(self, task: PodTask) -> TaskResult:
        """Orchestrator model delegates subtasks."""
        # First, have orchestrator model decompose the task
        decompose_task = PodTask(
            task_id=f"{task.task_id}_decompose",
            task_type="decomposition",
            prompt=f"""Decompose this task into subtasks:

TASK: {task.prompt}

Return JSON with this structure:
{{
    "subtasks": [
        {{"id": "1", "type": "...", "prompt": "...", "dependencies": []}}
    ]
}}""",
            schema={"type": "object", "properties": {"subtasks": {"type": "array"}}}
        )
        
        decomposition = await self._call_model("claude", decompose_task)
        
        if not decomposition.success:
            # Fall back to direct execution
            return await self._execute_specialist(task)
        
        try:
            plan = json.loads(decomposition.output)
            subtasks = plan.get("subtasks", [])
        except (json.JSONDecodeError, KeyError):
            return await self._execute_specialist(task)
        
        # Execute subtasks respecting dependencies
        results = {}
        total_tokens = decomposition.tokens_used
        total_latency = decomposition.latency_ms
        
        for subtask_def in subtasks:
            # Check dependencies
            deps = subtask_def.get("dependencies", [])
            dep_outputs = {d: results[d].output for d in deps if d in results}
            
            subtask = PodTask(
                task_id=f"{task.task_id}_{subtask_def['id']}",
                task_type=subtask_def.get("type", task.task_type),
                prompt=subtask_def["prompt"] + f"\n\nContext from dependencies: {json.dumps(dep_outputs)}",
                schema=task.schema,
                constraints=task.constraints
            )
            
            result = await self._execute_specialist(subtask)
            results[subtask_def['id']] = result
            total_tokens += result.tokens_used
            total_latency += result.latency_ms
        
        # Synthesize final output
        synthesis_task = PodTask(
            task_id=f"{task.task_id}_synthesize",
            task_type="synthesis",
            prompt=f"""Synthesize these subtask results into a final output:

ORIGINAL TASK: {task.prompt}

SUBTASK RESULTS:
{json.dumps({k: v.output for k, v in results.items()}, indent=2)}

Return the final, complete output."""
        )
        
        final = await self._call_model("claude", synthesis_task)
        total_tokens += final.tokens_used
        total_latency += final.latency_ms
        
        return TaskResult(
            task_id=task.task_id,
            model="hierarchical",
            output=final.output,
            tokens_used=total_tokens,
            latency_ms=total_latency,
            success=final.success,
            error=final.error
        )
    
    def _select_model(self, task: PodTask) -> str:
        """Select best model for task."""
        if task.preferred_model and task.preferred_model in self.models:
            return task.preferred_model
        
        # Simple heuristic - production would use learned routing
        task_model_map = {
            'code': 'claude',
            'narrative': 'gpt4',
            'visual': 'gemini',
            'bulk': 'deepseek',
            'quick': 'local'
        }
        
        for keyword, model in task_model_map.items():
            if keyword in task.task_type.lower():
                if model in self.models:
                    return model
        
        return 'claude'  # Default
    
    def _rank_models(self, task_type: str) -> List[str]:
        """Rank models by expected quality for task type."""
        # Simplified - would use performance history
        return list(self.models.keys())
    
    async def _call_model(self, model_name: str, task: PodTask) -> TaskResult:
        """Call a specific model."""
        import time
        
        if model_name not in self.models:
            return TaskResult(
                task_id=task.task_id,
                model=model_name,
                output=None,
                tokens_used=0,
                latency_ms=0,
                success=False,
                error=f"Model {model_name} not registered"
            )
        
        start_time = time.time()
        
        # Placeholder - actual API call would go here
        # client = self.models[model_name]['client']
        # response = await client.generate(task.prompt)
        
        latency = (time.time() - start_time) * 1000
        
        return TaskResult(
            task_id=task.task_id,
            model=model_name,
            output="[Generated content would go here]",
            tokens_used=100,  # Placeholder
            latency_ms=latency,
            success=True
        )
    
    def _score_output(self, result: TaskResult, task: PodTask) -> float:
        """Score output quality."""
        # Simplified scoring - production would use learned quality model
        score = 0.5
        
        if result.output:
            # Length heuristic (not too short, not too long)
            output_str = str(result.output)
            if 100 < len(output_str) < 10000:
                score += 0.2
            
            # Schema compliance
            if task.schema:
                try:
                    data = json.loads(output_str)
                    # Would validate against schema
                    score += 0.3
                except:
                    pass
        
        return score
    
    def _merge_outputs(self, outputs: List[Any], task_type: str) -> Any:
        """Merge multiple outputs into one."""
        # Task-type specific merging
        if 'card' in task_type.lower():
            # For cards, deduplicate and combine
            all_cards = []
            seen_ids = set()
            for output in outputs:
                try:
                    cards = json.loads(output) if isinstance(output, str) else output
                    if isinstance(cards, list):
                        for card in cards:
                            card_id = card.get('id', str(hash(json.dumps(card))))
                            if card_id not in seen_ids:
                                seen_ids.add(card_id)
                                all_cards.append(card)
                except:
                    pass
            return all_cards
        
        # Default: return longest output
        return max(outputs, key=lambda x: len(str(x)) if x else 0)
    
    def _build_stage_prompt(self, stage: str, task: PodTask, 
                           previous_output: Any) -> str:
        """Build prompt for pipeline stage."""
        if stage == 'generator':
            return task.prompt
        elif stage == 'refiner':
            return f"""Refine and improve this generated content:

ORIGINAL TASK: {task.prompt}

GENERATED CONTENT:
{previous_output}

Improve quality, fix errors, enhance clarity. Return the refined content only."""
        elif stage == 'validator':
            return f"""Validate this content against requirements:

REQUIREMENTS: {json.dumps(task.constraints)}

CONTENT:
{previous_output}

If valid, return the content unchanged.
If invalid, fix issues and return corrected content."""
        else:
            return task.prompt
```

---

This concludes Part I of Encyclopedia Volume II. The full document continues with:

- **Part II**: Engine-Agnostic File Formats (Scene descriptions, meshes, materials, animation, audio)
- **Part III**: Godot-Specific Integration (GDScript, TSCN, resources, headless export)
- **Part IV**: Web-Native Architecture (Three.js, Phaser, WebGL/WebGPU, Electron/Tauri, Steamworks.js)
- **Part V**: Content Generation Systems (Cards, encounters, dialog, levels, economy)
- **Part VI**: LLM Prompting (Structured output, JSON schemas, error recovery)
- **Part VII**: Testing and QA (Automated playtesting, balance simulation, profiling)
- **Part VIII**: Deployment and Live Ops (Steam launch, updates, telemetry, A/B testing)

---

*Continue to Part II: Engine-Agnostic File Formats...*
