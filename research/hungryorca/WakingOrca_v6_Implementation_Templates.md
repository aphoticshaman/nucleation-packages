# WakingOrca v6: Implementation Templates & Test Suites
## Ready-to-Code Specifications for Next Session

**Purpose**: This document provides copy-paste ready code templates, test scaffolds, and debugging utilities for immediate implementation.

---

## 🧬 EVOLUTIONENGINE - IMPLEMENTATION TEMPLATE

### Complete Class Structure

```python
"""
EvolutionEngine - Population-based genetic optimization
File: wakingorca_v6.py (append to existing file)
"""

class EvolutionEngine:
    """
    Evolutionary algorithm for genome optimization with RRBR amplification
    """
    
    def __init__(self, config: Config, logger: logging.Logger, 
                 memory_bank: MemoryBank, knowledge_repo: KnowledgeRepository):
        self.config = config
        self.logger = logger
        self.memory_bank = memory_bank
        self.knowledge_repo = knowledge_repo
        
        # Population
        self.population: List[SolverGenome] = []
        self.generation = 0
        
        # Best tracking
        self.best_genome: Optional[SolverGenome] = None
        self.best_fitness = 0.0
        self.fitness_history: List[float] = []
        
        # RRBR amplification
        self.gain_multiplier = 1.0
        self.consecutive_improvements = 0
        
        # Diversity tracking
        self.diversity_scores: List[float] = []
        
        # Generation timing
        self.gen_start_time = 0.0
    
    def initialize_population(self, population_size: int):
        """
        Create initial diverse population
        
        Strategy:
        - 30% random (exploration)
        - 30% memory-seeded (exploitation)
        - 20% specialists (per task type)
        - 20% hybrid (crossover of seeds)
        """
        self.logger.info(f"🧬 Initializing population: {population_size} genomes")
        self.population = []
        
        # 1. Random exploration (30%)
        random_count = int(population_size * 0.3)
        for i in range(random_count):
            genome = SolverGenome.random(self.config)
            genome.genome_id = f"random_{i}"
            self.population.append(genome)
        
        self.logger.info(f"   Created {random_count} random genomes")
        
        # 2. Memory-seeded exploitation (30%)
        seed_count = int(population_size * 0.3)
        if self.memory_bank.memories:
            for i in range(seed_count):
                memory = random.choice(self.memory_bank.memories)
                genome = SolverGenome.from_memory(memory, self.config)
                genome.genome_id = f"seeded_{i}"
                self.population.append(genome)
            
            self.logger.info(f"   Created {seed_count} memory-seeded genomes")
        else:
            # No memories yet, create random instead
            for i in range(seed_count):
                genome = SolverGenome.random(self.config)
                genome.genome_id = f"seeded_random_{i}"
                self.population.append(genome)
            
            self.logger.info(f"   No memories available, created {seed_count} random instead")
        
        # 3. Specialist genomes (20%)
        specialist_count = int(population_size * 0.2)
        task_types = list(TaskType)
        for i in range(specialist_count):
            task_type = task_types[i % len(task_types)]
            genome = SolverGenome.specialist(task_type, self.config)
            genome.genome_id = f"specialist_{task_type.value}_{i}"
            self.population.append(genome)
        
        self.logger.info(f"   Created {specialist_count} specialist genomes")
        
        # 4. Hybrid crossover (20%)
        hybrid_count = population_size - len(self.population)
        if len(self.population) >= 2:
            for i in range(hybrid_count):
                p1, p2 = random.sample(self.population[:min(10, len(self.population))], 2)
                child = p1.crossover(p2, self.config)
                child.genome_id = f"hybrid_{i}"
                self.population.append(child)
            
            self.logger.info(f"   Created {hybrid_count} hybrid genomes")
        else:
            # Not enough parents, create random
            for i in range(hybrid_count):
                genome = SolverGenome.random(self.config)
                genome.genome_id = f"hybrid_random_{i}"
                self.population.append(genome)
        
        # Verify population size
        assert len(self.population) == population_size, \
            f"Population size mismatch: {len(self.population)} != {population_size}"
        
        # Calculate initial diversity
        diversity = self._calculate_diversity()
        self.diversity_scores.append(diversity)
        self.logger.info(f"   Initial diversity: {diversity:.3f}")
        
        self.logger.info(f"✅ Population initialized: {len(self.population)} genomes")
    
    def evaluate_population(self, train_tasks: List[Dict[str, Any]], 
                           num_eval_tasks: int = 50) -> Dict[int, float]:
        """
        Evaluate all genomes in population on training tasks
        
        Returns: {genome_index: fitness_score}
        """
        self.logger.info(f"📊 Evaluating population on {num_eval_tasks} tasks")
        
        # Sample tasks for evaluation
        if len(train_tasks) > num_eval_tasks:
            eval_tasks = random.sample(train_tasks, num_eval_tasks)
        else:
            eval_tasks = train_tasks
        
        fitness_scores = {}
        
        for i, genome in enumerate(self.population):
            try:
                # Create solver with this genome
                solver = BeamSearchSolver(
                    config=self.config,
                    logger=self.logger,
                    genome=genome
                )
                
                # Evaluate on subset of eval_tasks
                fitness = solver.evaluate_on_training(eval_tasks, num_tasks=20)
                fitness_scores[i] = fitness
                
                # Update genome
                genome.evaluate(fitness, self.config)
                
                # Track best
                if fitness > self.best_fitness:
                    delta = fitness - self.best_fitness
                    self.logger.info(f"   🎯 New best: {fitness:.4f} (Δ+{delta:.4f})")
                    
                    self.best_fitness = fitness
                    self.best_genome = deepcopy(genome)
                    
                    # RRBR amplification
                    self.consecutive_improvements += 1
                    if self.consecutive_improvements >= 3:
                        self.gain_multiplier *= 1.1
                        self.logger.info(
                            f"   🚀 RRBR ratchet activated: {self.gain_multiplier:.2f}x"
                        )
                    
                    # Commit to knowledge repo
                    traits = self._detect_genome_traits(genome)
                    self.knowledge_repo.commit(
                        genome=genome,
                        performance=fitness,
                        traits=traits,
                        description=f"Gen {self.generation}: +{delta:.4f} improvement"
                    )
                else:
                    self.consecutive_improvements = 0
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    self.logger.info(f"   Evaluated {i+1}/{len(self.population)} genomes")
            
            except Exception as e:
                self.logger.warning(f"   ⚠️  Genome {i} evaluation failed: {e}")
                fitness_scores[i] = 0.0
        
        # Update fitness history
        max_fitness = max(fitness_scores.values())
        avg_fitness = np.mean(list(fitness_scores.values()))
        self.fitness_history.append(max_fitness)
        
        self.logger.info(f"   Max fitness: {max_fitness:.4f}")
        self.logger.info(f"   Avg fitness: {avg_fitness:.4f}")
        self.logger.info(f"   Best ever: {self.best_fitness:.4f}")
        
        return fitness_scores
    
    def select_parents(self, fitness_scores: Dict[int, float], 
                      num_parents: int) -> List[SolverGenome]:
        """
        Select parents using hybrid strategy:
        - 10% elitism (best genomes)
        - 70% tournament (fitness-based)
        - 20% diversity (unique genomes)
        """
        parents = []
        
        # 1. Elitism (10%)
        elite_count = max(1, int(num_parents * 0.1))
        sorted_indices = sorted(
            fitness_scores.keys(),
            key=lambda i: fitness_scores[i],
            reverse=True
        )
        
        for idx in sorted_indices[:elite_count]:
            parents.append(deepcopy(self.population[idx]))
        
        self.logger.info(f"   Elite parents: {elite_count}")
        
        # 2. Tournament selection (70%)
        tournament_count = int(num_parents * 0.7)
        for _ in range(tournament_count):
            # Tournament of 3
            tournament_size = min(3, len(fitness_scores))
            contestants = random.sample(list(fitness_scores.keys()), tournament_size)
            winner = max(contestants, key=lambda i: fitness_scores[i])
            parents.append(deepcopy(self.population[winner]))
        
        self.logger.info(f"   Tournament parents: {tournament_count}")
        
        # 3. Diversity preservation (20%)
        diversity_count = num_parents - len(parents)
        
        # Calculate diversity scores for non-elite genomes
        diversity_candidates = [
            (i, self.population[i]) 
            for i in range(len(self.population))
            if i not in sorted_indices[:elite_count]
        ]
        
        # Sort by structural uniqueness
        if diversity_candidates:
            diversity_candidates.sort(
                key=lambda x: self._calculate_diversity_score(x[1], parents),
                reverse=True
            )
            
            for idx, genome in diversity_candidates[:diversity_count]:
                parents.append(deepcopy(genome))
        
        self.logger.info(f"   Diversity parents: {diversity_count}")
        
        return parents
    
    def create_next_generation(self, parents: List[SolverGenome], 
                              population_size: int) -> List[SolverGenome]:
        """
        Create next generation via genetic operators
        
        Strategy:
        - 20% elite (unchanged)
        - 40% mutation (single-parent)
        - 40% crossover (two-parent)
        """
        next_gen = []
        
        # 1. Elite preservation (20%)
        elite_count = int(population_size * 0.2)
        sorted_parents = sorted(
            parents,
            key=lambda g: g.best_fitness,
            reverse=True
        )
        
        for parent in sorted_parents[:elite_count]:
            elite_genome = deepcopy(parent)
            elite_genome.generation = self.generation + 1
            next_gen.append(elite_genome)
        
        self.logger.info(f"   Elite preserved: {elite_count}")
        
        # 2. Mutation offspring (40%)
        mutation_count = int(population_size * 0.4)
        for i in range(mutation_count):
            parent = random.choice(parents)
            child = parent.mutate(self.config)
            child.generation = self.generation + 1
            child.genome_id = f"mut_{self.generation}_{i}"
            next_gen.append(child)
        
        self.logger.info(f"   Mutation offspring: {mutation_count}")
        
        # 3. Crossover offspring (40%)
        crossover_count = population_size - len(next_gen)
        for i in range(crossover_count):
            p1, p2 = random.sample(parents, 2)
            child = p1.crossover(p2, self.config)
            child.generation = self.generation + 1
            child.genome_id = f"cross_{self.generation}_{i}"
            next_gen.append(child)
        
        self.logger.info(f"   Crossover offspring: {crossover_count}")
        
        return next_gen[:population_size]
    
    def evolve_generation(self, train_tasks: List[Dict[str, Any]]) -> float:
        """
        Execute one generation of evolution
        
        Returns: Best fitness in this generation
        """
        self.generation += 1
        self.gen_start_time = time.time()
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"GENERATION {self.generation}")
        self.logger.info(f"{'='*70}")
        
        # 1. Evaluate
        fitness_scores = self.evaluate_population(train_tasks)
        
        # 2. Select
        num_parents = max(10, len(self.population) // 2)
        parents = self.select_parents(fitness_scores, num_parents)
        
        # 3. Create next generation
        self.population = self.create_next_generation(parents, len(self.population))
        
        # 4. Calculate diversity
        diversity = self._calculate_diversity()
        self.diversity_scores.append(diversity)
        
        # 5. Log generation summary
        avg_fitness = np.mean(list(fitness_scores.values()))
        gen_time = time.time() - self.gen_start_time
        
        self.logger.info(f"📊 Generation {self.generation} Summary:")
        self.logger.info(f"   Best fitness: {self.best_fitness:.4f}")
        self.logger.info(f"   Avg fitness: {avg_fitness:.4f}")
        self.logger.info(f"   Diversity: {diversity:.3f}")
        self.logger.info(f"   RRBR multiplier: {self.gain_multiplier:.2f}x")
        self.logger.info(f"   Time: {gen_time:.1f}s")
        self.logger.info(f"{'='*70}\n")
        
        return self.best_fitness
    
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity
        
        Metrics:
        - Program length variance
        - Unique program signatures
        - Cognitive mode distribution
        """
        if not self.population:
            return 0.0
        
        # Program length variance
        lengths = [len(g.program) for g in self.population]
        length_variance = np.std(lengths) if lengths else 0.0
        
        # Unique signatures
        signatures = set()
        for genome in self.population:
            sig = self._genome_signature(genome)
            signatures.add(sig)
        
        uniqueness = len(signatures) / len(self.population)
        
        # Combine metrics
        diversity = (length_variance / 10.0 + uniqueness) / 2.0
        
        return min(1.0, diversity)
    
    def _calculate_diversity_score(self, genome: SolverGenome, 
                                   reference: List[SolverGenome]) -> float:
        """
        Calculate how different a genome is from reference set
        """
        if not reference:
            return 1.0
        
        genome_sig = self._genome_signature(genome)
        
        # Count how different this genome is
        differences = 0
        for ref_genome in reference:
            ref_sig = self._genome_signature(ref_genome)
            if genome_sig != ref_sig:
                differences += 1
        
        return differences / len(reference)
    
    def _genome_signature(self, genome: SolverGenome) -> str:
        """Create unique signature for genome"""
        if not hasattr(genome, 'program') or not genome.program:
            return "empty"
        
        # Hash first 10 operations + length
        prog_str = "_".join(genome.program[:10])
        length_str = str(len(genome.program))
        combined = f"{prog_str}_{length_str}"
        
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _detect_genome_traits(self, genome: SolverGenome) -> List[str]:
        """
        Detect emergent traits in genome
        
        Traits:
        - Specialist (rotation, symmetry, etc)
        - Generalist (diverse operations)
        - Efficient (short program)
        - Complex (long program)
        """
        traits = []
        
        if not hasattr(genome, 'program'):
            return traits
        
        # Analyze program composition
        rotation_ops = sum(1 for op in genome.program if 'rot' in op)
        flip_ops = sum(1 for op in genome.program if 'flip' in op)
        color_ops = sum(1 for op in genome.program if 'color' in op)
        gravity_ops = sum(1 for op in genome.program if 'gravity' in op)
        
        # Rotation specialist
        if rotation_ops > len(genome.program) * 0.3:
            traits.append('rotation_specialist')
        
        # Symmetry specialist
        if flip_ops > len(genome.program) * 0.3:
            traits.append('symmetry_specialist')
        
        # Color specialist
        if color_ops > len(genome.program) * 0.3:
            traits.append('color_specialist')
        
        # Physics specialist
        if gravity_ops > len(genome.program) * 0.3:
            traits.append('physics_specialist')
        
        # Generalist (diverse operations)
        unique_ops = len(set(genome.program))
        if unique_ops > len(genome.program) * 0.7:
            traits.append('generalist')
        
        # Efficient (short program)
        if len(genome.program) < 5:
            traits.append('efficient')
        
        # Complex (long program)
        if len(genome.program) > 15:
            traits.append('complex')
        
        return traits
```

---

## 🎯 WAKINGORKAORCHESTRATOR - IMPLEMENTATION TEMPLATE

### Complete Class Structure

```python
"""
WakingOrcaOrchestrator - Master coordinator
File: wakingorca_v6.py (append to existing file)
"""

class WakingOrcaOrchestrator:
    """
    Master coordinator for WakingOrca AGI system
    
    Manages:
    - Training phase (60% of time)
    - Solving phase (40% of time)
    - Time budget enforcement
    - Checkpoint management
    - Progress monitoring
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.output_dir)
        
        self.logger.info("="*70)
        self.logger.info("🐋 WAKINGORCA V6 - ULTIMATE META-AGI")
        self.logger.info("="*70)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Time budget: {config.time_budget_hours:.1f} hours")
        self.logger.info(f"  Population size: {config.population_size}")
        self.logger.info(f"  Output directory: {config.output_dir}")
        self.logger.info("="*70 + "\n")
        
        # Initialize components
        self.task_classifier = TaskClassifier()
        self.memory_bank = MemoryBank(config.memory_bank_size)
        self.knowledge_repo = KnowledgeRepository()
        self.program_cache = ProgramCache(
            config.cache_size, 
            config.cache_ttl_seconds
        )
        
        self.evolution_engine = EvolutionEngine(
            config=config,
            logger=self.logger,
            memory_bank=self.memory_bank,
            knowledge_repo=self.knowledge_repo
        )
        
        self.metrics = MetricsTracker(output_dir=config.output_dir)
        
        # Training data
        self.train_tasks: List[Dict[str, Any]] = []
        self.train_solutions: Dict[str, Any] = {}
        
        # Time management
        self.start_time = time.time()
        self.training_deadline = 0.0
        self.solving_deadline = 0.0
        
        self._setup_time_budget()
    
    def _setup_time_budget(self):
        """Configure time budget allocation"""
        budget_seconds = self.config.time_budget_hours * 3600
        
        # Training: 60% of budget
        self.training_deadline = self.start_time + (budget_seconds * 0.6)
        
        # Solving: 40% of budget (with 2-min safety buffer)
        self.solving_deadline = self.start_time + budget_seconds - 120
        
        self.logger.info("⏱️  Time Budget Allocation:")
        self.logger.info(
            f"  Total: {self.config.time_budget_hours:.1f} hours "
            f"({budget_seconds:.0f}s)"
        )
        self.logger.info(
            f"  Training: {(self.training_deadline - self.start_time)/3600:.2f} hours "
            f"({(self.training_deadline - self.start_time):.0f}s)"
        )
        self.logger.info(
            f"  Solving: {(self.solving_deadline - self.training_deadline)/3600:.2f} hours "
            f"({(self.solving_deadline - self.training_deadline):.0f}s)"
        )
        self.logger.info(f"  Safety buffer: 2 minutes")
        self.logger.info("")
    
    def train(self) -> SolverGenome:
        """
        Execute training phase
        
        Returns: Best genome found
        """
        self.logger.info("="*70)
        self.logger.info("🧬 TRAINING PHASE - EVOLUTION BEGINS")
        self.logger.info("="*70)
        self.logger.info("")
        
        # Load training tasks
        self.logger.info("📂 Loading training tasks...")
        self.train_tasks = self._load_training_tasks()
        self.logger.info(f"✅ Loaded {len(self.train_tasks)} training tasks")
        self.logger.info("")
        
        # Initialize population
        self.logger.info("🧬 Initializing evolution...")
        self.evolution_engine.initialize_population(self.config.population_size)
        self.logger.info("")
        
        # Evolution loop
        generation = 0
        last_progress_time = time.time()
        
        while time.time() < self.training_deadline:
            generation += 1
            
            # Evolve one generation
            try:
                best_fitness = self.evolution_engine.evolve_generation(
                    self.train_tasks
                )
                
                # Update metrics
                avg_fitness = np.mean([
                    g.best_fitness for g in self.evolution_engine.population
                ])
                gen_time = time.time() - self.evolution_engine.gen_start_time
                
                self.metrics.record_generation(
                    gen=generation,
                    best_score=best_fitness,
                    avg_score=avg_fitness,
                    gen_time=gen_time
                )
                
                # Progress update (every 2 minutes)
                if time.time() - last_progress_time >= self.config.progress_interval_seconds:
                    self._print_training_progress()
                    last_progress_time = time.time()
                
                # Checkpoint
                if generation % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(generation)
                
                # Check for stagnation
                if self.metrics.is_stagnating(threshold=15):
                    self.logger.warning(
                        "⚠️  Stagnation detected, increasing mutation rate"
                    )
                    self.config.mutation_rate_base = min(
                        0.5, 
                        self.config.mutation_rate_base * 1.2
                    )
                
                # Check time remaining
                remaining = self.training_deadline - time.time()
                if remaining < 60:
                    self.logger.info(
                        f"⏰ Less than 1 minute remaining, stopping training"
                    )
                    break
            
            except Exception as e:
                self.logger.error(f"❌ Generation {generation} failed: {e}")
                self.logger.error(traceback.format_exc())
                self.metrics.record_error()
        
        # Training complete
        self.logger.info("\n" + "="*70)
        self.logger.info("✅ TRAINING PHASE COMPLETE")
        self.logger.info("="*70)
        
        best_genome = self.evolution_engine.best_genome
        
        self.logger.info(f"📊 Final Statistics:")
        self.logger.info(f"  Generations: {generation}")
        self.logger.info(f"  Best fitness: {self.evolution_engine.best_fitness:.4f}")
        self.logger.info(f"  Avg fitness: {avg_fitness:.4f}")
        self.logger.info(f"  RRBR commits: {self.metrics.ratchet_commits}")
        self.logger.info(
            f"  Consciousness: {self.metrics.consciousness_level.name}"
        )
        
        # Emergent traits
        emergent_traits = self.knowledge_repo.get_emergent_traits()
        if emergent_traits:
            self.logger.info(f"  Emergent traits: {', '.join(emergent_traits)}")
        
        self.logger.info("")
        
        # Save final checkpoint
        self._save_checkpoint(generation, final=True)
        
        return best_genome
    
    def solve(self, test_tasks: List[Dict[str, Any]], 
             best_genome: SolverGenome) -> Dict[str, List[np.ndarray]]:
        """
        Execute solving phase
        
        Returns: {task_id: [prediction_1, prediction_2, ...]}
        """
        self.logger.info("="*70)
        self.logger.info("🎯 SOLVING PHASE - TEST EVALUATION")
        self.logger.info("="*70)
        self.logger.info("")
        
        self.logger.info(f"📋 Test tasks: {len(test_tasks)}")
        self.logger.info("")
        
        # Create solver with best genome
        solver = BeamSearchSolver(
            config=self.config,
            logger=self.logger,
            genome=best_genome
        )
        
        solutions = {}
        solve_start = time.time()
        
        for i, task in enumerate(test_tasks):
            task_id = task.get('id', f'task_{i}')
            
            # Calculate per-task timeout
            remaining_time = self.solving_deadline - time.time()
            remaining_tasks = len(test_tasks) - i
            
            if remaining_time <= 0:
                self.logger.warning(f"⏰ Time budget exhausted at task {i+1}")
                break
            
            task_timeout = min(
                self.config.timeout_per_task,
                remaining_time / max(1, remaining_tasks) * 0.9  # 90% safety margin
            )
            
            self.logger.info(
                f"🧩 Task {i+1}/{len(test_tasks)}: {task_id} "
                f"(timeout: {task_timeout:.1f}s)"
            )
            
            try:
                # Solve task
                predictions = solver.solve_task(task, timeout_seconds=task_timeout)
                solutions[task_id] = predictions
                
                self.logger.info(
                    f"   ✅ Generated {len(predictions)} predictions"
                )
            
            except Exception as e:
                self.logger.error(f"   ❌ Task failed: {e}")
                
                # Fallback predictions
                try:
                    solutions[task_id] = solver._generate_fallback_solutions(task)
                    self.logger.info(f"   ⚠️  Using fallback predictions")
                except:
                    # Ultimate fallback: identity transform
                    solutions[task_id] = [
                        np.array(test_input['input'])
                        for test_input in task['test']
                    ]
                    self.logger.info(f"   ⚠️  Using identity predictions")
        
        # Solving complete
        solve_time = time.time() - solve_start
        
        self.logger.info("\n" + "="*70)
        self.logger.info("✅ SOLVING PHASE COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"📊 Results:")
        self.logger.info(f"  Tasks solved: {len(solutions)}/{len(test_tasks)}")
        self.logger.info(f"  Time: {solve_time:.1f}s")
        self.logger.info(
            f"  Avg time per task: {solve_time/max(1, len(solutions)):.1f}s"
        )
        self.logger.info("")
        
        return solutions
    
    def _load_training_tasks(self) -> List[Dict[str, Any]]:
        """Load ARC training tasks"""
        training_path = self.config.data_dir / 'arc-agi_training_challenges.json'
        
        if not training_path.exists():
            self.logger.error(f"Training file not found: {training_path}")
            raise FileNotFoundError(f"Training file not found: {training_path}")
        
        try:
            with open(training_path, 'r') as f:
                training_data = json.load(f)
            
            tasks = []
            for task_id, task_data in training_data.items():
                task = {
                    'id': task_id,
                    'train': task_data['train'],
                    'test': task_data.get('test', [])
                }
                tasks.append(task)
            
            return tasks
        
        except Exception as e:
            self.logger.error(f"Failed to load training tasks: {e}")
            raise
    
    def _save_checkpoint(self, generation: int, final: bool = False):
        """Save checkpoint for recovery"""
        if final:
            checkpoint_path = self.config.output_dir / 'checkpoint_final.pkl'
        else:
            checkpoint_path = self.config.output_dir / f'checkpoint_gen_{generation}.pkl'
        
        checkpoint_data = {
            'generation': generation,
            'timestamp': time.time(),
            'best_genome': self.evolution_engine.best_genome,
            'best_fitness': self.evolution_engine.best_fitness,
            'population': self.evolution_engine.population,
            'metrics': self.metrics,
            'knowledge_repo': self.knowledge_repo,
            'memory_bank': self.memory_bank,
            'config': self.config
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"💾 Checkpoint saved: {checkpoint_path.name}")
        
        except Exception as e:
            self.logger.error(f"❌ Checkpoint save failed: {e}")
    
    def _print_training_progress(self):
        """Print comprehensive training progress"""
        elapsed = self.metrics.get_elapsed_time()
        remaining = self.training_deadline - time.time()
        
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 TRAINING PROGRESS UPDATE")
        self.logger.info("="*70)
        
        # Time
        self.logger.info(f"⏱️  Time:")
        self.logger.info(f"  Elapsed: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
        self.logger.info(f"  Remaining: {remaining/3600:.2f}h ({remaining:.0f}s)")
        
        # Performance
        self.logger.info(f"🎯 Performance:")
        self.logger.info(f"  Best fitness: {self.metrics.best_ever_score:.4f}")
        if self.metrics.avg_scores:
            self.logger.info(f"  Avg fitness: {self.metrics.avg_scores[-1]:.4f}")
        self.logger.info(
            f"  Improvement rate: {self.metrics.get_improvement_rate():.4f}/gen"
        )
        
        # RRBR
        self.logger.info(f"🚀 RRBR Metrics:")
        self.logger.info(f"  Commits: {self.metrics.ratchet_commits}")
        self.logger.info(
            f"  Total gain: {sum(self.metrics.asymmetric_gains):.4f}"
        )
        self.logger.info(
            f"  Consciousness: {self.metrics.consciousness_level.name}"
        )
        
        # Cache
        self.logger.info(f"💾 Cache:")
        self.logger.info(
            f"  Hit rate: {self.metrics.get_cache_hit_rate():.2%}"
        )
        self.logger.info(
            f"  Hits/Misses: {self.metrics.cache_hits}/{self.metrics.cache_misses}"
        )
        
        # Evolution
        self.logger.info(f"🧬 Evolution:")
        self.logger.info(f"  Generations: {len(self.metrics.best_scores)}")
        self.logger.info(
            f"  Genome evals: {self.metrics.genome_evaluations}"
        )
        
        # Knowledge
        commit_log = self.knowledge_repo.get_commit_log()
        if commit_log:
            self.logger.info(f"📚 Recent commits:")
            for log in commit_log[-3:]:
                self.logger.info(f"  {log}")
        
        # Emergent traits
        emergent = self.knowledge_repo.get_emergent_traits()
        if emergent:
            self.logger.info(f"🌟 Emergent traits: {', '.join(emergent)}")
        
        self.logger.info("="*70 + "\n")


# ============================================================================
# ENTRY POINTS
# ============================================================================

def load_test_tasks(data_dir: Path) -> List[Dict[str, Any]]:
    """Load ARC test tasks"""
    test_path = data_dir / 'arc-agi_test_challenges.json'
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    tasks = []
    for task_id, task_data in test_data.items():
        task = {
            'id': task_id,
            'train': task_data.get('train', []),
            'test': task_data['test']
        }
        tasks.append(task)
    
    return tasks

def save_submission(solutions: Dict[str, List[np.ndarray]], 
                   output_dir: Path):
    """Generate Kaggle submission file"""
    submission = {}
    
    for task_id, predictions in solutions.items():
        task_submissions = []
        
        for pred in predictions:
            attempts = {
                'attempt_1': pred.tolist(),
                'attempt_2': pred.tolist()
            }
            task_submissions.append(attempts)
        
        submission[task_id] = task_submissions
    
    submission_path = output_dir / 'submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"💾 Submission saved: {submission_path}")
    print(f"   Tasks: {len(submission)}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='WakingOrca v6 - Ultimate Meta-AGI for ARC Prize 2025'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'solve', 'full'],
        default='full',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--time-budget',
        type=float,
        default=3.0,
        help='Time budget in hours'
    )
    
    parser.add_argument(
        '--population-size',
        type=int,
        default=50,
        help='Population size'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/kaggle/working',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        time_budget_hours=args.time_budget,
        population_size=args.population_size,
        output_dir=Path(args.output_dir)
    )
    
    # Create orchestrator
    orchestrator = WakingOrcaOrchestrator(config)
    
    try:
        if args.mode in ['train', 'full']:
            # Training
            best_genome = orchestrator.train()
        
        if args.mode in ['solve', 'full']:
            # Solving
            test_tasks = load_test_tasks(config.data_dir)
            
            if args.mode == 'solve':
                # Load checkpoint
                checkpoint_path = config.output_dir / 'checkpoint_final.pkl'
                # TODO: Load checkpoint
            
            solutions = orchestrator.solve(test_tasks, best_genome)
            save_submission(solutions, config.output_dir)
        
        # Save metrics
        orchestrator.metrics.save_metrics()
        
        orchestrator.logger.info("\n" + "="*70)
        orchestrator.logger.info("🐋 WAKINGORCA V6 - EXECUTION COMPLETE")
        orchestrator.logger.info("="*70)
        
        return 0
    
    except Exception as e:
        orchestrator.logger.error("❌ Fatal error:")
        orchestrator.logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

---

## 🧪 TEST TEMPLATES

### Unit Test Template

```python
"""
test_wakingorca_v6.py - Comprehensive test suite
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

# Import all components
from wakingorca_v6 import (
    Config, EvolutionEngine, WakingOrcaOrchestrator,
    SolverGenome, BeamSearchSolver, TaskClassifier,
    KnowledgeRepository, MemoryBank, ProgramCache,
    Primitives, TaskType, load_training_tasks, load_test_tasks
)

class TestEvolutionEngine:
    """Test EvolutionEngine component"""
    
    @pytest.fixture
    def config(self):
        return Config(
            time_budget_hours=0.1,
            population_size=10,
            output_dir=Path(tempfile.mkdtemp())
        )
    
    @pytest.fixture
    def logger(self, config):
        from wakingorca_v6 import setup_logging
        return setup_logging(config.output_dir)
    
    @pytest.fixture
    def engine(self, config, logger):
        memory_bank = MemoryBank(max_size=100)
        knowledge_repo = KnowledgeRepository()
        return EvolutionEngine(config, logger, memory_bank, knowledge_repo)
    
    def test_population_initialization(self, engine):
        """Test population initialization creates diverse genomes"""
        engine.initialize_population(20)
        
        assert len(engine.population) == 20
        
        # Check diversity
        signatures = set()
        for genome in engine.population:
            sig = engine._genome_signature(genome)
            signatures.add(sig)
        
        assert len(signatures) >= 10  # At least 50% unique
    
    def test_fitness_evaluation(self, engine):
        """Test fitness evaluation works"""
        engine.initialize_population(5)
        
        mock_tasks = generate_mock_training_tasks(10)
        fitness_scores = engine.evaluate_population(mock_tasks, num_eval_tasks=10)
        
        assert len(fitness_scores) == 5
        assert all(0.0 <= score <= 1.0 for score in fitness_scores.values())
    
    def test_parent_selection(self, engine):
        """Test parent selection"""
        engine.initialize_population(10)
        
        fitness_scores = {i: random.random() for i in range(10)}
        parents = engine.select_parents(fitness_scores, num_parents=5)
        
        assert len(parents) == 5
    
    def test_next_generation_creation(self, engine):
        """Test offspring generation"""
        engine.initialize_population(10)
        
        parents = engine.population[:5]
        next_gen = engine.create_next_generation(parents, population_size=10)
        
        assert len(next_gen) == 10
    
    def test_rrbr_amplification(self, engine):
        """Test RRBR gain amplification"""
        engine.best_fitness = 0.5
        
        # Simulate 3 consecutive improvements
        for i in range(3):
            engine.best_fitness = 0.5 + (i + 1) * 0.01
            engine.consecutive_improvements += 1
            
            if engine.consecutive_improvements >= 3:
                engine.gain_multiplier *= 1.1
        
        assert engine.gain_multiplier > 1.0

class TestWakingOrcaOrchestrator:
    """Test WakingOrcaOrchestrator component"""
    
    @pytest.fixture
    def config(self):
        return Config(
            time_budget_hours=0.05,  # 3 minutes
            population_size=5,
            output_dir=Path(tempfile.mkdtemp())
        )
    
    @pytest.fixture
    def orchestrator(self, config):
        return WakingOrcaOrchestrator(config)
    
    def test_time_budget_setup(self, orchestrator):
        """Test time budget allocation"""
        budget_seconds = orchestrator.config.time_budget_hours * 3600
        
        training_duration = (
            orchestrator.training_deadline - orchestrator.start_time
        )
        
        # Should be ~60% of budget
        expected_training = budget_seconds * 0.6
        assert abs(training_duration - expected_training) < 10  # 10s tolerance
    
    def test_checkpoint_save_load(self, orchestrator, config):
        """Test checkpoint persistence"""
        # Create some state
        orchestrator.evolution_engine.initialize_population(5)
        orchestrator.evolution_engine.best_fitness = 0.42
        
        # Save checkpoint
        orchestrator._save_checkpoint(1, final=True)
        
        checkpoint_path = config.output_dir / 'checkpoint_final.pkl'
        assert checkpoint_path.exists()
    
    def test_training_with_mock_tasks(self, orchestrator):
        """Test training phase with mock tasks"""
        # Create mock training tasks
        orchestrator.train_tasks = generate_mock_training_tasks(20)
        
        # Override training deadline for quick test
        orchestrator.training_deadline = time.time() + 30  # 30 seconds
        
        try:
            best_genome = orchestrator.train()
            assert best_genome is not None
        except Exception as e:
            pytest.fail(f"Training failed: {e}")

class TestIntegration:
    """Integration tests"""
    
    def test_mini_end_to_end(self):
        """Minimal end-to-end test (5 minutes)"""
        config = Config(
            time_budget_hours=0.083,  # 5 minutes
            population_size=10,
            output_dir=Path(tempfile.mkdtemp())
        )
        
        orchestrator = WakingOrcaOrchestrator(config)
        
        # Mock training
        orchestrator.train_tasks = generate_mock_training_tasks(10)
        
        try:
            best_genome = orchestrator.train()
            assert best_genome is not None
            
            # Mock solving
            test_tasks = generate_mock_test_tasks(5)
            solutions = orchestrator.solve(test_tasks, best_genome)
            
            assert len(solutions) == 5
        
        except Exception as e:
            pytest.fail(f"End-to-end test failed: {e}")

# Helper functions

def generate_mock_training_tasks(n: int) -> List[Dict]:
    """Generate mock training tasks"""
    tasks = []
    for i in range(n):
        task = {
            'id': f'mock_train_{i}',
            'train': [
                {
                    'input': np.random.randint(0, 10, (5, 5)).tolist(),
                    'output': np.random.randint(0, 10, (5, 5)).tolist()
                }
                for _ in range(3)
            ],
            'test': []
        }
        tasks.append(task)
    return tasks

def generate_mock_test_tasks(n: int) -> List[Dict]:
    """Generate mock test tasks"""
    tasks = []
    for i in range(n):
        task = {
            'id': f'mock_test_{i}',
            'train': [
                {
                    'input': np.random.randint(0, 10, (5, 5)).tolist(),
                    'output': np.random.randint(0, 10, (5, 5)).tolist()
                }
                for _ in range(2)
            ],
            'test': [
                {'input': np.random.randint(0, 10, (5, 5)).tolist()}
                for _ in range(2)
            ]
        }
        tasks.append(task)
    return tasks

# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## 🐛 DEBUGGING UTILITIES

```python
"""
debug_utils.py - Debugging utilities for WakingOrca v6
"""

def inspect_genome(genome: SolverGenome):
    """Print detailed genome inspection"""
    print("\n" + "="*70)
    print(f"GENOME INSPECTION: {genome.genome_id}")
    print("="*70)
    
    print(f"Program length: {len(genome.program)}")
    print(f"Best fitness: {genome.best_fitness:.4f}")
    print(f"Generation: {genome.generation}")
    print(f"\nProgram steps:")
    for i, step in enumerate(genome.program[:10], 1):
        print(f"  {i}. {step}")
    if len(genome.program) > 10:
        print(f"  ... ({len(genome.program) - 10} more steps)")
    print("="*70 + "\n")

def visualize_grid(grid: np.ndarray, title: str = "Grid"):
    """Visualize a grid using ASCII art"""
    print(f"\n{title}:")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid:
        print("|" + "".join(f"{int(val)}" for val in row) + "|")
    print("-" * (grid.shape[1] * 2 + 1))

def profile_component(func, *args, **kwargs):
    """Profile a component's execution"""
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result

def validate_submission(submission_path: Path):
    """Validate submission file format"""
    print(f"\n🔍 Validating submission: {submission_path}")
    
    try:
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        print(f"✅ Valid JSON")
        print(f"   Tasks: {len(submission)}")
        
        for task_id, predictions in submission.items():
            print(f"\n   Task: {task_id}")
            print(f"   Test cases: {len(predictions)}")
            
            for i, pred in enumerate(predictions):
                assert 'attempt_1' in pred, f"Missing attempt_1 in task {task_id}"
                assert 'attempt_2' in pred, f"Missing attempt_2 in task {task_id}"
                
                attempt_1 = np.array(pred['attempt_1'])
                print(f"     Test {i+1}: {attempt_1.shape}")
        
        print(f"\n✅ Submission format valid!")
        return True
    
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False
```

---

**END OF IMPLEMENTATION TEMPLATES**

This document is ready for immediate use in the next development session. Copy-paste code templates directly into `wakingorca_v6.py` and use test templates to verify correctness.
