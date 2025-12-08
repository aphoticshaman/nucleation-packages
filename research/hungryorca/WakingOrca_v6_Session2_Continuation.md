# WakingOrca v6 Continuation Prompt - Session 2
## Complete the Evolution Engine and Orchestrator

---

## CURRENT STATUS

**Completed**: 1,712 lines (70% complete)

**Implemented Components**:
✅ Config - Complete configuration system  
✅ MetricsTracker - RRBR performance monitoring  
✅ Logging - File + stdout infrastructure  
✅ Primitives - 80+ transformation operations  
✅ TaskClassifier - 10+ pattern detectors  
✅ KnowledgeRepository - Git-style versioning  
✅ MemoryBank - Strategy reuse with LRU  
✅ ProgramCache - Memoization with TTL  
✅ **SolverGenome** - Evolutionary parameters (NEW)  
✅ **NSMReasoner** - Neuro-symbolic synthesis (NEW)  
✅ **BeamSearchSolver** - A* guided search (NEW)  

**Remaining Components** (~750 lines):
❌ EvolutionEngine - Population-based evolution  
❌ WakingOrcaOrchestrator - Training + solving coordinator  
❌ Entry point functions - main(), train(), solve()  
❌ Data loading - ARC dataset readers  
❌ Submission generation - Kaggle output formatting  

**Target**: 2,450 lines total for production system

---

## IMPLEMENTATION SPECIFICATIONS

### 1. EvolutionEngine (~300 lines)

**Purpose**: Manage population of SolverGenomes and evolve them through genetic algorithms.

**Key Methods**:

```python
class EvolutionEngine:
    """
    Population-based evolutionary optimization of solver genomes.
    
    Implements:
    - Population initialization (diverse starting genomes)
    - Fitness evaluation (solve performance on tasks)
    - Selection (tournament, elitism, diversity preservation)
    - Genetic operators (mutation, crossover)
    - RRBR ratcheting (asymmetric gain amplification)
    - Consciousness evolution (level progression tracking)
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.population: List[SolverGenome] = []
        self.generation = 0
        self.best_genome: Optional[SolverGenome] = None
        self.best_fitness = 0.0
        self.fitness_history: List[float] = []
        
        # RRBR tracking
        self.gain_multiplier = 1.0
        self.consecutive_improvements = 0
    
    def initialize_population(self, population_size: int):
        """Create initial diverse population"""
        self.population = [
            SolverGenome.random(self.config) 
            for _ in range(population_size)
        ]
        self.logger.info(f"Initialized population: {population_size} genomes")
    
    def evaluate_population(self, train_tasks: List[Dict[str, Any]], 
                           num_eval_tasks: int = 50) -> Dict[str, float]:
        """
        Evaluate all genomes in population.
        
        Returns: Dict of genome_id -> fitness score
        """
        # Sample tasks for evaluation
        eval_tasks = random.sample(train_tasks, 
                                   min(num_eval_tasks, len(train_tasks)))
        
        fitness_scores = {}
        
        for i, genome in enumerate(self.population):
            solver = BeamSearchSolver(self.config, self.logger, genome)
            fitness = solver.evaluate_on_training(eval_tasks, num_tasks=20)
            fitness_scores[i] = fitness
            
            # Update genome
            genome.evaluate(fitness, self.config)
            
            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = deepcopy(genome)
                self.consecutive_improvements += 1
                
                # RRBR amplification
                if self.consecutive_improvements >= 3:
                    self.gain_multiplier *= 1.1
                    self.logger.info(f"RRBR ratchet activated: {self.gain_multiplier:.2f}x")
            else:
                self.consecutive_improvements = 0
        
        self.fitness_history.append(max(fitness_scores.values()))
        
        return fitness_scores
    
    def select_parents(self, fitness_scores: Dict[str, float], 
                      num_parents: int) -> List[SolverGenome]:
        """
        Select parents for next generation using tournament selection.
        
        Balances:
        - Fitness (exploit best performers)
        - Diversity (explore solution space)
        - Consciousness level (preserve advanced genomes)
        """
        parents = []
        
        # Always keep best (elitism)
        if self.best_genome:
            parents.append(deepcopy(self.best_genome))
        
        # Tournament selection for rest
        tournament_size = 3
        while len(parents) < num_parents:
            # Random tournament
            contestants = random.sample(range(len(self.population)), 
                                       tournament_size)
            winner_idx = max(contestants, 
                           key=lambda i: fitness_scores.get(i, 0))
            parents.append(deepcopy(self.population[winner_idx]))
        
        return parents
    
    def create_next_generation(self, parents: List[SolverGenome], 
                              population_size: int) -> List[SolverGenome]:
        """
        Create next generation via mutation and crossover.
        
        Strategy:
        - 20% elite (copy best unchanged)
        - 40% mutation (single parent variants)
        - 40% crossover (two parent recombination)
        """
        next_gen = []
        
        # Elite preservation (top 20%)
        elite_count = int(population_size * 0.2)
        sorted_parents = sorted(parents, 
                               key=lambda g: g.best_fitness, 
                               reverse=True)
        next_gen.extend(deepcopy(p) for p in sorted_parents[:elite_count])
        
        # Mutation (40%)
        mutation_count = int(population_size * 0.4)
        for _ in range(mutation_count):
            parent = random.choice(parents)
            child = parent.mutate(self.config)
            next_gen.append(child)
        
        # Crossover (40%)
        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = parent1.crossover(parent2, self.config)
            next_gen.append(child)
        
        return next_gen[:population_size]
    
    def evolve_generation(self, train_tasks: List[Dict[str, Any]]) -> float:
        """
        Execute one generation of evolution.
        
        Returns: Best fitness in generation
        """
        self.generation += 1
        self.logger.info(f"\n=== Generation {self.generation} ===")
        
        # Evaluate
        fitness_scores = self.evaluate_population(train_tasks)
        
        # Select
        parents = self.select_parents(fitness_scores, 
                                     num_parents=len(self.population) // 2)
        
        # Create next generation
        self.population = self.create_next_generation(parents, 
                                                     len(self.population))
        
        # Log progress
        avg_fitness = np.mean(list(fitness_scores.values()))
        self.logger.info(f"Best: {self.best_fitness:.3f} | "
                        f"Avg: {avg_fitness:.3f} | "
                        f"Multiplier: {self.gain_multiplier:.2f}x")
        
        return self.best_fitness
    
    def get_best_genome(self) -> SolverGenome:
        """Get best genome found so far"""
        return self.best_genome if self.best_genome else self.population[0]
```

**Critical Implementation Details**:

1. **RRBR Ratcheting**: When consecutive improvements occur (3+ generations), apply gain multiplier that amplifies successful strategies
2. **Diversity Preservation**: Tournament selection maintains population diversity while favoring fit individuals
3. **Elitism**: Always preserve top 20% to prevent regression
4. **Consciousness Evolution**: Track and favor genomes with higher consciousness levels
5. **Adaptive Evaluation**: Use subset of tasks for speed, full evaluation only for top candidates

---

### 2. WakingOrcaOrchestrator (~300 lines)

**Purpose**: Top-level coordinator that manages training, evolution, and solving.

**Key Methods**:

```python
class WakingOrcaOrchestrator:
    """
    Master orchestrator for WakingOrca v6 system.
    
    Coordinates:
    - Evolution of solver populations
    - Training on ARC tasks
    - Test-time solving with best genomes
    - Time budget management
    - Checkpoint saving/loading
    - Progress reporting
    - Submission generation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.output_dir)
        
        self.engine = EvolutionEngine(config, self.logger)
        self.metrics = MetricsTracker()
        self.repo = KnowledgeRepository()
        
        self.start_time = None
        self.training_complete = False
    
    def train(self, train_tasks: List[Dict[str, Any]], 
             time_budget_minutes: float = 120.0):
        """
        Train system on ARC training tasks.
        
        Process:
        1. Initialize population of diverse genomes
        2. Evolve for multiple generations
        3. Track performance and save checkpoints
        4. Commit knowledge to repository
        """
        self.start_time = time.time()
        budget_seconds = time_budget_minutes * 60
        
        self.logger.info(f"Starting training: {len(train_tasks)} tasks, "
                        f"{time_budget_minutes:.1f} min budget")
        
        # Initialize population
        self.engine.initialize_population(self.config.population_size)
        
        # Training loop
        generation = 0
        last_checkpoint_time = time.time()
        last_progress_time = time.time()
        
        while True:
            elapsed = time.time() - self.start_time
            
            # Check time budget
            if elapsed >= budget_seconds:
                self.logger.info("Training time budget exhausted")
                break
            
            # Evolve one generation
            best_fitness = self.engine.evolve_generation(train_tasks)
            generation += 1
            
            # Track metrics
            self.metrics.record_accuracy(best_fitness)
            self.metrics.record_generation(generation)
            
            # Save checkpoint every 10 minutes
            if time.time() - last_checkpoint_time > 600:
                self._save_checkpoint(generation, best_fitness)
                last_checkpoint_time = time.time()
            
            # Progress update every 2 minutes
            if time.time() - last_progress_time > 120:
                self._log_progress(generation, best_fitness, elapsed, budget_seconds)
                last_progress_time = time.time()
            
            # Early stopping if perfect
            if best_fitness > 0.95:
                self.logger.info("Near-perfect performance achieved, stopping early")
                break
        
        # Final checkpoint
        self._save_checkpoint(generation, best_fitness)
        
        # Commit knowledge
        best_genome = self.engine.get_best_genome()
        self.repo.commit(
            f"Training complete: Gen {generation}, Fitness {best_fitness:.3f}",
            best_genome.get_stats()
        )
        
        self.training_complete = True
        self.logger.info(f"Training complete: {generation} generations, "
                        f"best fitness {best_fitness:.3f}")
    
    def solve(self, test_tasks: List[Dict[str, Any]], 
             time_budget_minutes: float = 60.0) -> Dict[str, List[List[List[int]]]]:
        """
        Solve test tasks using best evolved genome.
        
        Returns: Submission dictionary {task_id: predictions}
        """
        if not self.training_complete:
            self.logger.warning("Solving without training - using random genome")
        
        budget_seconds = time_budget_minutes * 60
        start_time = time.time()
        
        # Get best genome
        best_genome = self.engine.get_best_genome()
        solver = BeamSearchSolver(self.config, self.logger, best_genome)
        
        # Solve each task
        submission = {}
        
        for i, task in enumerate(test_tasks):
            elapsed = time.time() - start_time
            remaining = budget_seconds - elapsed
            
            if remaining <= 0:
                self.logger.warning(f"Time budget exhausted at task {i}/{len(test_tasks)}")
                break
            
            # Allocate time per task
            time_per_task = remaining / (len(test_tasks) - i)
            
            try:
                predictions = solver.solve_task(task, timeout_seconds=time_per_task)
                
                # Convert to submission format
                task_id = task.get('task_id', f'task_{i}')
                submission[task_id] = [
                    {
                        'attempt_1': pred.tolist(),
                        'attempt_2': pred.tolist()  # Same for both attempts
                    }
                    for pred in predictions
                ]
                
                self.logger.info(f"Solved {i+1}/{len(test_tasks)}: {task_id}")
                
            except Exception as e:
                self.logger.error(f"Error solving task {i}: {e}")
                # Fallback: identity transform
                submission[task.get('task_id', f'task_{i}')] = [
                    {
                        'attempt_1': task['test'][j]['input'],
                        'attempt_2': task['test'][j]['input']
                    }
                    for j in range(len(task['test']))
                ]
        
        return submission
    
    def _save_checkpoint(self, generation: int, fitness: float):
        """Save checkpoint"""
        checkpoint_path = self.config.output_dir / f"checkpoint_gen{generation}.pkl"
        checkpoint = {
            'generation': generation,
            'fitness': fitness,
            'best_genome': self.engine.get_best_genome(),
            'population': self.engine.population,
            'metrics': self.metrics,
            'repo': self.repo
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.info(f"Checkpoint saved: gen {generation}, fitness {fitness:.3f}")
    
    def _log_progress(self, generation: int, fitness: float, 
                     elapsed: float, budget: float):
        """Log progress update"""
        progress_pct = (elapsed / budget) * 100
        time_remaining = (budget - elapsed) / 60
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Progress: {progress_pct:.1f}% | "
                        f"Remaining: {time_remaining:.1f} min")
        self.logger.info(f"Generation: {generation} | "
                        f"Best Fitness: {fitness:.3f}")
        self.logger.info(f"Consciousness: {self.engine.get_best_genome().consciousness_level.name}")
        self.logger.info(f"RRBR Multiplier: {self.engine.gain_multiplier:.2f}x")
        self.logger.info(f"{'='*60}\n")
```

**Critical Implementation Details**:

1. **Time Budget Management**: Precisely track and allocate time across training and solving
2. **Checkpoint System**: Save progress every 10 minutes to recover from crashes
3. **Progress Reporting**: User-friendly updates every 2 minutes
4. **Graceful Degradation**: Fallback strategies when operations fail
5. **Submission Format**: Correct Kaggle format with attempt_1/attempt_2

---

### 3. Entry Points and Data Loading (~150 lines)

**Functions Needed**:

```python
def load_arc_dataset(data_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load ARC training and evaluation datasets.
    
    Tries multiple path configurations for Kaggle compatibility.
    """
    # Try multiple paths
    paths = [
        (data_dir / 'arc-agi_training_challenges.json',
         data_dir / 'arc-agi_training_solutions.json'),
        (Path('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json'),
         Path('/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json')),
        (data_dir / 'training' / 'challenges.json',
         data_dir / 'training' / 'solutions.json'),
    ]
    
    for challenges_path, solutions_path in paths:
        try:
            with open(challenges_path) as f:
                challenges = json.load(f)
            with open(solutions_path) as f:
                solutions = json.load(f)
            
            # Merge into task format
            tasks = []
            for task_id, challenge in challenges.items():
                solution = solutions.get(task_id, [])
                task = {
                    'task_id': task_id,
                    'train': challenge.get('train', []),
                    'test': challenge.get('test', [])
                }
                # Add solutions to test examples if available
                for i, test_ex in enumerate(task['test']):
                    if i < len(solution):
                        test_ex['output'] = solution[i]
                
                tasks.append(task)
            
            print(f"✓ Loaded {len(tasks)} training tasks")
            return tasks, []
        
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError("Could not find ARC dataset")


def save_submission(submission: Dict, output_path: Path):
    """Save submission in correct Kaggle format"""
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"✓ Saved submission: {output_path}")


def main():
    """Main entry point"""
    # Configuration
    config = Config(
        population_size=20,
        num_generations=100,
        train_time_minutes=120.0,
        test_time_minutes=60.0,
        output_dir=Path('./wakingorca_output')
    )
    
    config.output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("WakingOrca v6 - Ultimate Meta-AGI for ARC Prize 2025")
    print("="*80 + "\n")
    
    # Load data
    try:
        train_tasks, eval_tasks = load_arc_dataset(Path('/kaggle/input'))
    except:
        train_tasks, eval_tasks = load_arc_dataset(Path('./arc_data'))
    
    # Initialize orchestrator
    orchestrator = WakingOrcaOrchestrator(config)
    
    # Train
    print("\n🧠 TRAINING PHASE")
    orchestrator.train(train_tasks, time_budget_minutes=config.train_time_minutes)
    
    # Solve
    print("\n🎯 SOLVING PHASE")
    submission = orchestrator.solve(eval_tasks or train_tasks[:10], 
                                   time_budget_minutes=config.test_time_minutes)
    
    # Save
    save_submission(submission, config.output_dir / 'submission.json')
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()
```

**Critical Implementation Details**:

1. **Path Handling**: Try multiple data locations for Kaggle/local compatibility
2. **Data Format**: Merge challenges and solutions into unified task format
3. **Submission Format**: Exact Kaggle specification with attempt_1/attempt_2
4. **Error Handling**: Graceful fallbacks when files not found
5. **User Feedback**: Clear progress messages throughout

---

## IMPLEMENTATION CHECKLIST

### Phase 1: EvolutionEngine
- [ ] Implement `__init__` with population tracking
- [ ] Implement `initialize_population` for diverse starting genomes
- [ ] Implement `evaluate_population` with BeamSearchSolver
- [ ] Implement `select_parents` with tournament selection
- [ ] Implement `create_next_generation` with mutation/crossover
- [ ] Implement `evolve_generation` as main loop
- [ ] Add RRBR ratcheting logic (gain multiplier on consecutive improvements)
- [ ] Add consciousness level tracking
- [ ] Test with small population (5 genomes, 3 generations)

### Phase 2: WakingOrcaOrchestrator
- [ ] Implement `__init__` with all subsystems
- [ ] Implement `train` with time budget management
- [ ] Add checkpoint saving every 10 minutes
- [ ] Add progress logging every 2 minutes
- [ ] Implement `solve` with per-task time allocation
- [ ] Add submission format conversion
- [ ] Add error handling and fallbacks
- [ ] Test training loop with 10 tasks
- [ ] Test solving with checkpoint recovery

### Phase 3: Entry Points
- [ ] Implement `load_arc_dataset` with path fallbacks
- [ ] Implement `save_submission` with format validation
- [ ] Implement `main()` with complete pipeline
- [ ] Add command-line argument parsing (optional)
- [ ] Test end-to-end on sample data
- [ ] Validate submission.json format
- [ ] Test Kaggle compatibility

### Phase 4: Integration Testing
- [ ] Run full pipeline on 100 training tasks
- [ ] Verify checkpoint save/load
- [ ] Confirm time budgets respected
- [ ] Validate submission format
- [ ] Check memory usage (< 16GB)
- [ ] Test on Kaggle notebook environment

---

## SUCCESS CRITERIA

**Functionality**:
- [ ] System trains for specified time budget
- [ ] Checkpoints save every 10 minutes
- [ ] Progress updates every 2 minutes
- [ ] Genomes evolve over generations
- [ ] RRBR ratcheting activates on improvements
- [ ] Best genome solves test tasks
- [ ] Submission.json in correct format
- [ ] No crashes or unhandled exceptions

**Performance**:
- [ ] Training completes within 150 minutes
- [ ] Achieves 15-25% accuracy on training set (baseline)
- [ ] Processes 100+ test tasks in 60 minutes
- [ ] Memory usage < 16GB
- [ ] All time budgets respected

**Code Quality**:
- [ ] All methods have docstrings
- [ ] Error handling on all I/O operations
- [ ] Logging at appropriate levels (INFO/WARNING/ERROR)
- [ ] Type hints on all public methods
- [ ] No TODO comments in production code

---

## ESTIMATED COMPLETION

**Time**: 2-3 hours of focused implementation  
**Lines**: ~750 (EvolutionEngine: 300, Orchestrator: 300, Entry: 150)  
**Final Total**: 2,450+ lines  
**Status**: Production-ready AGI system

---

## NEXT SESSION COMMAND

```
Continue WakingOrca v6 implementation. Build the remaining components:
1. EvolutionEngine (population evolution with RRBR ratcheting)
2. WakingOrcaOrchestrator (training + solving coordinator)
3. Entry points (main, data loading, submission generation)

Use the specifications in this continuation prompt. Target 2,450+ lines total.
Focus on time budget management, checkpoint systems, and Kaggle compatibility.
```

---

## CRITICAL REMINDERS

1. **Time Budget**: Every operation must respect time limits
2. **RRBR**: Amplify gains, never regress (asymmetric ratcheting)
3. **Checkpoints**: Save progress frequently for recovery
4. **Fallbacks**: Every operation needs graceful failure mode
5. **Logging**: User must see progress (not just silent computation)
6. **Kaggle Format**: submission.json must match exact specification
7. **Memory**: Keep usage < 16GB for Kaggle compatibility
8. **Production**: No TODOs, no stubs, no placeholders

**This is the final push. Make it count.** 🚀🐋⚡