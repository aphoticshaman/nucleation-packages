# QUANTUM_ML.skill.md

## Quantum Machine Learning: QAOA, VQE, and Variational Quantum Algorithms

**Version**: 1.0
**Domain**: Quantum Computing, Optimization, Machine Learning
**Prerequisites**: Linear algebra, basic quantum mechanics concepts, ML fundamentals
**Complexity**: Advanced

---

## 1. EXECUTIVE SUMMARY

Quantum ML uses quantum circuits as parameterized models trained via classical optimization. This skill covers variational quantum algorithms (QAOA, VQE), quantum neural networks, quantum kernels, and the path to practical quantum advantage.

**Current Reality**: We're in the NISQ (Noisy Intermediate-Scale Quantum) era. Useful quantum ML requires noise-aware algorithms and careful problem selection.

---

## 2. QUANTUM COMPUTING BASICS

### 2.1 Qubits

**Classical bit**: 0 or 1
**Qubit**: α|0⟩ + β|1⟩ (superposition)

Where |α|² + |β|² = 1.

**Measurement**: Collapses to |0⟩ with probability |α|², |1⟩ with probability |β|².

### 2.2 Multiple Qubits

Two qubits: α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩

**Entanglement**: Cannot write state as product of individual qubit states. Correlations stronger than classical.

### 2.3 Quantum Gates

| Gate | Matrix | Effect |
|------|--------|--------|
| X | [[0,1],[1,0]] | Bit flip |
| Z | [[1,0],[0,-1]] | Phase flip |
| H | [[1,1],[1,-1]]/√2 | Superposition |
| CNOT | Control-target flip | Entanglement |
| RX(θ) | Rotate around X | Parameterized |
| RY(θ) | Rotate around Y | Parameterized |
| RZ(θ) | Rotate around Z | Parameterized |

### 2.4 Quantum Circuits

```
|0⟩ ─H─●─RY(θ₁)─M
      │
|0⟩ ─X─RZ(θ₂)──M
```

Read left to right. H creates superposition, CNOT (●-X) entangles, RY/RZ are parameterized rotations, M measures.

---

## 3. VARIATIONAL QUANTUM ALGORITHMS (VQAs)

### 3.1 The Variational Principle

```
┌─────────────────────────────────────────┐
│      Classical Optimizer                 │
│  (Adam, COBYLA, SPSA, etc.)              │
└────────────────┬─────────────┬──────────┘
                 │             │
            Parameters     Gradient
                 │             │
                 ▼             │
┌────────────────────────────────────────┐
│      Quantum Circuit U(θ)               │
│      (Ansatz)                           │
└────────────────┬───────────────────────┘
                 │
            Expectation
                 │
                 ▼
┌────────────────────────────────────────┐
│      Cost Function                      │
│      ⟨ψ(θ)|H|ψ(θ)⟩                      │
└────────────────────────────────────────┘
```

### 3.2 VQA Components

**Ansatz**: Parameterized quantum circuit U(θ)
**Cost function**: What to minimize (energy, distance, etc.)
**Optimizer**: Classical algorithm to update θ
**Measurement**: Extract information from quantum state

### 3.3 Expressibility vs. Trainability

**Expressibility**: How many states can the ansatz reach?
**Trainability**: Can we find good parameters?

**Barren Plateaus**: Random deep circuits have exponentially vanishing gradients. Solutions:
- Shallow circuits
- Structured ansätze
- Layer-wise training
- Clever initialization

---

## 4. QAOA (Quantum Approximate Optimization Algorithm)

### 4.1 Purpose

Solve combinatorial optimization problems:
- MaxCut
- Traveling salesman
- Portfolio optimization
- Scheduling

### 4.2 Problem Encoding

Optimization problem → Ising Hamiltonian:

```
H_C = Σᵢⱼ Jᵢⱼ ZᵢZⱼ + Σᵢ hᵢ Zᵢ
```

Solution = ground state of H_C.

### 4.3 QAOA Circuit

Alternating layers of:
- **Problem unitary**: U_C(γ) = exp(-iγH_C)
- **Mixer unitary**: U_M(β) = exp(-iβH_M)

Where H_M = ΣᵢXᵢ.

```
|+⟩^n ─[U_C(γ₁)]─[U_M(β₁)]─[U_C(γ₂)]─[U_M(β₂)]─...─ Measure
```

### 4.4 QAOA Implementation

```python
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA

# Define problem (MaxCut on graph)
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Graph as QUBO
qp = QuadraticProgram()
qp.binary_var('x0')
qp.binary_var('x1')
qp.binary_var('x2')
qp.maximize(linear={'x0': 1}, quadratic={('x0','x1'): -2, ('x1','x2'): -2})

# Run QAOA
qaoa = QAOA(optimizer=COBYLA(), reps=3)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)
```

### 4.5 QAOA Parameters

**p (depth)**: Number of layers. Higher p → better approximation, harder to optimize.
**Typical range**: p = 1-5 for NISQ devices.

**Initialization**:
- Random (often bad)
- Linear schedule (γₖ = kπ/2p, βₖ = (1-k/p)π/2)
- Transfer from smaller p

---

## 5. VQE (Variational Quantum Eigensolver)

### 5.1 Purpose

Find ground state energy of molecular/physical systems:
- Chemistry simulation
- Materials science
- Drug discovery

### 5.2 Hamiltonian

Molecular Hamiltonian in second quantization:
```
H = Σᵢⱼ hᵢⱼ aᵢ†aⱼ + Σᵢⱼₖₗ hᵢⱼₖₗ aᵢ†aⱼ†aₖaₗ
```

Mapped to qubits via Jordan-Wigner or Bravyi-Kitaev transformation.

### 5.3 VQE Ansätze

**UCCSD (Unitary Coupled Cluster Singles Doubles)**:
- Chemistry-inspired
- Deep circuits
- Expressive but hard to train

**Hardware-efficient**:
```python
def hardware_efficient_ansatz(num_qubits, depth):
    qc = QuantumCircuit(num_qubits)
    params = []

    for d in range(depth):
        # Rotation layer
        for i in range(num_qubits):
            theta = Parameter(f'θ_{d}_{i}')
            params.append(theta)
            qc.ry(theta, i)

        # Entanglement layer
        for i in range(num_qubits - 1):
            qc.cx(i, i+1)

    return qc, params
```

### 5.4 VQE Implementation

```python
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator

# Define molecule
driver = PySCFDriver(atom='H 0 0 0; H 0 0 0.735', basis='sto-3g')
problem = driver.run()

# Map to qubit Hamiltonian
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

# Run VQE
vqe = VQE(
    estimator=Estimator(),
    ansatz=ansatz,
    optimizer=COBYLA()
)
result = vqe.compute_minimum_eigenvalue(hamiltonian)
print(f"Ground state energy: {result.eigenvalue}")
```

### 5.5 Measurement Overhead

Hamiltonian has many terms. Each term needs separate measurement basis.

**Grouping**: Measure commuting terms together
**Shots**: More measurements → better precision (error ~ 1/√shots)

---

## 6. QUANTUM NEURAL NETWORKS

### 6.1 Parameterized Quantum Circuits (PQCs)

```
Classical input x → Encoding → Parameterized layers → Measurement → Output
```

### 6.2 Data Encoding

**Basis encoding**: x = 5 → |101⟩
**Amplitude encoding**: x⃗ → Σᵢ xᵢ|i⟩ (efficient but hard to prepare)
**Angle encoding**: xᵢ → RY(xᵢ)|0⟩

```python
def angle_encoding(qc, x):
    for i, val in enumerate(x):
        qc.ry(val * np.pi, i)
```

### 6.3 Quantum Classifier

```python
class QuantumClassifier:
    def __init__(self, num_qubits, num_layers):
        self.qc = QuantumCircuit(num_qubits)
        self.params = []

        # Build parameterized circuit
        for layer in range(num_layers):
            # Rotation layer
            for i in range(num_qubits):
                theta = Parameter(f'θ_{layer}_{i}')
                self.params.append(theta)
                self.qc.ry(theta, i)

            # Entanglement
            for i in range(num_qubits - 1):
                self.qc.cx(i, i+1)

    def forward(self, x, params):
        # Encode input
        encoded = self.encode(x)

        # Bind parameters
        bound_circuit = encoded.bind_parameters(dict(zip(self.params, params)))

        # Execute and measure
        result = execute(bound_circuit)
        return self.decode(result)

    def train(self, X, y):
        def loss(params):
            predictions = [self.forward(x, params) for x in X]
            return cross_entropy(predictions, y)

        # Optimize
        result = minimize(loss, initial_params, method='COBYLA')
        return result.x
```

### 6.4 Quantum Kernels

Kernel trick with quantum feature maps:

```python
from qiskit_machine_learning.kernels import QuantumKernel

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
kernel = QuantumKernel(feature_map=feature_map)

# Compute kernel matrix
K = kernel.evaluate(X_train)

# Use with classical SVM
from sklearn.svm import SVC
svm = SVC(kernel='precomputed')
svm.fit(K, y_train)
```

**Advantage**: Potentially hard-to-compute classically kernel, leading to better separation.

---

## 7. NOISE AND ERROR MITIGATION

### 7.1 NISQ Noise Sources

- **Gate errors**: ~0.1-1% per gate
- **Measurement errors**: ~1-5% per qubit
- **Decoherence**: Qubits lose information over time
- **Crosstalk**: Gates affect neighboring qubits

### 7.2 Error Mitigation Techniques

**Zero-noise extrapolation**:
```python
# Run at different noise levels
results = []
for scale in [1.0, 1.5, 2.0]:
    noisy_result = run_with_noise_scale(circuit, scale)
    results.append(noisy_result)

# Extrapolate to zero noise
ideal = extrapolate(results)
```

**Measurement error mitigation**:
```python
from qiskit.utils.mitigation import CompleteMeasFitter

# Calibration circuits
meas_calibs = complete_meas_cal(qr)
cal_results = execute(meas_calibs)

# Build filter
meas_fitter = CompleteMeasFitter(cal_results)
filter = meas_fitter.filter

# Apply to results
mitigated = filter.apply(noisy_results)
```

**Probabilistic error cancellation**: Cancel errors statistically (expensive in shots)

### 7.3 Noise-Aware Training

```python
def noisy_expectation(circuit, params, shots=1000):
    # Simulate with noise model
    noise_model = get_device_noise()
    backend = AerSimulator(noise_model=noise_model)

    bound = circuit.bind_parameters(params)
    job = execute(bound, backend, shots=shots)

    return compute_expectation(job.result())
```

---

## 8. PRACTICAL CONSIDERATIONS

### 8.1 When to Use Quantum ML

**Potentially useful**:
- Optimization with specific structure (combinatorial)
- Certain kernel methods
- Simulation of quantum systems
- Very specific data structures

**Probably not useful**:
- General classification/regression
- Tasks with lots of data (classical ML wins)
- Anything requiring deep circuits (noise kills you)

### 8.2 Circuit Depth Limits

**Current devices**: ~100 useful gates before noise dominates
**Implication**: Shallow circuits only

### 8.3 Qubit Count

**Current devices**: 10-1000 qubits
**But**: Connectivity limited, not all qubits equal quality

### 8.4 Shot Budget

Each measurement gives one sample. Statistical precision:
```
Error ~ 1/√shots
```

Typical: 1000-10000 shots per circuit evaluation.

---

## 9. FRAMEWORKS

### 9.1 Qiskit (IBM)

```python
from qiskit import QuantumCircuit
from qiskit_algorithms import VQE
from qiskit_ibm_runtime import QiskitRuntimeService

# Connect to IBM Quantum
service = QiskitRuntimeService()
backend = service.least_busy(operational=True)

# Run VQE
result = vqe.run(backend)
```

### 9.2 PennyLane (Xanadu)

```python
import pennylane as qml

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def circuit(params, x):
    # Encode input
    qml.AngleEmbedding(x, wires=range(4))

    # Parameterized layers
    qml.BasicEntanglerLayers(params, wires=range(4))

    return qml.expval(qml.PauliZ(0))

# Train with PyTorch/JAX
params = np.random.random((3, 4))
opt = qml.GradientDescentOptimizer(stepsize=0.1)
for step in range(100):
    params = opt.step(cost, params)
```

### 9.3 Cirq (Google)

```python
import cirq

qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure(*qubits, key='result')
)

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
```

---

## 10. OUTLOOK

### 10.1 Near-term (NISQ)

- Error mitigation essential
- Shallow circuits only
- Hybrid algorithms (VQE, QAOA)
- Focus on specific applications

### 10.2 Medium-term (Early Fault-Tolerant)

- Error correction enables deeper circuits
- Quantum advantage for specific problems
- New algorithms possible

### 10.3 Long-term

- General quantum advantage
- Exponential speedups for simulation
- New ML paradigms

---

## 11. REFERENCES

**Foundational**:
- Peruzzo et al. (2014). VQE. *Nature Communications*
- Farhi et al. (2014). QAOA. arXiv:1411.4028
- McClean et al. (2016). Hybrid quantum-classical algorithms

**Reviews**:
- Bharti et al. (2022). Noisy intermediate-scale quantum algorithms. *Reviews of Modern Physics*
- Cerezo et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*

**Textbook**:
- Nielsen & Chuang. *Quantum Computation and Quantum Information*

---

## 12. VERSION HISTORY

- **v1.0** (2024-11): Initial comprehensive version

---

*Quantum ML is currently more about understanding when NOT to use it. The real opportunities are in specific applications where quantum structure matches problem structure.*
