# MSEGT – Meta-Symbolic Energy-Guided Transformer

This repository contains research code for an olympiad-level mathematical
reasoning system built around:

- NVIDIA GPT-OSS-120B Eagle3 as the base language model
- Symbolic math modules for combinatorics, geometry, number theory
- A Process Reward Model (PRM) trained on chain-of-thought traces
- Optional Hamiltonian (energy-stable) attention layers
- Lightweight MCTS-style search over reasoning trajectories
- Optional meta-learning (Reptile / MAML-style adapters)

The codebase is structured so you can:

1. Train a PRM on top of Eagle embeddings (mid-scale, 1×H100).
2. Add lightweight adapters / LoRA to specialize to AIMO-style tasks.
3. Run an inference-time search that combines:
   - multiple solution samples,
   - PRM scoring,
   - symbolic sanity checks.

This repo is **not** a polished library; it is a research sandbox.
