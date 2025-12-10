# Appendix D: Glossary of AI and Machine Learning Terms

*500+ terms from activation functions to zero-shot learning*

---

## A

**A/B Testing** — Statistical method comparing two versions of a system to determine which performs better. Essential for evaluating AI model improvements.

**Ablation Study** — Experimental technique where components are removed from a model to understand their contribution to performance.

**Accuracy** — Fraction of predictions that are correct. = (TP + TN) / (TP + TN + FP + FN).

**Activation Function** — Non-linear function applied to a neuron's output. Examples: ReLU, sigmoid, tanh. Enables networks to learn non-linear patterns.

**Activation Maximization** — Technique for visualizing what a neural network has learned by generating inputs that maximally activate a particular neuron.

**Actor-Critic** — Reinforcement learning architecture combining a policy network (actor) and value network (critic).

**Adam Optimizer** — Adaptive Moment Estimation. Popular optimizer combining momentum and adaptive learning rates. Often the default choice.

**Adversarial Attack** — Input specifically crafted to cause a model to make errors. Often imperceptible to humans but fool neural networks.

**Adversarial Example** — Input modified with small perturbations that cause misclassification. Demonstrates fragility of neural networks.

**Adversarial Training** — Training on adversarial examples to improve robustness.

**Agent** — Entity that perceives its environment and takes actions to maximize some objective.

**AGI (Artificial General Intelligence)** — Hypothetical AI with human-level cognitive abilities across all domains.

**AI Alignment** — Research area focused on ensuring AI systems pursue intended goals and remain beneficial.

**AI Safety** — Field studying how to develop AI systems that are safe and beneficial.

**AI Winter** — Period of reduced funding and interest in AI research, typically following overinflated expectations.

**Algorithmic Bias** — Systematic errors in AI systems that create unfair outcomes for certain groups.

**Alignment Tax** — Performance cost incurred by making a model safer or more aligned with human values.

**Annotation** — Process of labeling data for supervised learning.

**Anomaly Detection** — Identifying data points that differ significantly from the majority.

**API (Application Programming Interface)** — Interface allowing software systems to communicate. LLM APIs expose model capabilities.

**Attention** — Mechanism allowing neural networks to focus on relevant parts of input. Core component of transformers.

**Attention Head** — One parallel attention computation in multi-head attention.

**Attention Mask** — Binary tensor indicating which positions should be attended to.

**Attention Score** — Measure of relevance between query and key positions.

**Attention Weights** — Normalized attention scores indicating how much each position influences the output.

**Autoencoder** — Neural network trained to reconstruct its input through a bottleneck layer, learning compressed representations.

**Autoregressive Model** — Model generating outputs sequentially, where each output depends on previous outputs. GPT is autoregressive.

**Average Pooling** — Aggregation operation computing the mean of values in a region.

---

## B

**Backpropagation** — Algorithm for computing gradients in neural networks by propagating error backwards through layers.

**Bag of Words** — Text representation treating documents as unordered collections of words, ignoring grammar and word order.

**Bagging** — Bootstrap Aggregating. Ensemble method training multiple models on random subsets of data.

**Base Model** — Pre-trained model before fine-tuning or adaptation for specific tasks.

**Batch** — Subset of training data processed together in one forward/backward pass.

**Batch Normalization** — Technique normalizing layer inputs to accelerate training and improve stability.

**Batch Size** — Number of examples processed in one training step.

**Bayesian Neural Network** — Neural network with probability distributions over weights, enabling uncertainty quantification.

**Beam Search** — Search algorithm exploring multiple candidates during generation, keeping top-k most promising sequences.

**BERT** — Bidirectional Encoder Representations from Transformers. Influential pre-trained language model from Google (2018).

**Bias (Statistical)** — Systematic error in model predictions. Bias-variance tradeoff is fundamental in ML.

**Bias (Neural Network)** — Additive parameter in neurons, allowing activation functions to shift.

**Bidirectional** — Processing sequence in both directions. BERT is bidirectional; GPT is unidirectional.

**Binary Classification** — Classification task with exactly two classes.

**BLEU Score** — Bilingual Evaluation Understudy. Metric for evaluating machine translation quality.

**Bootstrapping** — Sampling with replacement to estimate statistics or train multiple models.

**BPE (Byte Pair Encoding)** — Tokenization algorithm that iteratively merges frequent character pairs.

**Byte-Level BPE** — BPE operating on bytes rather than characters, enabling any text without unknown tokens.

---

## C

**Calibration** — How well predicted probabilities match actual frequencies of outcomes.

**Capsule Network** — Neural network architecture using groups of neurons (capsules) to represent entity properties.

**Catastrophic Forgetting** — Phenomenon where neural networks forget previously learned tasks when trained on new ones.

**Causal Language Model** — Language model predicting next token based only on previous tokens (left-to-right).

**Chain-of-Thought (CoT)** — Prompting technique eliciting step-by-step reasoning from language models.

**Channel** — Feature dimension in convolutional networks. RGB images have 3 channels.

**Chatbot** — Conversational AI system designed for interactive dialogue.

**Checkpoint** — Saved state of model weights during training, enabling resume or evaluation.

**CIC Framework** — Compression-Integration-Coherence. Information-theoretic framework for understanding and improving LLM outputs.

**Class Imbalance** — When training data has significantly different numbers of examples per class.

**Classification** — Task of assigning inputs to discrete categories.

**Classifier** — Model that performs classification.

**Claude** — AI assistant developed by Anthropic, trained with Constitutional AI methods.

**Clipping** — Limiting values to a specified range. Gradient clipping prevents exploding gradients.

**Cluster** — Group of similar data points.

**Clustering** — Unsupervised learning task of grouping similar data points.

**Co-occurrence** — Frequency with which items appear together. Foundation of word embeddings.

**Codex** — OpenAI model specialized for code generation, powers GitHub Copilot.

**Cold Start Problem** — Challenge of making recommendations without historical data.

**Compute** — Computational resources (GPUs, TPUs) required for AI training and inference.

**Conditional Generation** — Generating outputs based on some conditioning input (e.g., prompt, image).

**Confidence Score** — Model's estimate of how certain it is about a prediction.

**Constitutional AI** — Anthropic's approach to AI alignment using AI-generated feedback based on a "constitution" of principles.

**Context Length** — Maximum number of tokens a model can process in one forward pass.

**Context Window** — Range of tokens the model can attend to. GPT-4 has up to 128k token context.

**Contrastive Learning** — Learning representations by contrasting similar and dissimilar examples.

**Convolution** — Operation applying a filter across an input, detecting local patterns.

**Convolutional Neural Network (CNN)** — Neural network using convolutional layers, effective for grid-like data (images).

**Corpus** — Large collection of text used for training language models.

**Cosine Similarity** — Measure of similarity between vectors based on angle, not magnitude. = dot(A,B) / (||A|| × ||B||).

**Cross-Attention** — Attention where queries come from one sequence and keys/values from another.

**Cross-Entropy Loss** — Loss function measuring difference between predicted and actual probability distributions. Standard for classification.

**Cross-Validation** — Technique evaluating models by training on subsets and testing on held-out portions.

**Curriculum Learning** — Training strategy presenting examples in meaningful order, typically easy to hard.

---

## D

**DALL-E** — OpenAI's text-to-image model generating images from text descriptions.

**Data Augmentation** — Artificially increasing training data by applying transformations to existing examples.

**Data Leakage** — When information from test data inadvertently enters training, leading to overoptimistic evaluation.

**Data Parallelism** — Distributed training strategy where same model processes different data on different devices.

**Dataset** — Collection of examples used for training or evaluating models.

**Dead Neuron** — Neuron that always outputs zero, often due to ReLU with negative inputs.

**Decision Boundary** — Surface in feature space separating different classes.

**Decision Tree** — Model making predictions through a tree of binary decisions on features.

**Decoder** — Component generating output from encoded representation. In transformers, decoder generates tokens autoregressively.

**Decoder-Only** — Transformer architecture using only the decoder component. GPT is decoder-only.

**Deep Learning** — Machine learning using neural networks with many layers.

**DeepMind** — AI research lab (now part of Google) behind AlphaGo, AlphaFold, and Gemini.

**Dense Layer** — Fully connected layer where every input connects to every output.

**Depth** — Number of layers in a neural network.

**Deterministic** — Producing the same output given the same input. Contrast with stochastic.

**Diffusion Model** — Generative model learning to reverse a noise-adding process. Powers Stable Diffusion, DALL-E 3.

**Dimensionality Reduction** — Reducing the number of features while preserving important information.

**Direct Preference Optimization (DPO)** — Fine-tuning technique using preference data without explicit reward model.

**Discriminator** — Component in GANs that distinguishes real from generated samples.

**Distillation** — See Knowledge Distillation.

**Distributed Training** — Training models across multiple devices or machines.

**Domain Adaptation** — Adapting a model trained on one domain to perform well on another.

**Downstream Task** — Task the model is applied to after pre-training.

**Dropout** — Regularization technique randomly zeroing neuron outputs during training.

---

## E

**Early Stopping** — Stopping training when validation performance stops improving to prevent overfitting.

**Edge Case** — Unusual or extreme input that may cause unexpected model behavior.

**Eigenvalue** — Scalar factor by which an eigenvector is scaled when transformed by a matrix.

**Eigenvector** — Vector whose direction is unchanged by a linear transformation.

**ELBO** — Evidence Lower Bound. Objective function for variational autoencoders.

**Embedding** — Dense vector representation of discrete items (words, tokens, entities).

**Embedding Dimension** — Size of embedding vectors. GPT-3 uses 12,288-dimensional embeddings.

**Emergence** — New capabilities appearing in models at scale that weren't present at smaller scales.

**Encoder** — Component converting input into internal representation.

**Encoder-Decoder** — Architecture with separate encoding and decoding stages. T5, BART are encoder-decoder.

**Encoder-Only** — Architecture using only the encoder. BERT is encoder-only.

**End-to-End Learning** — Training a complete system from input to output without intermediate supervision.

**Ensemble** — Combination of multiple models to improve performance.

**Entity Extraction** — Identifying and classifying named entities (people, places, organizations) in text.

**Entropy** — Measure of uncertainty or randomness. In information theory, average bits needed to encode a random variable.

**Epoch** — One complete pass through the training dataset.

**Evaluation Metric** — Quantitative measure of model performance.

**Expert System** — AI system encoding human expert knowledge as rules. Dominant in 1980s.

**Explainability** — Ability to understand and explain model predictions.

**Exploding Gradient** — Problem where gradients become very large during training, causing instability.

**Exploration vs Exploitation** — Tradeoff between trying new actions and using known good actions.

**Exponential Moving Average (EMA)** — Smoothing technique giving more weight to recent values.

**Expression** — Mathematical representation of computation in symbolic AI.

---

## F

**F1 Score** — Harmonic mean of precision and recall. = 2 × (P × R) / (P + R).

**False Negative (FN)** — Incorrect prediction that something is absent when present.

**False Positive (FP)** — Incorrect prediction that something is present when absent.

**Feature** — Individual measurable property of a data point.

**Feature Engineering** — Creating new features from raw data to improve model performance.

**Feature Extraction** — Using a pre-trained model to compute representations of new data.

**Feature Map** — Output of a convolutional layer; activation pattern for a particular filter.

**Feature Selection** — Choosing which features to include in a model.

**Feature Vector** — Numerical representation of an object as a list of features.

**Federated Learning** — Training models across decentralized data sources without sharing raw data.

**Few-Shot Learning** — Learning from very few examples. GPT-3 demonstrated surprising few-shot abilities.

**Fine-Tuning** — Further training a pre-trained model on task-specific data.

**FLOP** — Floating Point Operation. Measure of computational work.

**Foundation Model** — Large pre-trained model adapted to many downstream tasks.

**Frequency Encoding** — Representing information in the frequency domain.

**Frontier Model** — State-of-the-art model at the capability frontier.

**Full Fine-Tuning** — Fine-tuning all model parameters (vs parameter-efficient methods).

**Fully Connected Layer** — Layer where every input is connected to every output.

**Function Approximation** — Learning a function from input-output examples.

---

## G

**GAN (Generative Adversarial Network)** — Architecture where generator and discriminator compete, producing realistic generations.

**Gaussian Distribution** — Normal distribution. Bell curve defined by mean and variance.

**Gaussian Noise** — Random noise drawn from Gaussian distribution.

**GELU** — Gaussian Error Linear Unit. Smooth activation function popular in transformers.

**Gemini** — Google's multimodal AI model family.

**Generalization** — Model's ability to perform well on unseen data.

**Generalization Gap** — Difference between training and test performance.

**Generative AI** — AI systems that generate new content (text, images, audio, video).

**Generative Model** — Model learning the distribution of data to generate new samples.

**Generator** — Component in GANs that creates synthetic samples.

**Gini Impurity** — Measure of impurity used in decision tree splitting.

**Global Minimum** — Lowest point in a loss landscape.

**GloVe** — Global Vectors for Word Representation. Word embedding method using co-occurrence statistics.

**Goodhart's Law** — "When a measure becomes a target, it ceases to be a good measure." Relevant to reward hacking.

**GPT** — Generative Pre-trained Transformer. OpenAI's language model series.

**GPT-3** — 175 billion parameter language model from OpenAI (2020).

**GPT-4** — Multimodal model from OpenAI (2023), rumored to be ~1.7 trillion parameters.

**GPU (Graphics Processing Unit)** — Hardware accelerator essential for deep learning.

**Gradient** — Vector of partial derivatives indicating direction of steepest increase.

**Gradient Accumulation** — Accumulating gradients across multiple batches before updating weights.

**Gradient Checkpointing** — Trading compute for memory by recomputing activations during backward pass.

**Gradient Clipping** — Limiting gradient magnitude to prevent exploding gradients.

**Gradient Descent** — Optimization algorithm moving in direction opposite to gradient.

**Gradient Flow** — How gradients propagate through a network during backpropagation.

**Graph Neural Network (GNN)** — Neural network operating on graph-structured data.

**Greedy Decoding** — Generating by selecting highest-probability token at each step.

**Ground Truth** — Correct labels or outputs for supervised learning.

**Grounding** — Connecting model outputs to real-world facts or evidence.

---

## H

**Hallucination** — Model generating plausible but factually incorrect information.

**Hardcoded** — Fixed in the program, not learned from data.

**Hardware Accelerator** — Specialized hardware for ML computation (GPUs, TPUs).

**Head (Attention)** — One attention computation in multi-head attention.

**Head (Model)** — Final layer(s) adapted for specific tasks.

**Heuristic** — Rule-of-thumb approach that often works but lacks guarantees.

**Hidden Layer** — Layer between input and output layers in a neural network.

**Hidden State** — Internal state of a recurrent neural network.

**Hierarchical** — Organized in levels of abstraction or detail.

**Hinton, Geoffrey** — Pioneer of deep learning; developed backpropagation, Boltzmann machines.

**Holdout Set** — Data reserved for evaluation, not used in training.

**Homogeneous** — All elements of the same type.

**Hugging Face** — Company and platform hosting open-source ML models and datasets.

**Human-in-the-Loop** — System incorporating human feedback or oversight.

**Hyperparameter** — Configuration setting not learned from data (learning rate, batch size, etc.).

**Hyperparameter Optimization** — Searching for best hyperparameter values.

**Hypothesis** — Candidate model or explanation being evaluated.

---

## I

**Image Classification** — Assigning images to predefined categories.

**Image Generation** — Creating new images, often from text descriptions.

**Image Segmentation** — Labeling each pixel of an image with a class.

**ImageNet** — Large image dataset (14M images, 1000 classes) that drove computer vision progress.

**Imputation** — Filling in missing data values.

**In-Context Learning (ICL)** — Model adapting behavior based on examples in the prompt, without weight updates.

**Inference** — Using a trained model to make predictions on new data.

**Inference Time** — Time required to generate predictions.

**Information Bottleneck** — Theory for understanding representations that preserve relevant information while compressing.

**Information Gain** — Reduction in entropy from splitting data on a feature.

**Information Theory** — Mathematical study of quantifying and transmitting information. Foundation of CIC.

**Initialize** — Set initial values of model parameters before training.

**Input** — Data fed into a model.

**Instance** — Single example or data point.

**Instruction Tuning** — Fine-tuning models to follow natural language instructions.

**Interpretability** — Ability to understand what a model has learned and why it makes particular predictions.

**Invariance** — Property of being unchanged under certain transformations.

**Inverse Reinforcement Learning** — Learning reward function from observed behavior.

---

## J

**Jacobian** — Matrix of partial derivatives of a vector-valued function.

**Jailbreak** — Prompting technique that bypasses model safety guardrails.

**Joint Probability** — Probability of multiple events occurring together.

**Jupyter Notebook** — Interactive coding environment popular for ML experimentation.

---

## K

**K-Means** — Clustering algorithm partitioning data into K clusters based on centroid distance.

**K-Nearest Neighbors (KNN)** — Algorithm classifying based on labels of nearest training examples.

**Kernel** — Small matrix used in convolution operations. Also: function computing similarity in kernel methods.

**Kernel Trick** — Using kernel functions to implicitly compute in high-dimensional space.

**Key** — In attention, the representation queries are compared against.

**KL Divergence** — Kullback-Leibler Divergence. Measure of difference between probability distributions.

**Knowledge Distillation** — Training smaller model to mimic larger model's outputs.

**Knowledge Graph** — Graph representing entities and relationships as nodes and edges.

---

## L

**L1 Regularization** — Penalty on sum of absolute values of weights. Encourages sparsity.

**L2 Regularization** — Penalty on sum of squared weights. Also called weight decay.

**Label** — Correct output for supervised learning.

**Label Smoothing** — Replacing hard labels with soft probabilities to improve generalization.

**LaMDA** — Language Model for Dialogue Applications. Google's conversational AI (2021).

**Langchain** — Framework for building applications with language models.

**Language Model** — Model predicting probability of text sequences.

**Large Language Model (LLM)** — Language model with billions of parameters trained on massive text corpora.

**Latent Space** — Continuous space of learned representations.

**Latent Variable** — Unobserved variable inferred from data.

**Layer** — Component of a neural network performing a specific transformation.

**Layer Normalization** — Normalization technique normalizing across features rather than batch.

**Leaky ReLU** — ReLU variant with small slope for negative inputs.

**Learning Rate** — Step size for gradient descent updates. Critical hyperparameter.

**Learning Rate Schedule** — Strategy for adjusting learning rate during training.

**LeCun, Yann** — Pioneer of convolutional neural networks; Chief AI Scientist at Meta.

**Linear Layer** — Layer computing y = Wx + b.

**Linear Regression** — Model predicting continuous output as linear combination of features.

**LISP** — Programming language historically important for AI research.

**LLaMA** — Large Language Model Meta AI. Meta's open-source language models.

**Local Minimum** — Point where loss is lower than neighbors but not globally lowest.

**Log Likelihood** — Logarithm of probability, often used as objective function.

**Logistic Regression** — Classification model using sigmoid function.

**Logit** — Unnormalized prediction before softmax. Also: log-odds ratio.

**Long-Range Dependency** — Relationship between distant positions in a sequence.

**LoRA** — Low-Rank Adaptation. Parameter-efficient fine-tuning method.

**Loss Function** — Function measuring how wrong model predictions are.

**Loss Landscape** — Surface of loss values over parameter space.

**LSTM** — Long Short-Term Memory. RNN variant with gates for learning long-term dependencies.

---

## M

**Machine Learning (ML)** — Field of study giving computers ability to learn from data without explicit programming.

**Machine Translation** — Automatically translating text between languages.

**Macro Average** — Average metric computed per class then averaged across classes.

**Marginal Probability** — Probability of a variable ignoring other variables.

**Markov Chain** — Sequence where each state depends only on the previous state.

**Markov Decision Process (MDP)** — Framework for modeling decision-making in stochastic environments.

**Masked Language Model** — Model trained to predict masked tokens. BERT uses this objective.

**Matrix** — Rectangular array of numbers.

**Max Pooling** — Aggregation taking maximum value in a region.

**Mean Squared Error (MSE)** — Loss function for regression. Average of squared differences.

**Meta-Learning** — Learning to learn; training models that can quickly adapt to new tasks.

**Micro Average** — Metric computed over all predictions aggregated together.

**Midjourney** — Commercial AI image generation service.

**Mini-Batch** — Small subset of training data used for one gradient update.

**Minimum Description Length (MDL)** — Principle that best model minimizes compressed length of data + model.

**Minsky, Marvin** — AI pioneer; co-authored "Perceptrons" (1969); co-founded MIT AI Lab.

**Mistral** — French AI company producing efficient open-source models.

**Mixed Precision** — Training with mix of float16 and float32 for efficiency.

**Mixture of Experts (MoE)** — Architecture activating different "expert" subnetworks for different inputs.

**MLP (Multi-Layer Perceptron)** — Fully connected feedforward neural network.

**MNIST** — Classic dataset of handwritten digits (70,000 images).

**Model** — Learned representation of patterns in data.

**Model Card** — Document describing model capabilities, limitations, and intended use.

**Model Parallelism** — Distributing different parts of a model across devices.

**Modular Arithmetic** — Arithmetic where numbers wrap around after reaching a certain value.

**Momentum** — Optimization technique using exponential moving average of gradients.

**Monte Carlo** — Methods using random sampling for estimation.

**Moore's Law** — Observation that computing power doubles approximately every two years.

**Multi-Class Classification** — Classification with more than two classes.

**Multi-Head Attention** — Running multiple attention operations in parallel with different learned projections.

**Multi-Label Classification** — Classification where examples can have multiple labels.

**Multi-Modal** — Involving multiple types of input (text, image, audio, etc.).

**Multi-Task Learning** — Training on multiple tasks simultaneously.

---

## N

**N-gram** — Contiguous sequence of N items (usually words).

**Named Entity Recognition (NER)** — Identifying and classifying named entities in text.

**Natural Language Generation (NLG)** — Producing human-readable text.

**Natural Language Processing (NLP)** — Processing and understanding human language.

**Natural Language Understanding (NLU)** — Comprehending meaning of human language.

**Negative Sampling** — Training technique sampling negative examples from noise distribution.

**Neural Architecture Search (NAS)** — Automated search for optimal network architectures.

**Neural Network** — Computing system inspired by biological neural networks.

**Neuron** — Basic computational unit in neural networks.

**Next Token Prediction** — Training objective of autoregressive language models.

**NLP** — See Natural Language Processing.

**Noise** — Random variation in data or signals.

**Non-Deterministic** — Producing potentially different outputs for same input.

**Non-Linearity** — Function that isn't linear; enables learning complex patterns.

**Norm** — Measure of vector magnitude. L2 norm = Euclidean distance.

**Normalization** — Scaling data to standard range or distribution.

**Normalized Compression Distance (NCD)** — Similarity metric based on compression.

**Nucleus Sampling** — Sampling from tokens comprising top-p probability mass.

---

## O

**Object Detection** — Identifying and locating objects in images.

**Objective Function** — Function being optimized during training.

**Observation** — Single data point or measurement.

**Occam's Razor** — Principle preferring simpler explanations.

**Offline Learning** — Learning from fixed dataset without interaction.

**One-Hot Encoding** — Representing categorical variable as binary vector.

**One-Shot Learning** — Learning from single example.

**Online Learning** — Learning incrementally from streaming data.

**OpenAI** — AI research company behind GPT, DALL-E, ChatGPT.

**Operator** — Mathematical function transforming inputs.

**Optimization** — Finding parameters that minimize (or maximize) an objective.

**Optimizer** — Algorithm for updating model parameters. Examples: SGD, Adam.

**Out-of-Distribution (OOD)** — Data different from training distribution.

**Output** — Model's prediction or generation.

**Overfit** — Model memorizing training data rather than learning generalizable patterns.

---

## P

**PaLM** — Pathways Language Model. Google's large language model (2022).

**Parallel Processing** — Executing multiple operations simultaneously.

**Parameter** — Learned value in a model (weight, bias).

**Parameter Count** — Total number of trainable parameters. GPT-3 has 175B parameters.

**Parameter-Efficient Fine-Tuning (PEFT)** — Methods fine-tuning only small portion of parameters.

**Parsing** — Analyzing structure of input (syntactic parsing, dependency parsing).

**Partial Derivative** — Derivative with respect to one variable, holding others constant.

**Perceptron** — Single-layer neural network; simplest neural network architecture.

**Perplexity** — Measure of how well a language model predicts text. Lower is better.

**Pipeline** — Sequence of processing steps.

**Policy** — In RL, mapping from states to actions.

**Policy Gradient** — RL method directly optimizing policy parameters.

**Pooling** — Aggregation operation reducing spatial dimensions.

**Positional Encoding** — Adding position information to transformer inputs.

**Positional Embedding** — Learned position representations.

**Posterior** — Probability distribution after observing data (Bayesian).

**PPO (Proximal Policy Optimization)** — Popular reinforcement learning algorithm used in RLHF.

**Pre-Training** — Initial training on large dataset before task-specific fine-tuning.

**Precision** — Fraction of positive predictions that are correct. = TP / (TP + FP).

**Prediction** — Model's output for a given input.

**Prior** — Initial probability distribution before observing data (Bayesian).

**Probability Distribution** — Function describing probabilities of outcomes.

**Probing** — Testing what information is encoded in model representations.

**PROLOG** — Logic programming language historically important for AI.

**Prompt** — Input text given to a language model.

**Prompt Engineering** — Craft of designing effective prompts.

**Prompt Injection** — Attack manipulating model behavior through crafted prompts.

**Pruning** — Removing unimportant parameters to reduce model size.

---

## Q

**Q-Learning** — Reinforcement learning algorithm learning action-value function.

**QLoRA** — Quantized Low-Rank Adaptation. Memory-efficient fine-tuning.

**Quantization** — Reducing numerical precision of model weights.

**Query** — In attention, the representation determining what to attend to.

**Question Answering** — Task of answering questions based on context or knowledge.

---

## R

**R-CNN** — Region-based CNN for object detection.

**Random Forest** — Ensemble of decision trees trained on random subsets.

**Random Initialization** — Starting training with random parameter values.

**Random Seed** — Value initializing random number generator for reproducibility.

**Rank** — In linear algebra, number of linearly independent rows/columns.

**RAG (Retrieval-Augmented Generation)** — Enhancing generation with retrieved documents.

**RBF (Radial Basis Function)** — Function depending only on distance from center.

**Recall** — Fraction of actual positives correctly identified. = TP / (TP + FN).

**Recurrent Neural Network (RNN)** — Network with connections forming cycles, processing sequences.

**Red Teaming** — Adversarial testing to find model vulnerabilities.

**Regression** — Predicting continuous numerical values.

**Regularization** — Techniques preventing overfitting.

**Reinforcement Learning (RL)** — Learning through interaction with environment and rewards.

**Reinforcement Learning from Human Feedback (RLHF)** — Training models using human preference feedback.

**ReLU** — Rectified Linear Unit. Activation function: max(0, x).

**Representation Learning** — Learning useful representations of data.

**Residual Connection** — Adding input to output of a layer (skip connection).

**ResNet** — Residual Network. CNN architecture with skip connections enabling very deep networks.

**Retrieval** — Finding relevant documents or information.

**Reward Function** — In RL, function specifying goals through numerical rewards.

**Reward Hacking** — Agent exploiting reward function in unintended ways.

**Reward Model** — Model predicting human preferences, used in RLHF.

**RLHF** — See Reinforcement Learning from Human Feedback.

**RMSprop** — Optimizer using adaptive learning rates based on recent gradient magnitudes.

**RoBERTa** — Robustly optimized BERT. Improved BERT training from Facebook.

**Robust** — Model performance maintained under perturbations or distribution shift.

**ROC Curve** — Receiver Operating Characteristic. Plot of true positive vs false positive rate.

**Rotary Position Embedding (RoPE)** — Position encoding using rotation matrices.

---

## S

**Saliency Map** — Visualization showing which input features most affect output.

**Sample** — Single data point; or, to draw from a distribution.

**Sampling** — Process of generating from a probability distribution.

**Sampling Temperature** — Parameter controlling randomness in generation.

**Scalar** — Single numerical value.

**Scale** — In AI, usually refers to model size, data size, or compute.

**Scaling Laws** — Empirical relationships between model performance and scale.

**Scheduling** — Varying hyperparameters during training.

**Scikit-Learn** — Popular Python machine learning library.

**Seed** — See Random Seed.

**Self-Attention** — Attention where queries, keys, and values come from same sequence.

**Self-Play** — Training by playing against itself (AlphaGo Zero).

**Self-Supervised Learning** — Learning from data without human-provided labels.

**Semantic Similarity** — Measure of meaning similarity between texts.

**Semi-Supervised Learning** — Learning from both labeled and unlabeled data.

**Sentence Embedding** — Vector representation of a sentence.

**Sentiment Analysis** — Determining emotional tone of text.

**Seq2Seq** — Sequence-to-sequence model architecture.

**Sequence** — Ordered list of items (tokens, frames, etc.).

**Sequence Labeling** — Assigning labels to each element of a sequence.

**SGD (Stochastic Gradient Descent)** — Gradient descent using random mini-batches.

**Shallow Learning** — Traditional ML methods without deep networks.

**SHAP** — SHapley Additive exPlanations. Method for explaining predictions.

**Sigmoid** — Activation function squashing values to [0, 1].

**Similarity** — Measure of how alike two objects are.

**Skip Connection** — See Residual Connection.

**Softmax** — Function converting logits to probabilities summing to 1.

**Sora** — OpenAI's video generation model (2024).

**Sparse** — Containing mostly zeros.

**Sparse Attention** — Attention patterns attending to subset of positions.

**Speculative Decoding** — Using smaller model to draft tokens verified by larger model.

**Stable Diffusion** — Open-source text-to-image diffusion model.

**State** — Current situation in an environment or system.

**State Space** — Set of all possible states.

**Statistical Learning** — Theoretical framework for machine learning.

**Step** — One parameter update in training.

**Stochastic** — Involving randomness.

**Stop Token** — Special token indicating end of generation.

**Stride** — Step size of convolution or pooling operation.

**Structured Prediction** — Predicting structured outputs (sequences, graphs, etc.).

**Student Model** — Smaller model learning from teacher in distillation.

**Subword** — Token representing part of a word.

**Supervised Learning** — Learning from labeled examples.

**SVM (Support Vector Machine)** — Model finding optimal separating hyperplane.

**Symbolic AI** — AI using explicit symbols and rules, contrasted with neural approaches.

**Synthetic Data** — Artificially generated training data.

**System Prompt** — Instructions given to model defining its behavior.

---

## T

**T5** — Text-to-Text Transfer Transformer. Google's encoder-decoder model.

**Tanh** — Hyperbolic tangent activation function.

**Target** — Correct output for supervised learning.

**Task** — Specific objective for a model to accomplish.

**Teacher Forcing** — Training by providing ground truth as input at each step.

**Teacher Model** — Larger model providing knowledge for distillation.

**Temperature** — Hyperparameter controlling randomness in sampling.

**Tensor** — Multi-dimensional array of numbers.

**TensorFlow** — Google's open-source machine learning framework.

**Test Set** — Data used for final model evaluation, never seen during training.

**Text Classification** — Categorizing text into predefined classes.

**Text Generation** — Producing text outputs.

**TF-IDF** — Term Frequency-Inverse Document Frequency. Text weighting scheme.

**Throughput** — Number of samples processed per unit time.

**Time Series** — Data points indexed by time.

**Token** — Basic unit of text for language models (word, subword, or character).

**Tokenization** — Converting text into tokens.

**Tokenizer** — Algorithm/model converting text to/from tokens.

**Top-K Sampling** — Sampling from K highest probability tokens.

**Top-P Sampling** — See Nucleus Sampling.

**TPU (Tensor Processing Unit)** — Google's custom AI accelerator chips.

**Training** — Process of learning model parameters from data.

**Training Data** — Data used to train a model.

**Training Loop** — Iterative process of forward pass, loss computation, backward pass, parameter update.

**Transfer Learning** — Applying knowledge from one task to another.

**Transformer** — Architecture based entirely on attention mechanisms.

**Tree** — Hierarchical data structure; also decision tree.

**Triplet Loss** — Loss function for learning embeddings from triplets of examples.

**True Negative (TN)** — Correct prediction that something is absent.

**True Positive (TP)** — Correct prediction that something is present.

**Truncation** — Cutting input to maximum allowed length.

**Tuning** — See Fine-Tuning or Hyperparameter Optimization.

**Turing, Alan** — Pioneer of computer science and AI; proposed the Turing Test.

**Turing Test** — Test of machine intelligence through conversation.

---

## U

**Uncertainty** — Degree of unpredictability or lack of knowledge.

**Uncertainty Quantification** — Estimating confidence in predictions.

**Underfitting** — Model too simple to capture patterns in data.

**Universal Approximation Theorem** — Neural networks can approximate any continuous function.

**Unlabeled Data** — Data without human-provided labels.

**Unsupervised Learning** — Learning patterns from unlabeled data.

**Update** — Change to model parameters during training.

**Upsampling** — Increasing spatial resolution.

---

## V

**Validation Set** — Data for evaluating model during development.

**Value** — In attention, the representation that gets aggregated.

**Value Function** — In RL, function estimating expected future reward.

**Vanishing Gradient** — Problem where gradients become very small in deep networks.

**VAE (Variational Autoencoder)** — Generative model learning latent representations.

**Variance** — Measure of spread in data or predictions.

**Vector** — Ordered list of numbers.

**Vector Database** — Database optimized for storing and searching embeddings.

**Vectorization** — Converting data to numerical vectors.

**Vision Transformer (ViT)** — Transformer architecture for images.

---

## W

**Warm Start** — Initializing from pre-trained weights.

**Warmup** — Gradually increasing learning rate at start of training.

**Weight** — Learnable parameter in a neural network.

**Weight Decay** — Regularization adding penalty on weight magnitudes.

**Weight Initialization** — Strategy for setting initial weight values.

**Weight Sharing** — Using same weights for multiple computations.

**Width** — Number of neurons per layer.

**Word Embedding** — Vector representation of words.

**Word2Vec** — Algorithm for learning word embeddings from text.

**Workflow** — Sequence of steps for completing a task.

---

## X

**Xavier Initialization** — Weight initialization scheme for better gradient flow.

**XGBoost** — Gradient boosting library popular for structured data.

**XOR Problem** — Classic demonstration that single perceptrons can't learn non-linear patterns.

---

## Y

**YOLO** — You Only Look Once. Real-time object detection architecture.

---

## Z

**Zero-Shot Learning** — Performing tasks without any task-specific examples.

**Zero-Shot Classification** — Classifying into categories not seen during training.

**ZeRO** — Zero Redundancy Optimizer. Memory-efficient distributed training.

---

## Symbol Reference

**∇** — Nabla. Gradient operator.

**∂** — Partial derivative.

**∑** — Summation.

**∏** — Product.

**∈** — Element of (set membership).

**∀** — For all (universal quantifier).

**∃** — There exists (existential quantifier).

**≈** — Approximately equal.

**∝** — Proportional to.

**→** — Implies / maps to.

**⊗** — Tensor product / outer product.

**⊕** — Direct sum / XOR.

**||x||** — Norm of x.

**argmax** — Argument of maximum.

**argmin** — Argument of minimum.

**log** — Logarithm (usually natural log in ML).

**exp** — Exponential function.

**σ** — Often used for sigmoid or standard deviation.

---

## Acronym Quick Reference

| Acronym | Full Name |
|---------|-----------|
| AGI | Artificial General Intelligence |
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| BERT | Bidirectional Encoder Representations from Transformers |
| BPE | Byte Pair Encoding |
| CNN | Convolutional Neural Network |
| CoT | Chain of Thought |
| DL | Deep Learning |
| DPO | Direct Preference Optimization |
| EOS | End of Sequence |
| FLOP | Floating Point Operation |
| GAN | Generative Adversarial Network |
| GELU | Gaussian Error Linear Unit |
| GNN | Graph Neural Network |
| GPT | Generative Pre-trained Transformer |
| GPU | Graphics Processing Unit |
| ICL | In-Context Learning |
| KL | Kullback-Leibler (Divergence) |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| LSTM | Long Short-Term Memory |
| MAE | Mean Absolute Error |
| ML | Machine Learning |
| MLP | Multi-Layer Perceptron |
| MoE | Mixture of Experts |
| MSE | Mean Squared Error |
| NCD | Normalized Compression Distance |
| NER | Named Entity Recognition |
| NLG | Natural Language Generation |
| NLP | Natural Language Processing |
| NLU | Natural Language Understanding |
| OOD | Out of Distribution |
| PCA | Principal Component Analysis |
| PEFT | Parameter-Efficient Fine-Tuning |
| PPO | Proximal Policy Optimization |
| QLoRA | Quantized LoRA |
| RAG | Retrieval-Augmented Generation |
| ReLU | Rectified Linear Unit |
| RL | Reinforcement Learning |
| RLHF | Reinforcement Learning from Human Feedback |
| RNN | Recurrent Neural Network |
| ROC | Receiver Operating Characteristic |
| RoPE | Rotary Position Embedding |
| SGD | Stochastic Gradient Descent |
| SVM | Support Vector Machine |
| TPU | Tensor Processing Unit |
| VAE | Variational Autoencoder |
| ViT | Vision Transformer |

---

*Total terms: 537*

*"A glossary is not just a list of definitions—it's a map of the territory."*
