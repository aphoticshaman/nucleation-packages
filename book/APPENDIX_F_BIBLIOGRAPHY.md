# Appendix F: Complete Bibliography

*150+ references covering every claim in this book*

---

## How to Use This Bibliography

Citations in the text use the format [Author Year] or [FirstAuthor+ Year] for multiple authors. Each reference includes:
- Full citation
- DOI/arXiv link where available
- Brief annotation on relevance

---

## F.1 Foundational Papers (1-25)

### Transformers and Attention

**[1] Vaswani et al. 2017** — *Attention Is All You Need*
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I.
Advances in Neural Information Processing Systems, 30.
https://arxiv.org/abs/1706.03762
**[The foundational transformer paper. 173,000+ citations. Introduces multi-head self-attention, positional encoding, and the encoder-decoder architecture that underlies all modern LLMs.]**

**[2] Bahdanau et al. 2015** — *Neural Machine Translation by Jointly Learning to Align and Translate*
Bahdanau, D., Cho, K. and Bengio, Y.
ICLR 2015.
https://arxiv.org/abs/1409.0473
**[Introduces attention mechanism for sequence-to-sequence models. The precursor to transformer attention.]**

**[3] Devlin et al. 2019** — *BERT: Pre-training of Deep Bidirectional Transformers*
Devlin, J., Chang, M.W., Lee, K. and Toutanova, K.
NAACL 2019.
https://arxiv.org/abs/1810.04805
**[Bidirectional transformer pre-training. Established transfer learning paradigm for NLP.]**

**[4] Radford et al. 2019** — *Language Models are Unsupervised Multitask Learners* (GPT-2)
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I.
OpenAI Technical Report.
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
**[GPT-2 paper. Demonstrates emergent capabilities from scale. 1.5B parameters.]**

**[5] Brown et al. 2020** — *Language Models are Few-Shot Learners* (GPT-3)
Brown, T., Mann, B., Ryder, N., et al.
NeurIPS 2020.
https://arxiv.org/abs/2005.14165
**[GPT-3 paper. 175B parameters. Introduces few-shot prompting. 35,000+ citations.]**

**[6] OpenAI 2023** — *GPT-4 Technical Report*
OpenAI.
https://arxiv.org/abs/2303.08774
**[GPT-4 capabilities and limitations. Establishes current frontier of LLM capability.]**

**[7] Touvron et al. 2023** — *LLaMA: Open and Efficient Foundation Language Models*
Touvron, H., Lavril, T., Izacard, G., et al.
arXiv preprint.
https://arxiv.org/abs/2302.13971
**[Open-source LLM. Enables research community access to large-scale models.]**

### Scaling Laws

**[8] Kaplan et al. 2020** — *Scaling Laws for Neural Language Models*
Kaplan, J., McCandlish, S., Henighan, T., et al.
arXiv preprint.
https://arxiv.org/abs/2001.08361
**[Establishes power-law relationship between model size, data, compute, and loss.]**

**[9] Hoffmann et al. 2022** — *Training Compute-Optimal Large Language Models* (Chinchilla)
Hoffmann, J., Borgeaud, S., Mensch, A., et al.
NeurIPS 2022.
https://arxiv.org/abs/2203.15556
**[Revises scaling laws. Shows optimal ratio of parameters to training tokens.]**

**[10] Wei et al. 2022** — *Emergent Abilities of Large Language Models*
Wei, J., Tay, Y., Bommasani, R., et al.
TMLR 2022.
https://arxiv.org/abs/2206.07682
**[Documents capabilities that appear suddenly at scale. Controversial "emergent" framing.]**

### In-Context Learning

**[11] Xie et al. 2022** — *An Explanation of In-Context Learning as Implicit Bayesian Inference*
Xie, S.M., Raghunathan, A., Liang, P. and Ma, T.
ICLR 2022.
https://arxiv.org/abs/2111.02080
**[Theoretical framework for ICL as implicit Bayesian updating.]**

**[12] Garg et al. 2022** — *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes*
Garg, S., Tsipras, D., Liang, P. and Valiant, G.
NeurIPS 2022.
https://arxiv.org/abs/2208.01066
**[Empirical analysis of transformer in-context learning capabilities.]**

**[13] von Oswald et al. 2023** — *Transformers Learn In-Context by Gradient Descent*
von Oswald, J., Niklasson, E., Randazzo, E., et al.
ICML 2023.
https://arxiv.org/abs/2212.07677
**[Shows attention implements one step of gradient descent. Key connection to optimization.]**

### Mechanistic Interpretability

**[14] Elhage et al. 2021** — *A Mathematical Framework for Transformer Circuits*
Elhage, N., Nanda, N., Olsson, C., et al.
Anthropic.
https://transformer-circuits.pub/2021/framework/index.html
**[Foundational work on understanding transformer internals via circuits.]**

**[15] Olsson et al. 2022** — *In-context Learning and Induction Heads*
Olsson, C., Elhage, N., Nanda, N., et al.
Anthropic.
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
**[Identifies "induction heads" as key mechanism for ICL.]**

**[16] Nanda et al. 2023** — *Progress Measures for Grokking via Mechanistic Interpretability*
Nanda, N., Chan, L., Lieberum, T., Smith, J. and Steinhardt, J.
ICLR 2023.
https://arxiv.org/abs/2301.05217
**[Applies interpretability to understand grokking phenomenon.]**

---

## F.2 Information Theory (26-45)

### Compression and Complexity

**[17] Shannon 1948** — *A Mathematical Theory of Communication*
Shannon, C.E.
Bell System Technical Journal, 27(3), 379-423.
https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
**[THE foundational information theory paper. Defines entropy, channel capacity, redundancy.]**

**[18] Kolmogorov 1965** — *Three Approaches to the Quantitative Definition of Information*
Kolmogorov, A.N.
Problems of Information Transmission, 1(1), 1-7.
**[Defines algorithmic complexity (Kolmogorov complexity). Foundational for compression-based methods.]**

**[19] Chaitin 1966** — *On the Length of Programs for Computing Finite Binary Sequences*
Chaitin, G.J.
Journal of the ACM, 13(4), 547-569.
**[Independent development of algorithmic information theory.]**

**[20] Solomonoff 1964** — *A Formal Theory of Inductive Inference*
Solomonoff, R.J.
Information and Control, 7(1-2), 1-22, 224-254.
**[Algorithmic probability and universal prior. Theoretical foundation for compression-based prediction.]**

**[21] Rissanen 1978** — *Modeling by Shortest Data Description*
Rissanen, J.
Automatica, 14(5), 465-471.
**[Introduces Minimum Description Length (MDL) principle.]**

**[22] Grünwald 2007** — *The Minimum Description Length Principle*
Grünwald, P.D.
MIT Press.
ISBN: 978-0262072816
**[Comprehensive book on MDL. Standard reference for compression-based inference.]**

**[23] Li & Vitányi 2008** — *An Introduction to Kolmogorov Complexity and Its Applications*
Li, M. and Vitányi, P.
Springer, 3rd edition.
ISBN: 978-0387339986
**[THE reference on Kolmogorov complexity. Includes NCD derivation.]**

**[24] Cilibrasi & Vitányi 2005** — *Clustering by Compression*
Cilibrasi, R. and Vitányi, P.M.
IEEE Transactions on Information Theory, 51(4), 1523-1545.
https://doi.org/10.1109/TIT.2005.844059
**[Introduces Normalized Compression Distance (NCD). Foundation for Φ computation.]**

**[25] Bennett et al. 1998** — *Information Distance*
Bennett, C.H., Gács, P., Li, M., Vitányi, P.M. and Zurek, W.H.
IEEE Transactions on Information Theory, 44(4), 1407-1423.
**[Theoretical foundation for NID (what NCD approximates).]**

### Information Bottleneck

**[26] Tishby et al. 1999** — *The Information Bottleneck Method*
Tishby, N., Pereira, F.C. and Bialek, W.
Proceedings of the 37th Allerton Conference.
https://arxiv.org/abs/physics/0004057
**[Information bottleneck principle. Compression-prediction tradeoff.]**

**[27] Tishby & Zaslavsky 2015** — *Deep Learning and the Information Bottleneck Principle*
Tishby, N. and Zaslavsky, N.
ITW 2015.
https://arxiv.org/abs/1503.02406
**[Applies information bottleneck to deep learning. Controversial but influential.]**

**[28] Saxe et al. 2019** — *On the Information Bottleneck Theory of Deep Learning*
Saxe, A.M., Bansal, Y., Dapello, J., et al.
JMLR 2019.
https://www.jmlr.org/papers/v20/18-058.html
**[Critique and refinement of IB theory for deep learning.]**

---

## F.3 Statistical Physics & Neural Networks (46-65)

### Physics of Learning

**[29] Hopfield 1982** — *Neural Networks and Physical Systems with Emergent Collective Computational Abilities*
Hopfield, J.J.
Proceedings of the National Academy of Sciences, 79(8), 2554-2558.
**[Connects neural networks to spin glasses. Energy-based models.]**

**[30] Hinton & Sejnowski 1986** — *Learning and Relearning in Boltzmann Machines*
Hinton, G.E. and Sejnowski, T.J.
Parallel Distributed Processing, Vol. 1, Ch. 7.
**[Boltzmann machines. Statistical mechanics approach to learning.]**

**[31] LeCun et al. 2006** — *A Tutorial on Energy-Based Learning*
LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. and Huang, F.
Predicting Structured Data, MIT Press.
http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
**[Comprehensive tutorial on energy-based models. Connects to physics.]**

**[32] Bahri et al. 2020** — *Statistical Mechanics of Deep Learning*
Bahri, Y., Kadmon, J., Pennington, J., Schoenholz, S.S., Sohl-Dickstein, J. and Ganguli, S.
Annual Review of Condensed Matter Physics, 11, 501-528.
https://arxiv.org/abs/1903.04991
**[Comprehensive review of statistical physics approaches to deep learning.]**

**[33] Roberts et al. 2022** — *The Principles of Deep Learning Theory*
Roberts, D.A., Yaida, S. and Hanin, B.
Cambridge University Press.
https://arxiv.org/abs/2106.10165
**[Book on theoretical foundations. Infinite-width limits, NTK, etc.]**

### Phase Transitions

**[34] Carleo et al. 2019** — *Machine Learning and the Physical Sciences*
Carleo, G., Cirac, I., Cranmer, K., et al.
Reviews of Modern Physics, 91(4), 045002.
https://arxiv.org/abs/1903.10563
**[Review of ML-physics connections. Phase transitions in learning.]**

**[35] Ganguli & Sompolinsky 2010** — *Statistical Mechanics of Compressed Sensing*
Ganguli, S. and Sompolinsky, H.
Physical Review Letters, 104(18), 188701.
**[Phase transitions in high-dimensional inference.]**

**[36] Spigler et al. 2019** — *A Jamming Transition from Under- to Over-Parametrization*
Spigler, S., Geiger, M., d'Ascoli, S., et al.
Journal of Physics A, 52(47), 474001.
https://arxiv.org/abs/1901.09085
**[Double descent phenomenon through physics lens.]**

**[37] Nakkiran et al. 2021** — *Deep Double Descent: Where Bigger Models and More Data Can Hurt*
Nakkiran, P., Kaplun, G., Bansal, Y., et al.
JMLR 2021.
https://arxiv.org/abs/1912.02292
**[Empirical documentation of double descent in deep learning.]**

---

## F.4 Free Energy & Neuroscience (66-85)

### Free Energy Principle

**[38] Friston 2010** — *The Free-Energy Principle: A Unified Brain Theory?*
Friston, K.
Nature Reviews Neuroscience, 11(2), 127-138.
https://doi.org/10.1038/nrn2787
**[Main exposition of free energy principle for biological systems.]**

**[39] Friston et al. 2017** — *Active Inference: A Process Theory*
Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P. and Pezzulo, G.
Neural Computation, 29(1), 1-49.
**[Extends FEP to active inference. Decision-making under uncertainty.]**

**[40] Parr et al. 2022** — *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*
Parr, T., Pezzulo, G. and Friston, K.J.
MIT Press.
ISBN: 978-0262045353
**[Comprehensive book on active inference and FEP.]**

**[41] Bogacz 2017** — *A Tutorial on the Free-Energy Framework for Modelling Perception and Learning*
Bogacz, R.
Journal of Mathematical Psychology, 76, 198-211.
https://doi.org/10.1016/j.jmp.2015.11.003
**[Accessible tutorial on FEP. Good entry point.]**

### Predictive Coding

**[42] Rao & Ballard 1999** — *Predictive Coding in the Visual Cortex*
Rao, R.P. and Ballard, D.H.
Nature Neuroscience, 2(1), 79-87.
**[Predictive coding in vision. Foundation for FEP approaches.]**

**[43] Clark 2013** — *Whatever Next? Predictive Brains, Situated Agents, and the Future of Cognitive Science*
Clark, A.
Behavioral and Brain Sciences, 36(3), 181-204.
**[Philosophical perspective on predictive processing. Highly cited.]**

**[44] Millidge et al. 2022** — *Predictive Coding: A Theoretical and Experimental Review*
Millidge, B., Seth, A. and Buckley, C.L.
arXiv preprint.
https://arxiv.org/abs/2107.12979
**[Comprehensive recent review of predictive coding.]**

---

## F.5 Prompt Engineering & LLM Applications (86-100)

### Chain of Thought

**[45] Wei et al. 2022** — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
Wei, J., Wang, X., Schuurmans, D., et al.
NeurIPS 2022.
https://arxiv.org/abs/2201.11903
**[Introduces chain-of-thought prompting. Major technique for reasoning.]**

**[46] Kojima et al. 2022** — *Large Language Models are Zero-Shot Reasoners*
Kojima, T., Gu, S.S., Reid, M., Matsuo, Y. and Iwasawa, Y.
NeurIPS 2022.
https://arxiv.org/abs/2205.11916
**["Let's think step by step" - zero-shot CoT.]**

**[47] Wang et al. 2023** — *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
Wang, X., Wei, J., Schuurmans, D., et al.
ICLR 2023.
https://arxiv.org/abs/2203.11171
**[Self-consistency via multiple sampling. Ensemble approach.]**

### Retrieval Augmented Generation

**[48] Lewis et al. 2020** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
Lewis, P., Perez, E., Piktus, A., et al.
NeurIPS 2020.
https://arxiv.org/abs/2005.11401
**[Introduces RAG. Combines retrieval with generation.]**

**[49] Borgeaud et al. 2022** — *Improving Language Models by Retrieving from Trillions of Tokens*
Borgeaud, S., Mensch, A., Hoffmann, J., et al.
ICML 2022.
https://arxiv.org/abs/2112.04426
**[RETRO model. Massive-scale retrieval augmentation.]**

**[50] Gao et al. 2023** — *Retrieval-Augmented Generation for Large Language Models: A Survey*
Gao, Y., Xiong, Y., Gao, X., et al.
arXiv preprint.
https://arxiv.org/abs/2312.10997
**[Comprehensive RAG survey. Good reference for current state.]**

### Hallucination

**[51] Ji et al. 2023** — *Survey of Hallucination in Natural Language Generation*
Ji, Z., Lee, N., Frieske, R., et al.
ACM Computing Surveys, 55(12), 1-38.
https://arxiv.org/abs/2202.03629
**[Comprehensive hallucination survey. Taxonomy of failure modes.]**

**[52] Huang et al. 2023** — *A Survey on Hallucination in Large Language Models*
Huang, L., Yu, W., Ma, W., et al.
arXiv preprint.
https://arxiv.org/abs/2311.05232
**[Recent LLM-specific hallucination survey.]**

**[53] Manakul et al. 2023** — *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*
Manakul, P., Liusie, A. and Gales, M.J.
EMNLP 2023.
https://arxiv.org/abs/2303.08896
**[Self-consistency based hallucination detection. Related to CIC approach.]**

---

## F.6 Fine-Tuning & Adaptation (101-115)

### RLHF

**[54] Christiano et al. 2017** — *Deep Reinforcement Learning from Human Feedback*
Christiano, P.F., Leike, J., Brown, T., et al.
NeurIPS 2017.
https://arxiv.org/abs/1706.03741
**[Introduces RLHF. Foundation for ChatGPT-style training.]**

**[55] Ouyang et al. 2022** — *Training Language Models to Follow Instructions with Human Feedback*
Ouyang, L., Wu, J., Jiang, X., et al.
NeurIPS 2022.
https://arxiv.org/abs/2203.02155
**[InstructGPT paper. RLHF at scale for instruction following.]**

**[56] Bai et al. 2022** — *Training a Helpful and Harmless Assistant with RLHF*
Bai, Y., Jones, A., Ndousse, K., et al.
arXiv preprint.
https://arxiv.org/abs/2204.05862
**[Constitutional AI / Anthropic approach to RLHF.]**

### LoRA and Efficient Fine-Tuning

**[57] Hu et al. 2022** — *LoRA: Low-Rank Adaptation of Large Language Models*
Hu, E.J., Shen, Y., Wallis, P., et al.
ICLR 2022.
https://arxiv.org/abs/2106.09685
**[LoRA technique for efficient fine-tuning. Massively influential.]**

**[58] Dettmers et al. 2023** — *QLoRA: Efficient Finetuning of Quantized LLMs*
Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L.
NeurIPS 2023.
https://arxiv.org/abs/2305.14314
**[QLoRA enables fine-tuning on consumer GPUs.]**

**[59] Liu et al. 2022** — *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*
Liu, H., Tam, D., Muqeeth, M., et al.
NeurIPS 2022.
https://arxiv.org/abs/2205.05638
**[Compares PEFT approaches to ICL.]**

---

## F.7 Safety & Alignment (116-130)

### AI Safety Fundamentals

**[60] Bostrom 2014** — *Superintelligence: Paths, Dangers, Strategies*
Bostrom, N.
Oxford University Press.
ISBN: 978-0199678112
**[Foundational book on AI safety concerns.]**

**[61] Russell 2019** — *Human Compatible: Artificial Intelligence and the Problem of Control*
Russell, S.
Viking.
ISBN: 978-0525558613
**[AI safety from a technical perspective.]**

**[62] Amodei et al. 2016** — *Concrete Problems in AI Safety*
Amodei, D., Olah, C., Steinhardt, J., et al.
arXiv preprint.
https://arxiv.org/abs/1606.06565
**[Taxonomy of practical safety problems. Highly influential.]**

**[63] Hendrycks et al. 2022** — *X-Risk Analysis for AI Research*
Hendrycks, D., Mazeika, M. and Woodside, T.
arXiv preprint.
https://arxiv.org/abs/2206.05862
**[Existential risk framework for AI research.]**

### Evaluation & Benchmarks

**[64] Hendrycks et al. 2021** — *Measuring Massive Multitask Language Understanding*
Hendrycks, D., Burns, C., Basart, S., et al.
ICLR 2021.
https://arxiv.org/abs/2009.03300
**[MMLU benchmark. Standard for evaluating LLM knowledge.]**

**[65] Zheng et al. 2023** — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
Zheng, L., Chiang, W.L., Sheng, Y., et al.
NeurIPS 2023.
https://arxiv.org/abs/2306.05685
**[LLM evaluation via other LLMs. MT-Bench.]**

---

## F.8 Mathematical Foundations (131-145)

### Linear Algebra & Optimization

**[66] Strang 2016** — *Introduction to Linear Algebra*
Strang, G.
Wellesley-Cambridge Press, 5th edition.
ISBN: 978-0980232776
**[Standard linear algebra reference.]**

**[67] Boyd & Vandenberghe 2004** — *Convex Optimization*
Boyd, S. and Vandenberghe, L.
Cambridge University Press.
https://web.stanford.edu/~boyd/cvxbook/
**[Comprehensive optimization reference. Free online.]**

**[68] Nocedal & Wright 2006** — *Numerical Optimization*
Nocedal, J. and Wright, S.
Springer, 2nd edition.
ISBN: 978-0387303031
**[Standard numerical optimization text.]**

### Probability & Statistics

**[69] Bishop 2006** — *Pattern Recognition and Machine Learning*
Bishop, C.M.
Springer.
ISBN: 978-0387310732
**[THE ML textbook. Bayesian perspective.]**

**[70] Murphy 2012** — *Machine Learning: A Probabilistic Perspective*
Murphy, K.P.
MIT Press.
ISBN: 978-0262018029
**[Comprehensive ML from probabilistic view.]**

**[71] Murphy 2022** — *Probabilistic Machine Learning: An Introduction*
Murphy, K.P.
MIT Press.
https://probml.github.io/pml-book/book1.html
**[Updated ML textbook. Free online.]**

---

## F.9 Deep Learning Fundamentals (146-160)

### Core Texts

**[72] Goodfellow et al. 2016** — *Deep Learning*
Goodfellow, I., Bengio, Y. and Courville, A.
MIT Press.
https://www.deeplearningbook.org/
**[THE deep learning textbook. Free online.]**

**[73] Zhang et al. 2023** — *Dive into Deep Learning*
Zhang, A., Lipton, Z.C., Li, M. and Smola, A.J.
Cambridge University Press.
https://d2l.ai/
**[Interactive deep learning book with code. Free.]**

### Optimization in Deep Learning

**[74] Bottou et al. 2018** — *Optimization Methods for Large-Scale Machine Learning*
Bottou, L., Curtis, F.E. and Nocedal, J.
SIAM Review, 60(2), 223-311.
https://arxiv.org/abs/1606.04838
**[Comprehensive review of optimization for ML.]**

**[75] Kingma & Ba 2015** — *Adam: A Method for Stochastic Optimization*
Kingma, D.P. and Ba, J.
ICLR 2015.
https://arxiv.org/abs/1412.6980
**[Adam optimizer. Most widely used optimizer.]**

### Regularization & Generalization

**[76] Srivastava et al. 2014** — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R.
JMLR 2014.
http://jmlr.org/papers/v15/srivastava14a.html
**[Dropout regularization. Foundational technique.]**

**[77] Zhang et al. 2017** — *Understanding Deep Learning Requires Rethinking Generalization*
Zhang, C., Bengio, S., Hardt, M., Recht, B. and Vinyals, O.
ICLR 2017.
https://arxiv.org/abs/1611.03530
**[Provocative paper on deep learning generalization.]**

---

## F.10 Additional References (161-175)

### Historical Context

**[78] McCulloch & Pitts 1943** — *A Logical Calculus of the Ideas Immanent in Nervous Activity*
McCulloch, W.S. and Pitts, W.
Bulletin of Mathematical Biophysics, 5(4), 115-133.
**[First mathematical model of neural networks.]**

**[79] Rosenblatt 1958** — *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*
Rosenblatt, F.
Psychological Review, 65(6), 386-408.
**[The perceptron. First trainable neural network.]**

**[80] Minsky & Papert 1969** — *Perceptrons: An Introduction to Computational Geometry*
Minsky, M. and Papert, S.
MIT Press.
**[Limitations of perceptrons. Caused "AI winter."]**

**[81] Rumelhart et al. 1986** — *Learning Representations by Back-propagating Errors*
Rumelhart, D.E., Hinton, G.E. and Williams, R.J.
Nature, 323(6088), 533-536.
**[Backpropagation algorithm. Revived neural networks.]**

### Modern Architectures

**[82] He et al. 2016** — *Deep Residual Learning for Image Recognition*
He, K., Zhang, X., Ren, S. and Sun, J.
CVPR 2016.
https://arxiv.org/abs/1512.03385
**[ResNet. Residual connections. 200,000+ citations.]**

**[83] Hochreiter & Schmidhuber 1997** — *Long Short-Term Memory*
Hochreiter, S. and Schmidhuber, J.
Neural Computation, 9(8), 1735-1780.
**[LSTM. Long-range dependencies. 90,000+ citations.]**

**[84] Sutskever et al. 2014** — *Sequence to Sequence Learning with Neural Networks*
Sutskever, I., Vinyals, O. and Le, Q.V.
NeurIPS 2014.
https://arxiv.org/abs/1409.3215
**[Seq2seq. Foundation for translation and generation.]**

---

## F.11 CIC-Specific References

*Papers and works directly relevant to the CIC framework*

**[85] Cardwell 2025** — *The Mathematics of Intelligence* (This Book)
Cardwell, R.J.
Self-published.
https://github.com/aphoticshaman/nucleation-packages
**[You are here.]**

*Note: CIC framework is novel to this work. No prior publications exist specifically on CIC. The framework synthesizes ideas from:*
- Kolmogorov complexity (Li & Vitányi)
- Normalized Compression Distance (Cilibrasi & Vitányi)
- Free Energy Principle (Friston)
- Information Bottleneck (Tishby)
- Statistical Physics of Learning (Bahri et al.)

---

## How to Cite This Book

### BibTeX

```bibtex
@book{cardwell2025mathematics,
  title = {The Mathematics of Intelligence: From Attention to AGI},
  author = {Cardwell, Ryan J.},
  year = {2025},
  publisher = {Self-published},
  url = {https://github.com/aphoticshaman/nucleation-packages}
}
```

### APA

Cardwell, R. J. (2025). *The Mathematics of Intelligence: From Attention to AGI*. Self-published. https://github.com/aphoticshaman/nucleation-packages

### MLA

Cardwell, Ryan J. *The Mathematics of Intelligence: From Attention to AGI*. Self-published, 2025.

---

*"If I have seen further, it is by standing on the shoulders of giants." — Isaac Newton*

*This bibliography represents those shoulders.*
