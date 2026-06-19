# Lattice-Probe Scientific Improvement Review (Post-Fix Pass)

---

# Section 1 — Scientific Bottlenecks

1. **Time-Domain Learning Penalty**
```markdown
Problem: The models are forced to learn negacyclic convolution ($X^n+1$) from scratch in the time domain, which is statistically and parametrically inefficient.
Expected impact: Drastic reduction in required sample complexity.
Difficulty: Medium
Priority: 1
```

2. **Absence of Phase Transition Analysis**
```markdown
Problem: Testing only at standard, -35%, and -60% noise levels provides isolated data points rather than identifying the exact threshold (phase transition) where ML distinguishers succeed.
Expected impact: Defines the precise security margin.
Difficulty: Low
Priority: 2
```

3. **Disconnection from Cryptographic Metrics**
```markdown
Problem: Cryptographers evaluate distinguishability via Advantage, not AUROC. Without this translation, the claims lack standard cryptographic grounding.
Expected impact: Higher credibility with reviewers.
Difficulty: Trivial
Priority: 3
```

4. **Secret Overfitting Ambiguity**
```markdown
Problem: Testing primarily on datasets with fixed secrets leaves it ambiguous whether the model learns the generalized Module-LWE distribution or simply overfits to $s$.
Expected impact: Proves fundamental algorithmic generalization.
Difficulty: Low
Priority: 4
```

---

# Section 2 — Representation Improvements

## Transformer

The current flattened sequence is algebraically naive. To capture the structure of $\mathbb{Z}_q[X]/(X^n+1)$, the model must understand cyclic shifts and modular arithmetic boundaries.

* **RoPE (Rotary Positional Embeddings)**: Highly recommended. RoPE explicitly encodes relative positions as rotations in a complex plane. This mathematically mimics multiplication by $X$ in the polynomial ring, natively embedding the cyclic structure of the ring into the attention mechanism.
* **Hierarchical / Module-Aware Embeddings**: Add a 2D embedding space: one vector for the module index ($0 \dots k$) and one for the polynomial degree ($0 \dots 255$). 
* **NTT-Domain Representations**: Since polynomial multiplication is point-wise in the NTT domain, transforming $a$ and $b$ via NTT before feeding them to the Transformer reduces the sequence to 128 complex-like pairs, radically simplifying the learning task.

**Expected Gain**: Transitioning to RoPE + NTT representations is expected to lower the required sample complexity for distinguishing by at least $O(n)$, potentially shifting the attack feasibility curve.

## GNN

The current bipartite coefficient graph is structurally mismatched for time-domain convolutions.

1. **NTT Butterfly Graphs**: (Rank 1). Explicitly wire the graph to mirror the Cooley-Tukey NTT butterfly algorithm. This enforces the exact algebraic pathways of the negacyclic transform, allowing the GNN to perform spectral operations.
2. **Coefficient Interaction Graphs**: (Rank 2). A dense bipartite graph where $a_i$ connects to $b_k$ with edge attributes reflecting $(k-i) \pmod n$. This models the convolution matrix but is extremely dense ($O(n^2)$ edges).
3. **Factor Graphs**: (Rank 3). Representing equations as factor nodes and variables as variable nodes, specifically tailored to the modular arithmetic constraints.

---

# Section 3 — Architecture Search

## Sequence Models

### RoFormer (Transformer with RoPE)
```markdown
Expected benefit: Natively captures polynomial shifts.
Expected risk: None.
Compute cost: Equal to standard Transformer.
Scientific value: Very High (Demonstrates algebraic alignment).
```

### State-Space Models (Mamba)
```markdown
Expected benefit: Scales linearly $O(N)$, allowing expansion to $k=4, 6, 8$ without quadratic attention explosion.
Expected risk: Hidden state may bottleneck modular arithmetic tracking.
Compute cost: Lower than Transformer for large $k$.
Scientific value: High (Explores sub-quadratic architectures for cryptanalysis).
```

## Graph Models

### TransformerConv / GATv2
```markdown
Expected benefit: Dynamic attention over polynomial coefficients, isolating specific high-leakage interactions.
Expected risk: Overfitting to noise patterns.
Compute cost: High (Multi-head attention over dense graphs).
Scientific value: Medium.
```

### GIN (Graph Isomorphism Network)
```markdown
Expected benefit: Maximum expressive power for distinguishing graph structures (Weisfeiler-Lehman test equivalent).
Expected risk: LWE indistinguishability may theoretically bound GIN performance anyway.
Compute cost: Medium.
Scientific value: High (Provides theoretical upper bounds on GNN capability).
```

---

# Section 4 — Missing Experiments

## Noise Sweep
**Should it be added?** Absolutely. Cryptanalysis is about finding the breaking point. Testing noise reductions in 5% decrements (100% to 60%) will yield a phase-transition curve. This proves exactly how much margin NIST left on the table.

## Secret Rotation Study
Current conclusions heavily depend on secret reuse (which simulates a single-key attack). 
Comparing fixed vs. rotating vs. fresh secrets is scientifically mandatory. It answers the critical question: *Is the neural network learning a generalized distinguishing algorithm for the LWE distribution, or is it solving a specific instance of the Hidden Subgroup Problem?*

## Sample Complexity Scaling
Testing $2^{10}$ through $2^{24}$ is crucial. Cryptographers need to know the asymptotic scaling. If the distinguishing advantage plateaus at $2^{20}$, ML is fundamentally capped. If the advantage scales linearly with $\log(N)$, then the standard is theoretically threatened by massive data.

## Cross-Parameter Transfer
Train on $k=2$, evaluate on $k=3, 4$.
*Expected outcome*: Zero-shot transfer will likely fail due to dimensionality mismatches, but testing whether a model trained on ML-KEM-512 fine-tunes onto ML-KEM-768 with $10\times$ fewer samples proves that the ML model learns general lattice properties, not just parameter-specific noise artifacts.

---

# Section 5 — Statistical Methodology

Relying purely on AUROC is a red flag for cryptographic peer review.

**Advantage = 2 * AUROC - 1**
This is mandatory. In cryptography, an adversary's success is measured by their advantage over random guessing.

**Ranked Additional Metrics**:
1. **Permutation Tests**: (Highest usefulness). Randomly shuffle the labels of the test set and re-evaluate. This statistically proves the model is exploiting the algebraic structure of LWE, not side-channel artifacts in the PRNG or floating-point conversions.
2. **Confidence Intervals (BCa Bootstrap)**: (High). Bias-Corrected and Accelerated bootstraps provide much stronger statistical bounds than percentile bootstraps when dealing with edge-of-margin cryptography.
3. **Calibration Curves**: (Medium). Shows if the model's logits represent true probabilities of LWE vs. Uniform.
4. **Mutual Information Estimates**: (Low). Computationally difficult for neural networks and adds little beyond empirical Advantage.

---

# Section 6 — Cryptographic Research Extensions

## NTT-Aware Models
Yes. Modifying the input pipeline to feed the models data entirely in the NTT domain (where polynomial convolution becomes element-wise multiplication) fundamentally changes the ML task from learning convolutions to learning modular statistics. This is the most promising extension.

## Hybrid ML + Classical Attacks
Instead of end-to-end distinguishing, ML could be used as a heuristic oracle within the General Sieve Kernel (G6K). ML models are exceptionally good at predicting structural sparsity; training a model to predict which vectors in a lattice basis will yield the shortest vectors during BKZ reduction could dramatically speed up classical lattice reduction attacks.

## Hardness Prediction
Training a model to predict the required BKZ block size ($\beta$) for an arbitrary instance (or estimating the specific security margin) is highly valuable. This turns the ML model into an automated cryptanalysis estimation tool for parameter designers.

---

# Section 7 — Publication Readiness

## Must Have (Required before submission)
* Distinguishing Advantage reporting ($2 \times \text{AUROC} - 1$).
* Continuous Noise Sweep (Phase transition analysis).
* Sample Complexity Scaling bounds.
* Fixed vs. Rotating Secret Ablation.

## Strongly Recommended
* Upgrade Transformer with RoPE.
* Execute permutation tests to prove artifact immunity.
* Convert inputs to the NTT domain before ML ingestion.

## Nice To Have
* Evaluation of State-Space Models (Mamba) for linear scaling.
* Cross-parameter transfer learning experiments.
* Application to Hybrid ML+Sieving architectures.

---

# Section 8 — Top 20 Improvements

1. **Advantage Metric Reporting**
   * Scientific impact: Very High | Novelty: Low | Implementation difficulty: Trivial | Estimated effort: 10 mins | Publication value: Critical
2. **Phase Transition Noise Sweep**
   * Scientific impact: Very High | Novelty: Medium | Implementation difficulty: Low | Estimated effort: 1 day | Publication value: Critical
3. **Sample Complexity Scaling Analysis**
   * Scientific impact: Very High | Novelty: Low | Implementation difficulty: Low | Estimated effort: 1 day | Publication value: Critical
4. **Secret Rotation Ablation**
   * Scientific impact: High | Novelty: Medium | Implementation difficulty: Low | Estimated effort: 1 day | Publication value: Critical
5. **NTT-Domain Model Ingestion**
   * Scientific impact: Very High | Novelty: High | Implementation difficulty: Medium | Estimated effort: 2 days | Publication value: Very High
6. **RoPE Transformer Architecture**
   * Scientific impact: High | Novelty: High | Implementation difficulty: Medium | Estimated effort: 1 day | Publication value: Very High
7. **Permutation Tests for Artifact Denial**
   * Scientific impact: High | Novelty: Low | Implementation difficulty: Low | Estimated effort: 2 hours | Publication value: High
8. **BCa Bootstrap Confidence Intervals**
   * Scientific impact: Medium | Novelty: Low | Implementation difficulty: Low | Estimated effort: 1 hour | Publication value: High
9. **State-Space Model (Mamba) Baseline**
   * Scientific impact: High | Novelty: Very High | Implementation difficulty: High | Estimated effort: 3 days | Publication value: High
10. **GIN (Graph Isomorphism Network) Evaluation**
    * Scientific impact: Medium | Novelty: Medium | Implementation difficulty: Low | Estimated effort: 1 day | Publication value: High
11. **NTT Butterfly Graph Topology**
    * Scientific impact: High | Novelty: High | Implementation difficulty: Hard | Estimated effort: 4 days | Publication value: High
12. **Hierarchical 2D Positional Embeddings**
    * Scientific impact: Medium | Novelty: Medium | Implementation difficulty: Low | Estimated effort: 3 hours | Publication value: Medium
13. **Cross-Parameter Fine-Tuning**
    * Scientific impact: Medium | Novelty: High | Implementation difficulty: Medium | Estimated effort: 2 days | Publication value: Medium
14. **Calibration Curve Analysis**
    * Scientific impact: Medium | Novelty: Low | Implementation difficulty: Low | Estimated effort: 2 hours | Publication value: Medium
15. **Polynomial Interaction Graphs**
    * Scientific impact: Medium | Novelty: Medium | Implementation difficulty: Medium | Estimated effort: 2 days | Publication value: Medium
16. **Train/Test on Different PRNG Suites**
    * Scientific impact: Medium | Novelty: Low | Implementation difficulty: Low | Estimated effort: 2 hours | Publication value: Medium
17. **Hybrid ML-Sieving Discussion Section**
    * Scientific impact: Low | Novelty: High | Implementation difficulty: Trivial | Estimated effort: 2 hours | Publication value: Medium
18. **GATv2 / TransformerConv Ablation**
    * Scientific impact: Low | Novelty: Low | Implementation difficulty: Low | Estimated effort: 1 day | Publication value: Low
19. **Hardness Prediction Formulation**
    * Scientific impact: Low | Novelty: High | Implementation difficulty: Hard | Estimated effort: 5 days | Publication value: Low
20. **Mutual Information Estimation**
    * Scientific impact: Low | Novelty: Medium | Implementation difficulty: Medium | Estimated effort: 2 days | Publication value: Low

---

# Final Requirement

### Top 5 Changes Most Likely To Produce New Scientific Results

1. **NTT-Domain Model Ingestion**: Radically changes the representation from complex convolutions to point-wise arithmetic. Highly likely to drastically reduce sample complexity and break through the current AUROC ceilings on standardized parameters.
2. **Phase Transition Noise Sweep**: Mapping the exact boundary where ML succeeds vs. fails creates a concrete, citable formula for empirical ML cryptanalysis bounds against LWE.
3. **State-Space Models (Mamba)**: Applying sub-quadratic sequence models to cryptography allows analyzing $k=8$ or $k=10$ instances (impossible for standard Transformers due to $O(N^2)$ memory), pushing boundaries on massive parameter sets.
4. **Cross-Parameter Transfer Learning**: Successfully showing an ML model can generalize from ML-KEM-512 to ML-KEM-768 would be a massive breakthrough, proving neural networks learn universal lattice properties rather than instance-specific noise.
5. **Secret Rotation Study**: Will definitively answer whether ML distinguishers are solving LWE generally, or merely acting as highly optimized single-key recovery heuristics.

### Top 5 Changes Most Likely To Impress Cryptography Reviewers

1. **Advantage Metric Reporting ($2 \times \text{AUROC} - 1$)**: Speaks the native language of cryptographers. It instantly establishes credibility and demonstrates domain awareness.
2. **Permutation Tests**: Cryptographers are highly skeptical of ML results due to side-channel and dataset artifacts. Statistically proving immunity to artifacts via permutation tests destroys the most common reviewer objection.
3. **Sample Complexity Scaling Limits**: Providing concrete asymptotic bounds (e.g., proving that $O(N)$ data yields sub-exponential advantage) aligns the ML research with classical lattice reduction theory.
4. **RoPE for Transformers**: Demonstrating that the neural architecture was explicitly tailored to mimic the $X^n+1$ cyclic shift of the polynomial ring proves the authors are bridging math and ML gracefully.
5. **BCa Bootstrap Intervals**: Utilizing rigorous, bias-corrected statistical intervals for $p$-values and CI bounds proves the results are statistically unassailable at the margins.
