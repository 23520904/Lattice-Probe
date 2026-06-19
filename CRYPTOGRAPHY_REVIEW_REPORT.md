# Lattice-Probe: Cryptography Reviewer Improvement Pass

---

# Section 1: Reviewer Attack Surface

```markdown
Reviewer Objection: The neural network is not learning the Module-LWE distribution; it is exploiting pseudorandom number generator (PRNG) artifacts or dataset serialization flaws to distinguish the splits.
Why It Matters: If true, the entire paper measures PRNG weakness or software bugs, not cryptographic hardness.
Severity: CRITICAL
How To Address It: Execute permutation tests. Randomly shuffle the labels of the dataset, retrain, and show that AUROC drops exactly to 0.50. Additionally, test a model trained to distinguish Uniform-ChaCha20 vs Uniform-TrueRandom.
```

```markdown
Reviewer Objection: Testing on a dataset derived from a single fixed secret simulates a highly specific Hidden Subgroup Problem (HSP) instance, not the general LWE distinguishing problem.
Why It Matters: Proves the model can recover one key given excessive samples, but fails to prove ML can generalize to break the cryptographic scheme intrinsically.
Severity: CRITICAL
How To Address It: Implement a zero-shot cross-secret transfer experiment. Train on datasets generated with Secret A, test on datasets generated with Secret B.
```

```markdown
Reviewer Objection: The paper uses AUROC rather than Cryptographic Advantage. An AUROC of 0.52 sounds high to ML researchers but represents a marginal distinguishing advantage of only 0.04.
Why It Matters: Cryptographers cannot directly compare AUROC to classical lattice reduction costs.
Severity: HIGH
How To Address It: Switch primary reporting metrics to Distinguishing Advantage ($2 \times \text{AUROC} - 1$) and provide rigorous BCa (Bias-Corrected and Accelerated) confidence intervals.
```

```markdown
Reviewer Objection: The models fail to break standardized parameters (ML-KEM-512), and the attacks on weakened parameters do not provide asymptotic scaling bounds.
Why It Matters: Without an asymptotic curve relating sample complexity $N$ to required parameter reduction, the paper is just an empirical observation, not a predictive cryptanalytic tool.
Severity: MEDIUM
How To Address It: Conduct a full sample complexity scaling study ($2^{10}$ through $2^{24}$) and plot the logarithmic growth of Advantage.
```

---

# Section 2: Hidden Assumptions

```markdown
Assumption: The fixed-secret experimental design accurately models an attacker's real-world capabilities.
Risk: The neural network might strictly overfit to the algebraic fingerprint of $s$, rendering the model useless against any other public key.
Likelihood: Very High
Impact: Fatal to broad cryptographic claims.
Recommended Experiment: Train on rotating secrets or evaluate cross-secret transferability.
```

```markdown
Assumption: Downcasting polynomial coefficients to `int16` or `float32` introduces no distinguishable statistical artifacts.
Risk: Truncation or floating-point representation boundaries might slightly alter the uniform distribution compared to the LWE distribution's noise convolution.
Likelihood: Low (since $3329 < 2^{15}$)
Impact: Fatal
Recommended Experiment: Perform KS (Kolmogorov-Smirnov) tests on the exact bits of the generated uniform vs. LWE datasets prior to ML ingestion.
```

```markdown
Assumption: Time-domain polynomial coefficients are the optimal representation for ML.
Risk: Negacyclic convolution $X^n+1$ requires immense model capacity to learn. The model might fail on ML-KEM-512 not because it's secure against ML, but because the model capacity was exhausted learning basic ring arithmetic.
Likelihood: High
Impact: False negative on ML attack capabilities.
Recommended Experiment: Transform inputs to the NTT domain before training to eliminate the burden of learning convolutions.
```

---

# Section 3: Strongest Missing Experiments

```markdown
Hypothesis: ML distinguishers learn the general LWE distribution, not instance-specific keys.
Method: Train a Transformer on $10^6$ samples generated using Secret A. Evaluate the fixed model on $10^5$ samples generated using Secret B.
Expected Outcome: Distinguishing advantage drops significantly but remains non-zero.
What Would Be Learned: Whether ML attacks are fundamentally instance-bound.
Publication Value: Enormous. A positive result would be a major breakthrough in lattice cryptanalysis.
```

```markdown
Hypothesis: The observed distinguishing advantage is purely due to LWE structure, not dataset artifacts.
Method: Permutation Testing. Randomly permute the 0/1 labels of the test set, or train on a dataset with randomized labels.
Expected Outcome: Advantage strictly equals 0.00 with tight confidence bounds.
What Would Be Learned: Definitively rules out PRNG, serialization, and leakage bugs.
Publication Value: Mandatory for CRYPTO/EUROCRYPT acceptance.
```

```markdown
Hypothesis: The required sample complexity to maintain a constant Advantage scales exponentially as Gaussian noise $\sigma$ approaches standard bounds.
Method: Train models at $\sigma$ reductions of 40%, 30%, 20%, 10%. Plot required $N$ to achieve Advantage = 0.10.
Expected Outcome: Log-linear or exponential scaling curves.
What Would Be Learned: Predictive bounds on when/if ML could break full ML-KEM with unlimited data.
Publication Value: Transforms the paper from empirical trivia into a rigorous theoretical framework.
```

```markdown
Hypothesis: Operating in the NTT domain drastically reduces required sample complexity.
Method: Pre-process the $(a, b)$ pairs via FIPS-203 NTT. Train the identical Transformer architecture on the complex-like residue pairs.
Expected Outcome: Models achieve higher Advantage with $10\times$ fewer samples.
What Would Be Learned: ML models are heavily bottlenecked by learning negacyclic convolutions in the time domain.
Publication Value: Very High.
```

```markdown
Hypothesis: The ML model achieves higher Advantage than classical statistical tests (e.g., $\chi^2$) on identical datasets.
Method: Plot the Advantage of the Transformer against the Advantage of the $\chi^2$ distinguisher across a fine-grained noise sweep.
Expected Outcome: ML curve diverges and outperforms classical stats at edge-of-margin noise levels.
What Would Be Learned: Proves ML captures high-dimensional hidden correlations that linear statistics miss.
Publication Value: High. Justifies the use of ML in the first place.
```

---

# Section 4: Distinguishing vs Learning Artifacts

Cryptographers will default to assuming the model cheated. You must conclusively rule these out:

* **RNG artifacts**: ChaCha20 is cryptographically secure, but implementations can leak. **Experiment**: Train the model to distinguish `generate_uniform_sample` outputs from `/dev/urandom`. If it succeeds, the paper is invalid.
* **Serialization artifacts**: Compression algorithms (`.npz`) or casting (`int16`) might behave differently on true uniform data vs. LWE uniform data. **Experiment**: Permutation tests on labels.
* **Secret reuse artifacts**: Fixed secrets create static $b = a \cdot s$ mappings. **Experiment**: Train on entirely fresh secrets per sample and measure Advantage degradation.

---

# Section 5: Statistical Rigor

Current evidence would likely be **rejected** by CRYPTO/EUROCRYPT due to reliance on AUROC and lack of adversarial advantage framing.

**Recommended Upgrades (Ranked by Reviewer Importance)**:
1. **Advantage Reporting**: Must explicitly report `Advantage = 2 * AUROC - 1`. Cryptography deals in advantage bounds ($\epsilon$).
2. **Significance Testing (Permutation)**: Essential to prove the $\epsilon > 0$ result is statistically significant and not an artifact.
3. **Confidence Intervals**: Replace percentile bootstraps with BCa (Bias-Corrected and Accelerated) bootstraps for tight $p$-value equivalents on the Advantage.
4. **Effect Size Reporting**: Cohen's $d$ or Mutual Information estimates comparing the LWE vs. Uniform logits distributions.

---

# Section 6: Representation Critique

## Coefficient-Domain (Time-Domain) Inputs
```markdown
Benefits: Raw, unadulterated data. Zero preprocessing assumptions.
Drawbacks: Forces the neural network to internally reverse-engineer the $X^n+1$ negacyclic convolution algorithm.
Potential Reviewer Criticism: "You failed to break ML-KEM-512 because your model wasted its parameter budget learning arithmetic rather than cryptography."
Evidence Needed: Proof that the model has the capacity to perform cyclic convolutions (e.g., via RoPE embeddings).
```

## NTT-Domain Inputs
```markdown
Benefits: Polynomial multiplication reduces to element-wise operations. Drastically simplifies the algebraic attack surface.
Drawbacks: Outputs are 128 pairs of complex-like residues; models require architectural adjustments to handle dual-residue structures.
Potential Reviewer Criticism: "Standard lattice reduction attacks do not rely on NTT. You are making assumptions about the representation." (Easily countered, as NTT is a known bijection).
Evidence Needed: A direct head-to-head performance comparison of Time-Domain vs NTT-Domain on the exact same dataset.
```

---

# Section 7: Architecture Critique

**Better Experiments $\gg$ Better Architecture.**

A skeptical cryptographer does not care if you used a Performer, Mamba, or GraphSAGE. They view neural networks as black-box heuristics. 
Endless architecture churn (swapping GAT for TransformerConv) contributes **zero** to scientific understanding unless it fundamentally alters the asymptotic sample complexity bounds. 

Focusing on experiments (Secret Rotation, Noise Sweeps, Sample Complexity) proves *cryptographic properties*. Focusing on architecture merely proves *engineering optimization*. The only architectural changes worth making are those that explicitly mirror the underlying math (e.g., RoPE for cyclic shifts, or NTT processing).

---

# Section 8: Publication Roadmap

If preparing for submission to TCHES tomorrow:

## Must Fix (Could cause rejection)
1. Convert all AUROC metrics to Distinguishing Advantage.
2. Execute and append Permutation Tests to definitively rule out PRNG/leakage artifacts.
3. Implement the Secret Rotation (Cross-Secret Transfer) ablation.

## Should Fix (Would substantially strengthen)
1. Add a continuous Noise Sweep (e.g., 90% to 50% in 5% increments) to plot the exact phase transition.
2. Add Sample Complexity scaling analysis.

## Optional
1. NTT-domain architecture variant.
2. Upgrading to RoPE.

---

# Section 9: Highest ROI Improvements

1. ```markdown
   Scientific Impact: High
   Reviewer Impact: Critical
   Implementation Cost: Low (2 hours)
   Risk: Low
   Priority: 1 (Permutation Tests)
   ```
2. ```markdown
   Scientific Impact: High
   Reviewer Impact: Critical
   Implementation Cost: Low (10 mins)
   Risk: Low
   Priority: 2 (Advantage Metric Translation)
   ```
3. ```markdown
   Scientific Impact: Very High
   Reviewer Impact: High
   Implementation Cost: Medium (1 day)
   Risk: Medium (Results might show poor generalization)
   Priority: 3 (Cross-Secret Transfer Learning)
   ```
4. ```markdown
   Scientific Impact: Very High
   Reviewer Impact: High
   Implementation Cost: Low (1 day compute)
   Risk: Low
   Priority: 4 (Phase Transition Noise Sweep)
   ```
5. ```markdown
   Scientific Impact: High
   Reviewer Impact: Medium
   Implementation Cost: Low (1 day compute)
   Risk: Low
   Priority: 5 (Sample Complexity Scaling Curve)
   ```
6. ```markdown
   Scientific Impact: High
   Reviewer Impact: Medium
   Implementation Cost: High (3 days)
   Risk: High (Requires new models)
   Priority: 6 (NTT-Domain Representation Pipeline)
   ```
7. ```markdown
   Scientific Impact: Medium
   Reviewer Impact: High
   Implementation Cost: Low (1 hour)
   Risk: Low
   Priority: 7 (BCa Bootstrap Intervals)
   ```
8. ```markdown
   Scientific Impact: Low
   Reviewer Impact: High
   Implementation Cost: Low (2 hours)
   Risk: Low
   Priority: 8 (PRNG Artifact Testing / ChaCha20 vs URANDOM)
   ```
9. ```markdown
   Scientific Impact: Medium
   Reviewer Impact: Low
   Implementation Cost: Medium (2 days)
   Risk: Medium
   Priority: 9 (RoPE Transformer Upgrade)
   ```
10. ```markdown
    Scientific Impact: Medium
    Reviewer Impact: Low
    Implementation Cost: High (4 days)
    Risk: High
    Priority: 10 (State-Space / Mamba Evaluation)
    ```

---

# Final Deliverable

## Top 5 Changes Most Likely To Change The Conclusions

1. **Cross-Secret Transfer Evaluation**: If models trained on Secret A fail completely on Secret B, the conclusion changes from "ML breaks LWE" to "ML solves a specific Hidden Subgroup instance given excessive data."
2. **NTT-Domain Processing**: Operating in the NTT domain may drastically lower the sample complexity threshold, potentially moving the needle on breaking closer-to-standard parameter margins.
3. **Phase Transition Noise Sweeps**: Replaces isolated data points with a definitive curve, establishing an explicit mathematical boundary for when LWE succumbs to ML heuristics.
4. **Sample Complexity Scaling Studies**: If the required data volume scales exponentially, the conclusion changes to validate NIST's parameters mathematically against ML attacks.
5. **Permutation Testing**: If permutation tests fail (Advantage $> 0$), the entire paper's conclusions are invalidated as an artifact of data leakage.

## Top 5 Changes Most Likely To Convince A Skeptical Cryptographer

1. **Reporting Distinguishing Advantage**: Converting AUROC to $2 \times \text{AUROC} - 1$ speaks the language of the reviewer. Cryptographers ignore AUROC; they demand to see $\epsilon$ advantage bounds.
2. **Permutation / Artifact Tests**: A skeptical reviewer assumes the ML model cheated (e.g., learned the PRNG state or float truncation). Rigorous artifact denial is the only way to bypass this skepticism.
3. **Cross-Secret Generalization Proof**: Cryptographers care about breaking the scheme, not a single hardcoded key. Proving the model generalizes across keys proves it learned the underlying math.
4. **Direct Classical Baseline Overlay**: Plotting ML Advantage identically over $\chi^2$ Advantage on the exact same noise sweep proves definitively that ML extracts deeper correlations than standard statistics.
5. **BCa Confidence Intervals**: Demonstrates statistical maturity. Tight, bias-corrected bounds on the Advantage prove the attack is definitively non-random at the margins.
