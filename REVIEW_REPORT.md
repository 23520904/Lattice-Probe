# Lattice-Probe Expert Code Review & Improvement Report

## 1. Executive Assessment
**Score**: 7/10
* **Cryptographic rigor**: 8/10
* **ML methodology**: 5/10
* **Engineering quality**: 8/10
* **Publication readiness**: 5/10

**Reasoning**: The codebase represents a clean, efficient, and well-tested implementation of the FIPS-203 Module-LWE scheme. The NTT correctly manages twiddle factors and the data generation respects strict CSPRNG isolation, ensuring no cross-sample leakage. However, the partial-bit recovery experiment is structurally flawed (leading to failed logistic regressions) and the GNN architecture leverages `GraphSAGE` in a way that blindly ignores edge features containing polynomial coefficients. These methodology gaps currently compromise the scientific validity of the results, requiring immediate fixes before publication.

---

## 2. Cryptographic Audit
The cryptographic foundations are robust, but feature nuanced design decisions:

* **NTT Implementation (`ring.py`)**: Uses positive `ZETAS` during INTT. FIPS-203 Algorithm 42 specifies `-zeta`, but this deviation matches the actual `pqcrystals` reference C implementation. This is mathematically sound.
* **Rejection Sampling (`sampler.py`)**: `discrete_gaussian` drops tails beyond 6σ. For $2^{20}$ samples, the probability of encountering a 6σ event is negligible. This is acceptable for the evaluated bounds.
* **Secret Generation**: `generate_dataset.py` reuses a single secret across all dataset splits. While this correctly mirrors a distinguishing attack against a *single public key*, it fundamentally breaks the bit recovery experiment (detailed below).
* **RNG Usage**: `fresh_rng()` cleanly isolates CSPRNG state per sample.

*Severity*: CRITICAL
*Location*: `generate_dataset.py` & `bit_recovery.py`
*Problem*: Bit recovery trains on fixed-secret dataset splits.
*Why it matters*: Logistic Regression target bits variance is 0. The regressors cannot fit, resulting in `NaN` accuracy for every bit.
*Suggested fix*: Restructure `bit_recovery.py` to generate or load samples featuring varying secrets per sample.

---

## 3. Data Leakage Audit
No true leakage exists that could allow the model to cheat, but one minor artifact exists:

* **Shard Generation Precision**: `save_shard` downcasts the strictly integer Module-LWE arrays (coefficients $<3329$) to `float32`. This wastes memory/disk space, but because $3329 \ll 2^{24}$, no precision is lost. This should be optimized to `int16`.
* **Preprocessing Leakage**: The data is strictly isolated across splits, and the `BCEWithLogitsLoss` training is cleanly separated.
* **Checkpoint Metadata**: The `args` dictionary is meticulously maintained, which is excellent for reproducibility.

---

## 4. Representation Review

### Transformer Representation
* **Coefficient encoding**: The `to_sequence` method concatenates `a` and `b` linearly. 
* **Positional embeddings**: Basic 1D embeddings are used. This fails to encode the 2D algebraic topology of the polynomial modules (degrees $0..255$ within module variables $0..k$). 
* **Recommendation**: Implement hierarchical positional embeddings (Module ID + Polynomial Degree ID) to give the Transformer structural ring-awareness.

### Graph Representation
* **Node/Edge design**: The bipartite design links variable node $c$ to equation node $c$.
* **Message passing limitations**: By linking $c \leftrightarrow c$, the topology explicitly models *point-wise* multiplication. However, the coefficients represent time-domain polynomials, which require *negacyclic convolution*. The graph completely fails to capture the true structural dependencies of $b = a \cdot s + e$.
* **Recommendation**: Map polynomials to the NTT domain *before* constructing the graph. Only in the NTT domain does point-wise multiplication hold true.

---

## 5. Model Architecture Review

### Transformer
* **Current architecture**: 8-layer, pre-LN model using a CLS token.
* **Assessment**: Sufficient for establishing baseline Transformer capacity, but standard 1D positional encodings are sub-optimal for algebraic representations.
* **Recommendation**: Implement Rotary Positional Embeddings (RoPE) to allow the attention mechanism to natively grasp the cyclic shift nature of polynomial rings.

### GNN
* **Current architecture**: 6-layer `GraphSAGE`.
* **Assessment**: GraphSAGE is a fatal choice for this representation. It strictly aggregates neighbor node features and entirely ignores edge attributes. Since `to_graph` places the `a` coefficients on the edges, the GNN is completely blind to the `a` polynomials!
* **Recommendation**: Immediately replace `GraphSAGE` with an edge-aware architecture such as `GATConv` (Graph Attention Network) or `GINEConv`.

---

## 6. Statistical Validity Review
* **AUROC usage**: Accurate and appropriate for distinguishing tasks.
* **Bootstrap methodology**: The 100-resample percentile bootstrap is sufficient for establishing variance, though BCa (Bias-Corrected and Accelerated) intervals would be slightly more rigorous.
* **Significance testing**: The $\chi^2$ baseline correctly identifies underlying distribution variances to validate the ML baseline.
* **Suggestion**: The paper should explicitly translate AUROC into *Distinguishing Advantage*, a standard cryptographic metric: `Advantage = 2 * AUROC - 1`.

---

## 7. Reproducibility Review
* **Seed management**: Controlled correctly via `nacl.utils.random` for non-deterministic cryptographic rigor, combined with numpy's `default_rng`.
* **Deterministic training**: The lack of fixed PyTorch seeds is fine since bootstrap intervals cover empirical variance, but makes exact run replication impossible.
* **Suggestion**: Include an optional `--seed` flag in `train.py` for auditability.

---

## 8. Research Gap Analysis
* **Secret Rotation**: The current experiments train and test on identical, fixed secrets within a split. A major missing ablation is whether a model trained on Secret $S_1$ can distinguish samples utilizing a completely unseen Secret $S_2$.
* **Noise Sweep**: Testing specific edge-cases ($W1, W2, W3$) is good, but a full scaling curve graphing AUROC against Gaussian $\sigma \in [0.1, 1.5]$ would drastically improve the paper's scientific authority.
* **Sample Complexity**: Testing distinguishing advantage over log-scale training dataset sizes ($2^{10}$ through $2^{24}$) should be included.

---

## 9. Publication Readiness Assessment
* **Target Venue**: IACR ePrint / TCHES.
* **Strongest aspects**: Flawless adherence to FIPS-203 math, extensive unit testing, and rigorous classical baseline comparisons.
* **Weakest aspects**: The partial-bit recovery experiment is currently broken, and the GNN representation makes fundamental architectural errors (GraphSAGE ignoring edge attributes). 
* **Required fixes before submission**: The `bit_recovery.py` and GNN architecture issues *must* be resolved, otherwise reviewers will rightfully reject the machine learning conclusions regarding GNNs.

---

## 10. Prioritized Improvement Roadmap

### Top 10 Improvements Ranked By Scientific Value

1. **Fix Bit-Recovery Dataset Methodology (CRITICAL)**
   * **Impact**: Restores the broken partial-bit recovery experiment.
   * **Difficulty**: Low
   * **Estimated effort**: 1 hour
   * **Reasoning**: The current script fails to fit logistic regressors due to zero-variance target labels. Must generate samples with fresh secrets.

2. **Replace GraphSAGE with GATConv/GINEConv (HIGH)**
   * **Impact**: Allows the GNN to actually see the `a` polynomials.
   * **Difficulty**: Low
   * **Estimated effort**: 1 hour
   * **Reasoning**: GraphSAGE ignores `edge_attr`. A convolution that incorporates edge attributes is mandatory to learn $b = a \cdot s + e$.

3. **Change Graph Topology to NTT Domain (HIGH)**
   * **Impact**: Accurately models the underlying math.
   * **Difficulty**: Medium
   * **Estimated effort**: 1 day
   * **Reasoning**: The current graph models point-wise multiplication, which is only valid if polynomials are transformed into the NTT domain first.

4. **Implement Secret Rotation Ablation (MED)**
   * **Impact**: Tests model generalization bounds.
   * **Difficulty**: Low
   * **Estimated effort**: 1 hour
   * **Reasoning**: Proves whether Transformers are learning the LWE distribution or simply memorizing a specific LWE secret key.

5. **Ring-aware Transformer Tokenization (MED)**
   * **Impact**: Gives the Transformer structural awareness.
   * **Difficulty**: Low
   * **Estimated effort**: 2 hours
   * **Reasoning**: Flat sequences are naive; 2D positional embeddings (Module ID + Degree ID) map natively to polynomial ring algebra.

6. **Report Distinguishing Advantage (MED)**
   * **Impact**: Aligns ML metrics with cryptographic literature.
   * **Difficulty**: Trivial
   * **Estimated effort**: 10 minutes
   * **Reasoning**: Cryptographers expect `Advantage = 2 * AUROC - 1`.

7. **Perform Fine-grained Noise Sweep (MED)**
   * **Impact**: Provides a definitive scaling curve.
   * **Difficulty**: Low
   * **Estimated effort**: 1 day (compute time)
   * **Reasoning**: $W1, W2, W3$ are sparse points. A continuous sweep graphs the exact mathematical boundary where LWE falls to ML attacks.

8. **Optimize Data Storage to int16 (LOW)**
   * **Impact**: Halves storage and I/O costs.
   * **Difficulty**: Trivial
   * **Estimated effort**: 5 minutes
   * **Reasoning**: $q=3329 < 32767$, so `float32` is a massive waste of resources.

9. **Upgrade Transformer with RoPE (LOW)**
   * **Impact**: May improve convergence.
   * **Difficulty**: Medium
   * **Estimated effort**: 3 hours
   * **Reasoning**: Rotary embeddings inherently understand cyclic shifts, heavily mimicking $X^n+1$ arithmetic.

10. **Refactor Baseline `run_mlp` Memory Profile (LOW)**
    * **Impact**: Prevents OOM crashes.
    * **Difficulty**: Low
    * **Estimated effort**: 30 minutes
    * **Reasoning**: Concatenating enormous numpy arrays into memory for `run_mlp` is an architecture bottleneck for datasets $>2^{20}$.
