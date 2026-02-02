# QuantumMusic

**Bridging Cold Start and Continuation Challenges in Music Recommendation Systems using Quantum Techniques**

A quantum-assisted music recommendation pipeline that combines **RAG** (Retrieval-Augmented Generation), **Classical KNN**, **Quantum K-NN (QKNN)**, and **Grover's Algorithm** for emotion-aware, context-driven song recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Quantum Components](#quantum-components)
- [Circuit Diagrams](#circuit-diagrams)
- [Results & Metrics](#results--metrics)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [References](#references)

---

## Overview

This project addresses:

1. **Cold-start problem** – New users or tracks with no interaction history  
2. **Emotional context** – Natural language queries like *"Calm but slightly energetic music for late-night coding"*  
3. **Quantum-assisted similarity** – QKNN (SWAP test) and Grover-based amplification for discovery and diversity  

**Tech stack:** Python, Qiskit, LangChain, FAISS, Sentence Transformers, Groq LLaMA, scikit-learn, pandas.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUANTUM MUSIC RECOMMENDATION                        │
└─────────────────────────────────────────────────────────────────────────────┘

  User Query (NL)     FAISS Retrieval      LLM Context        Context Vector
       │                     │                   │                   │
       ▼                     ▼                   ▼                   ▼
  "Calm energetic     Top-K song chunks    Structured summary    384D → 16D
   music for coding"   (embeddings)        (mood, energy, ...)   (PCA + norm)
                                                                     │
                                                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │              QUANTUM LAYER (Amplitude Encoding → 4 qubits)              │
  │  |ψ⟩ = v₀|0000⟩ + v₁|0001⟩ + ... + v₁₅|1111⟩                            │
  └─────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
    Classical KNN              Quantum KNN                 Grover Search
    (cosine similarity)        (SWAP test fidelity)         (amplify relevant)
         │                           │                           │
         ▼                           ▼                           ▼
    Top-K songs                Top-K by fidelity           Amplified ranking
    (high precision)           (stable, diverse)            (high diversity)
```

---

## Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | **RAG** | Natural language → FAISS retrieval → LLM context extraction → 16D context vector |
| 2 | **Context → Quantum** | 16D vector L2-normalized → amplitude encoding → 4-qubit state \|ψ⟩ |
| 3 | **KNN** | Classical cosine similarity between query 16D and song 16D vectors |
| 4 | **QKNN** | SWAP test between query state and song states → fidelity scores |
| 5 | **Grover** | Mark high-relevance songs (oracle) → diffusion → amplified probabilities |

---

## Quantum Components

### 1. Amplitude Encoding (16D → 4 Qubits)

A 16-dimensional normalized context vector **v** is encoded into a 4-qubit state:

```
|ψ⟩ = v₀|0000⟩ + v₁|0001⟩ + v₂|0010⟩ + ... + v₁₅|1111⟩
```

- Each component **vᵢ** is the amplitude for basis state **|i⟩** (e.g. |0011⟩ = index 3).
- Requirement: **‖v‖ = 1** so that the state is valid (probabilities sum to 1).
- Implemented in Qiskit using `Initialize(vector)` on 4 qubits.

**Example (conceptually):**

```
16D normalized vector (from RAG + PCA):
[0.25, -0.18, 0.12, 0.31, -0.15, 0.22, 0.08, -0.11,
 0.19, 0.14, -0.16, 0.13, 0.09, -0.20, 0.17, 0.23]

Quantum state:
|ψ⟩ = 0.25|0000⟩ - 0.18|0001⟩ + 0.12|0010⟩ + 0.31|0011⟩ + ...
```

### 2. SWAP Test (QKNN Similarity)

Used to estimate **fidelity** between query state |ψ⟩ and song state |φ⟩:

- **Fidelity** = |⟨ψ|φ⟩|² (0 = orthogonal, 1 = identical).
- Circuit: 1 ancilla + 4 qubits (query) + 4 qubits (song). Ancilla in |+⟩, controlled-SWAPs between register pairs, then H on ancilla and measure.
- **P(ancilla = 0)** = (1/2) + (1/2)|⟨ψ|φ⟩|² ⇒ **F = 2·P(0) − 1** (for real amplitudes).

### 3. Grover's Algorithm (Re-ranking)

- **Oracle:** Marks “good” songs (e.g. relevance above a threshold) by phase flip (Z on marked computational basis states).
- **Diffusion:** Inversion about the mean to amplify marked states.
- **Iterations:** ~ π/4 × √(N/M) for N candidates and M marked items.
- **Output:** Measurement gives indices with high probability for relevant songs; used to re-rank and boost diversity.

---

## Circuit Diagrams

The following diagrams are generated by the project. **To generate them locally**, from the project root run:

```bash
pip install qiskit qiskit[visualization] matplotlib numpy
python scripts/generate_circuits.py
```

Images are saved to `assets/images/`. If the images are not yet in the repo, run the command above and commit the generated PNGs.

### Amplitude Encoding Circuit (16D → 4 qubits)

The circuit below shows how a 16-dimensional normalized vector is loaded into a 4-qubit register using Qiskit’s `Initialize` (amplitude encoding).

![Amplitude Encoding Circuit](assets/images/amplitude_encoding_circuit.png)

*Caption: Amplitude encoding of a 16D context vector into 4 qubits. The Initialize block prepares |ψ⟩ = Σᵢ vᵢ|i⟩.*

### SWAP Test Circuit (QKNN)

Structure: ancilla (H → c-SWAPs → H) + two 4-qubit registers (query and song). Measurement of the ancilla gives P(0), from which fidelity is computed.

![SWAP Test Circuit](assets/images/swap_test_circuit.png)

*Caption: SWAP test for fidelity between query and song quantum states. Fidelity = 2·P(ancilla=0) − 1.*

### Grover Oracle & Diffusion

- **Oracle:** Multi-controlled Z (and X gates for basis state selection) to flip phase of marked states.
- **Diffusion:** H – X – multi-controlled Z – X – H for inversion about the mean.

![Grover Oracle](assets/images/grover_oracle.png)

*Caption: Grover oracle marking “good” (high-relevance) song indices.*

![Grover Diffusion](assets/images/grover_diffusion.png)

*Caption: Grover diffusion operator (inversion about the mean).*

![Grover Full Circuit](assets/images/grover_full_circuit.png)

*Caption: Full Grover circuit: superposition → oracle → diffusion (repeated) → measurement.*

---

## Quantum State Representation (Example)

For a single query, the 16D context vector (after PCA and L2 normalization) maps to a 4-qubit state as follows.

**Conceptual mapping:**

| Basis state | Binary | Amplitude (example) | Probability |
|-------------|--------|----------------------|-------------|
| \|0⟩   | 0000 | v₀ = 0.25 | 0.0625 |
| \|1⟩   | 0001 | v₁ = -0.18 | 0.0324 |
| \|2⟩   | 0010 | v₂ = 0.12 | 0.0144 |
| \|3⟩   | 0011 | v₃ = 0.31 | 0.0961 |
| ... | ... | ... | ... |
| \|15⟩  | 1111 | v₁₅ = 0.23 | 0.0529 |

**Qiskit verification:**

- Build state with `Statevector(vector_16D)` or circuit with `Initialize(vector_16D)`.
- Check `np.allclose(np.abs(statevector.data), np.abs(vector_16D))` and norm ≈ 1.

---

## Results & Metrics

Evaluation uses ground truth from classical similarity and compares KNN, QKNN, and Grover over Top-50/100/500/1000.

| Method | Precision (Top-50) | Recall (Top-50) | NDCG (Top-100) | Diversity |
|--------|--------------------|------------------|----------------|-----------|
| **KNN**  | 0.765 | 0.153 | 0.878 | 0.968 |
| **QKNN** | 0.383 | 0.077 | 0.561 | 0.749 |
| **Grover** | 0.325 | 0.065 | 0.371 | **0.987** |

- **KNN:** Best precision and NDCG; strong baseline.
- **QKNN:** Lower prediction error (MAE), good novelty and stability.
- **Grover:** Highest diversity and strong score amplification (e.g. top item ~0.9976); best for discovery and re-ranking.

*Metrics are normalized to [0,1]; higher is better except for error metrics.*

---

- Blanzieri, E., & Pastorello, D. (2024). *A quantum k-nearest neighbors algorithm based on Euclidean distance estimation.*
- Kerenidis, I., & Prakash, A. *Quantum recommendation systems.* arXiv:1603.08675.
- Sawerwain, M., & Wróblewski, M. (2018). *Recommendation systems with the quantum k-NN and Grover algorithms.* International Journal of Applied Mathematics and Computer Science, 29(1), 139–150.
- Wang, S., Xu, C., Ding, A. S., & Tang, Z. (2021). *Novel emotion-aware hybrid music recommendation method using deep neural network.* Electronics, 10(15), 1769.

---

## License

See repository or project documentation for license and dataset usage terms.
