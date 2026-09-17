# LLM Quantization: How to Run Big Models on Tiny Hardware
---

## 1. Introduction: The Precision-Performance Trade-off

In the deployment of Large Language Models (LLMs), the sheer scale of modern model architectures presents a formidable barrier. A 70-billion-parameter foundation model stored in standard 16-bit floating-point format ($\text{FP16}$ or $\text{BF16}$) requires **$\approx 140\text{ GB}$ of High Bandwidth Memory (HBM)** purely to load static model weights, excluding the dynamic memory required for Key-Value (KV) caching and runtime activations.

This memory footprint fundamentally constrains inference throughput, escalates infrastructure costs, and prevents deployment on resource-constrained edge accelerators.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    The Core Proposition of Quantization                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   High-Precision Continuous Domain             Low-Precision Discrete Domain│
│   [ FP32 / FP16 / BF16 ]                       [ FP8 / INT8 / INT4 / 1-bit ]│
│   • 16 to 32 bits per value                    • 1 to 8 bits per value      │
│   • Enormous dynamic range                     • Bounded discrete grid      │
│   • Memory Bandwidth Bottleneck                • 2x to 8x Throughput Surge  │
│                                                                             │
│                                                                             │
│             Loss of Precision ──────► Massive Memory & Compute Gain         │
│          (Controlled Discretization)     (Accelerated Memory Bandwidth)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Quantization** is the process of mapping high-precision, continuous numerical representations (e.g., 32-bit or 16-bit floating-point numbers) to lower-precision, discrete numerical formats (e.g., 8-bit, 4-bit, or ternary integer/floating-point values). 

While quantization intentionally introduces a **loss of mathematical precision**, modern deep neural networks—and LLMs in particular—exhibit remarkable **overparameterization and structural redundancy**. By applying mathematically principled quantization algorithms, systems engineers can achieve **$2\times$ to $8\times$ memory compression** and **$2\times$ to $4\times$ inference speedups** with near-zero loss in downstream reasoning accuracy and perplexity.

---

## 2. Mathematical Foundations of Quantization

At its core, quantization projects a real continuous value $x \in [\alpha, \beta] \subset \mathbb{R}$ onto a finite set of $2^b$ discrete quantization bins, where $b$ is the target bit-width.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Uniform Quantization Function                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        Continuous Real Line (FP16):                                         │
│        ◄───────|──────────────|──────────────|──────────────|───────►       │
│               -3.4           -1.1           +0.8           +2.9             │
│                        │              │              │                      │
│                        ▼  (Scale S, Zero-Point Z, Rounding, Clamping)       │
│                                                                             │
│        Discrete Quantized Grid (INT4: -8 to +7):                            │
│        ◄───────[■]────────────[■]────────────[■]────────────[■]───────►     │
│                -8             -3              +2             +7             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Uniform Affine (Asymmetric) Quantization
The standard uniform affine quantization function maps a real tensor $\mathbf{X}$ to an integer grid $\mathbf{X}_q \in [q_{\min}, q_{\max}]$:

$$\mathbf{X}_q = \text{clip}\left( \left\lfloor \frac{\mathbf{X}}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max} \right)$$

where:
* $\lfloor \cdot \rceil$ denotes a rounding-to-nearest-integer operation.
* $\text{clip}(v, a, b) = \max(a, \min(v, b))$.
* $S \in \mathbb{R}^+$ is the **Scale Factor**, defining the step size between consecutive discrete levels.
* $Z \in \mathbb{Z}$ is the **Zero-Point**, ensuring that the real value $0.0$ maps exactly to an integer in the quantized space (vital for zero-padding in convolutional/attention layers).

The scale factor $S$ and zero-point $Z$ are derived from the real tensor's dynamic range $[\alpha, \beta] = [\min(\mathbf{X}), \max(\mathbf{X})]$:

$$S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}$$

$$Z = \text{clip}\left( \left\lfloor \frac{-\alpha}{S} \right\rceil + q_{\min}, \; q_{\min}, \; q_{\max} \right)$$

#### Dequantization
To reconstruct the approximated floating-point tensor $\hat{\mathbf{X}} \approx \mathbf{X}$:

$$\hat{\mathbf{X}} = S \cdot (\mathbf{X}_q - Z)$$

The quantization error (or distortion) is defined as:

$$\mathbf{E} = \mathbf{X} - \hat{\mathbf{X}} = \mathbf{X} - S \cdot \left( \text{clip}\left( \left\lfloor \frac{\mathbf{X}}{S} \right\rceil + Z, q_{\min}, q_{\max} \right) - Z \right)$$

---

### 2.2 Symmetric Quantization
When the distribution of values is approximately symmetric around zero, setting $Z = 0$ simplifies hardware computation:

$$S = \frac{\max(|\mathbf{X}|)}{q_{\max}}$$

$$\mathbf{X}_q = \text{clip}\left( \left\lfloor \frac{\mathbf{X}}{S} \right\rceil, \; -q_{\max}, \; q_{\max} \right)$$

$$\hat{\mathbf{X}} = S \cdot \mathbf{X}_q$$

**Hardware Benefit**: In symmetric quantization, matrix multiplications simplify because the zero-point cross-terms vanish:
$$\mathbf{Y} = \mathbf{X}\mathbf{W} \approx (S_X \mathbf{X}_q)(S_W \mathbf{W}_q) = (S_X S_W) \cdot (\mathbf{X}_q \mathbf{W}_q)$$
The core matrix multiplication $(\mathbf{X}_q \mathbf{W}_q)$ executes entirely using ultra-fast integer tensor cores (e.g., `DP4A` or `INT8 Tensor Core MMA` instructions), followed by a single floating-point scalar scaling per tile.

---

### 2.3 Quantization Granularity

The choice of scale sharing directly governs the trade-off between numerical accuracy and memory overhead:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Granularity of Scale Factors                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Per-Tensor:         Single S for entire weight matrix W (Low overhead,  │
│                         poor accuracy with outliers)                        │
│                                                                             │
│  2. Per-Channel:        One S per output row/column (Standard for weights) │
│                                                                             │
│  3. Group-wise / Block: Matrix partitioned into small blocks (e.g., g=64);  │
│                         independent S per group (High accuracy, best for    │
│                         INT4/INT2 regimes)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Per-Tensor Quantization**: A single scale factor $S$ for the entire tensor matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$. Outliers in one channel degrade quantization resolution for all other channels.
2. **Per-Channel (Per-Row/Per-Column) Quantization**: Each row or column maintains its own scale factor $S_i$. This isolates variance across channels, preserving accuracy with negligible memory overhead ($O(d_{\text{out}})$ scales).
3. **Group-wise (Block-wise) Quantization**: The weight tensor is partitioned into contiguous sub-vectors of size $g$ (typically $g \in \{32, 64, 128\}$), with an independent scale factor $S_{i, j}$ for each group. This is the gold standard for sub-4-bit quantization (AWQ, GPTQ).

---

## 3. Numerical Data Formats: Floating-Point vs. Integer

Modern AI accelerators support a rich hierarchy of numerical representations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Bit-Level Layout of Common AI Data Formats               │
├─────────────────────────────────────────────────────────────────────────────┤
│  FP16 (16-bit):  [ Sign: 1 ] [ Exponent: 5 bits  ] [ Mantissa: 10 bits   ]  │
│  BF16 (16-bit):  [ Sign: 1 ] [ Exponent: 8 bits  ] [ Mantissa: 7 bits    ]  │
│                                                                             │
│  FP8 E4M3:       [ Sign: 1 ] [ Exponent: 4 bits  ] [ Mantissa: 3 bits    ]  │
│  FP8 E5M2:       [ Sign: 1 ] [ Exponent: 5 bits  ] [ Mantissa: 2 bits    ]  │
│                                                                             │
│  INT8 (8-bit):   [ Sign: 1 ] [ Two's Complement Integer Magnitude: 7 b   ]  │
│  INT4 (4-bit):   [ Sign: 1 ] [ Two's Complement Integer Magnitude: 3 b   ]  │
│  Ternary (1.58b):[ Value in {-1, 0, +1} mapped to 2-bit binary integer   ]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Data Format | Bit Width | Exponent / Mantissa | Dynamic Range | Primary Usage / Hardware Support |
| :--- | :--- | :--- | :--- | :--- |
| **FP16** | 16 | 5 / 10 | $\approx 10^{-5}$ to $6.5 \times 10^4$ | Standard training & baseline inference |
| **BF16** | 16 | 8 / 7 | $\approx 10^{-38}$ to $3.4 \times 10^{38}$ | Preferred pretraining format (matches FP32 range) |
| **FP8 (E4M3)** | 8 | 4 / 3 | $\approx 10^{-2}$ to $4.48 \times 10^2$ | Inference weights & forward activations (NVIDIA Hopper/Blackwell) |
| **FP8 (E5M2)** | 8 | 5 / 2 | $\approx 10^{-5}$ to $5.73 \times 10^4$ | Gradients, KV-cache, and high dynamic range layers |
| **INT8** | 8 | N/A (Linear) | $[-128, 127]$ | Classic post-training integer inference |
| **INT4** | 4 | N/A (Linear) | $[-8, 7]$ | Weight-only quantized LLMs (AWQ, GPTQ) |
| **NVFP4** | 4 | 2 / 1 (Microscaled) | Varied per tile | Native Blackwell sub-byte tensor core format |
| **Ternary (BitNet)**| 1.58 | N/A | $\{-1, 0, +1\}$ | Zero-multiplication integer addition architectures |

---

## 4. Major Quantization Methodologies for LLMs

Quantization approaches are broadly categorized into **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Taxonomy of LLM Quantization Algorithms                  │
├──────────────────────────────────────┬──────────────────────────────────────┤
│  Post-Training Quantization (PTQ)    │  Quantization-Aware Fine-Tuning (QAT)│
├──────────────────────────────────────┼──────────────────────────────────────┤
│  • Weight-Only (W4A16, W8A16)        │  • QLoRA (NF4 + Double Quantization) │
│    - GPTQ (Second-order Hessian)     │  • Straight-Through Estimator (STE)  │
│    - AWQ (Activation-aware scaling)  │  • Distillation-based QAT            │
│  • Weight-Activation (W8A8, W4A4)    │                                      │
│    - SmoothQuant (Channel migration) │                                      │
│    - LLM.int8() (Outlier separation) │                                      │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

### 4.1 Weight-Only Post-Training Quantization (W4A16 / W8A16)

In memory-bound autoregressive decoding, memory bandwidth is dominated by reading model weights. Weight-only quantization compresses weights to 4-bit or 8-bit while keeping activations in 16-bit floating point ($\text{W4A16}$). During computation, weights are dequantized on-the-fly in fast on-chip SRAM registers before matrix multiplication.

#### A. GPTQ (Generalized Post-Training Quantization)
GPTQ frames quantization as an optimal brain surgeon optimization problem. For a weight matrix $\mathbf{W}$ and input calibration activations $\mathbf{X}$, GPTQ minimizes the squared output error:

$$\min_{\hat{\mathbf{W}}} \|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_2^2$$

Using a second-order Taylor expansion, the optimal quantization step for weight column $q$ updates the remaining unquantized weights $\mathbf{W}_{:, >q}$ using the inverse Hessian matrix $\mathbf{H} = 2\mathbf{X}\mathbf{X}^T$:

$$\mathbf{W}_{:, >q} \leftarrow \mathbf{W}_{:, >q} - \frac{w_q - \text{quant}(w_q)}{[\mathbf{H}^{-1}]_{qq}} \cdot \mathbf{H}^{-1}_{:, >q}$$

**Impact**: Quantizes a 70B model to 4-bit in under 4 GPU hours with minimal loss in perplexity.

#### B. AWQ (Activation-Aware Weight Quantization)
AWQ observes that **not all weights are equally important**. By inspecting the activation magnitudes $\mathbf{S}_X = \frac{1}{N}\sum |\mathbf{X}|$, AWQ identifies the top 1% of salient weight channels that carry critical semantic information.

Rather than keeping salient weights in FP16 (which creates hardware execution divergence), AWQ applies a per-channel protection scale $s > 1$:

$$\mathbf{W}' = \mathbf{W} \cdot \text{diag}(s), \quad \mathbf{X}' = \text{diag}(s)^{-1} \cdot \mathbf{X}$$

$$\mathbf{W}'\mathbf{X}' = (\mathbf{W} \cdot \text{diag}(s))(\text{diag}(s)^{-1} \cdot \mathbf{X}) = \mathbf{W}\mathbf{X}$$

By mathematically scaling down the activation outliers and scaling up the weights prior to group-wise quantization, the relative quantization error on salient channels is minimized.

---

### 4.2 Weight-Activation Quantization (W8A8 / W4A4)

To accelerate both compute-bound prefill phases and memory-bound decode phases, both weights and activations must be quantized ($\text{W8A8}$). This enables true integer matrix multiplications ($\text{INT8} \times \text{INT8} \rightarrow \text{INT32}$) on hardware Tensor Cores.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   The SmoothQuant Activation-Weight Migration               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Activations (Severe Outliers)          Raw Weights (Smooth)           │
│   [  0.1   120.5    0.2   -85.4  ]     x    [  0.4   -0.2    0.1    0.3  ]  │
│                  │                                       │                  │
│                  ▼                                       ▼                  │
│   Divide Activations by Scale s              Multiply Weights by Scale s    │
│                  │                                       │                  │
│                  ▼                                       ▼                  │
│   Smoothed Activations (Easy INT8)          Smoothed Weights (Easy INT8)    │
│   [  0.5     1.2    0.4    -1.1  ]     x    [  2.4   -1.8    0.6    1.5  ]  │
│                                                                             │
│   Result: INT8 x INT8 Tensor Core Matrix Multiplication with Zero Outliers! │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### A. The Emergent Outlier Problem
In models exceeding $6.7\text{B}$ parameters, activation tensors exhibit **emergent systematic outliers**—specific feature dimensions whose magnitudes reach $100\times$ the average activation value. In standard per-tensor or per-token activation quantization, these outliers force the scale factor $S$ to expand, destroying the quantization resolution for all non-outlier channels.

#### B. SmoothQuant
SmoothQuant solves this by mathematically migrating the quantization difficulty from activations to weights. Because weights have a uniform, narrow dynamic range, they can absorb variance from activations:

$$s_j = \frac{\max(|\mathbf{X}_{:, j}|)^\alpha}{\max(|\mathbf{W}_{j, :}|)^{1-\alpha}}$$

where $\alpha \in [0, 1]$ is a hyperparameter balancing difficulty between activations and weights ($\alpha = 0.5$ is typical). Both activations and weights are smoothly quantized into INT8, enabling 100% INT8 matrix multiplication across all linear layers.

---

### 4.3 Quantization-Aware Fine-Tuning: QLoRA

**QLoRA (Dettmers et al., 2023)** enables parameter-efficient fine-tuning of multi-billion parameter models on single consumer GPUs by introducing three key innovations:

1. **NormalFloat4 (NF4)**: An information-theoretically optimal quantile quantization format for normally distributed weights, allocating discrete bins such that each bin has an equal probability mass.
2. **Double Quantization (DQ)**: Quantizing the quantization scale factors themselves (compressing 32-bit float scale constants to 8-bit integers), saving an additional $0.37\text{ bits/parameter}$.
3. **Paged Optimizers**: Utilizing CUDA unified virtual memory to automatically page optimizer states between GPU VRAM and CPU RAM during gradient checkpointing spikes.

---

## 5. Systems and Performance Gains: Memory, Bandwidth, and Energy

Quantization delivers transformative systems-level performance advantages across modern GPU architectures:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Impact of Quantization Across Scale                      │
├────────────────────────────┬───────────────────────┬────────────────────────┤
│  Model: LLaMA-3-70B        │  Static VRAM Required │  Tokens/sec Throughput │
├────────────────────────────┼───────────────────────┼────────────────────────┤
│  FP16 (Baseline)           │  ~140 GB (2x A100 80G)│  Baseline (1.0x)       │
│  FP8 (E4M3)                │  ~70 GB (1x A100 80G) │  1.8x – 2.2x           │
│  INT4 (AWQ / GPTQ)         │  ~35 GB (1x A100 80G) │  2.5x – 3.8x           │
│  1.58-bit (BitNet b1.58)   │  ~14 GB (Consumer GPU)│  4.0x – 6.0x           │
└────────────────────────────┴───────────────────────┴────────────────────────┘
```

### 1. Arithmetic Intensity Shift
During decode, the arithmetic intensity is memory bandwidth bound:
$$I = \frac{\text{Operational FLOPs}}{\text{Bytes Transferred}}$$
Reducing weight precision from 16 bits to 4 bits cuts memory traffic by **$4\times$**, effectively quadrupling the operational arithmetic intensity on the Roofline model and directly translating into a proportional decrease in Inter-Token Latency (ITL).

### 2. Energy Efficiency & Silicon Footprint
Integer arithmetic requires significantly less silicon area and electrical energy than floating-point arithmetic:
* An **$\text{INT8}$ addition** consumes $\approx 0.03\text{ pJ}$ compared to $\approx 0.4\text{ pJ}$ for **$\text{FP16}$ addition** ($>10\times$ energy reduction).
* An **$\text{INT8}$ multiplication** consumes $\approx 0.2\text{ pJ}$ compared to $\approx 1.1\text{ pJ}$ for **$\text{FP16}$ multiplication**.

---

## 6. Challenges and Failure Modes in Quantization

1. **Accuracy Degradation at $<3$ Bits**: Uniform linear quantization breaks down below 3 bits per parameter due to catastrophic representational collapse. Overcoming this requires non-linear codebooks (e.g., AQLM, QuIP#) or native 1-bit pretraining (BitNet).
2. **Dequantization Kernel Overhead**: In naive implementations, unpacking 4-bit integers and converting them to FP16 in CUDA kernels can introduce memory overhead that negates bandwidth savings. Production engines require hand-optimized, register-level SIMD assembly (e.g., Marlin, CUTLASS GEMM kernels).
3. **KV-Cache Quantization Drift**: Unlike static weights, dynamic activations in the KV cache expand over thousands of tokens. Asymmetric drift in Key vectors can cause attention score distortions unless per-channel scaling and attention sinks are preserved.

---

## 7. Conclusion

Quantization represents one of the most foundational disciplines in deep learning systems engineering. By trading redundant floating-point precision for compact discrete representations, it resolves the fundamental memory-bandwidth wall of transformer inference.

Through mathematical breakthroughs like **Hessian-based compensation (GPTQ)**, **salient activation scaling (AWQ)**, **channel smoothing (SmoothQuant)**, and **optimal NormalFloat distributions (QLoRA)**, quantization enables frontier intelligence to run with unprecedented speed, economic efficiency, and hardware portability. 

As silicon architectures converge around native sub-byte tensor cores (FP4, MXFP4) and zero-multiplication ternary computing (BitNet), quantization will remain the central bridge connecting theoretical AI models with scalable, real-world deployment.
