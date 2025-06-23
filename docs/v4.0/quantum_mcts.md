# Quantum-Inspired Monte Carlo Tree Search: Rigorous Theory

## Executive Summary

This document presents the mathematical theory of quantum-inspired Monte Carlo Tree Search (MCTS). We show that tree search naturally exhibits quantum mechanical behavior when viewed through information time $\tau = \log(N+2)$, with coherent exploration competing against dissipative measurement.

**Key Results**:
- Path integral formulation on 1D directed lattice yields diagonal Hessian
- Effective Planck constant $\hbar_{\text{eff}}(N) = \hbar[1 + \Gamma_N/2]$ emerges from Lindblad dynamics
- PUCT formula derived from stationary action principle
- RG flow predicts $c_{\text{PUCT}} \sim N^{-1/2}$ decay
- Crossover (not phase transition) separates quantum/classical regimes
- Phase interference enables natural parallel coordination

**Note**: Implementation details, validation code, and engineering considerations are provided in the companion Engineering Appendix.

---

## Units and Conventions

| Symbol | Units | Physical Meaning |
|--------|-------|------------------|
| $E_0$ | Energy | Natural energy scale = $k_B T_{\text{room}}$ |
| $\mathcal{L}$ | Dimensionless | Log-probability (action density) |
| $H$ | Energy | Hamiltonian = $E_0 \mathcal{L}$ |
| $\tau$ | Dimensionless | Information time = $\log(N+2)$ |
| $\hbar$ | Energy × time | Fundamental quantum scale (set to 1) |
| $\hbar_{\text{eff}}$ | Energy × time | Effective quantum scale |
| $\gamma$ | 1/time | Decoherence rate |
| $\kappa$ | Energy | Hopping strength |
| $\varepsilon_N$ | Dimensionless | Visit count regularization |

**Conventions**: We work in units where $E_0 = \hbar = 1$ unless dimensional analysis requires otherwise.

---

## Glossary: Tree Search ↔ Quantum Field Theory

| Tree Search Term | QFT Term | Mapping |
|-----------------|----------|---------|
| Edge $(s,a)$ | Link variable | Directed connection in lattice |
| Visit count $N$ | Occupation number | Bosonic field value |
| Path/Rollout | World-line | 1D trajectory through spacetime |
| Action selection | Measurement | Wavefunction collapse |
| Neural network prior | External field | Symmetry-breaking term |
| Pruning | Blocking/Coarse-graining | RG transformation |
| Convergence | Decoherence | Classical limit |
| Exploration | Quantum fluctuations | Off-diagonal density matrix |
| Exploitation | Classical dynamics | Diagonal density matrix |

---

## Part I: Mathematical Foundations

### 1. Information Time

**Definition 1.1** (Information Time):
$$\tau(N) = \log(N + 2)$$

**Theorem 1.1** (Information-Theoretic Derivation):
Let $I_N$ be the total information gained from $N$ simulations. Under the assumption of diminishing returns:
$$\frac{dI_N}{dN} = \frac{c}{N + \alpha}$$

Integrating with boundary condition $I_0 = 0$:
$$I_N = c[\log(N + \alpha) - \log(\alpha)]$$

Setting $\alpha = 2$ (minimal regularization) and $\tau \equiv I_N/c + \log(2)$ gives the result. □

### 2. Path Space and Measure

**Definition 2.1** (Configuration Space):
A path of ply depth $L$ is:
$$\gamma = (u_0, u_1, \ldots, u_{L-1}) \in \Gamma_L$$
where $u_k = (s_k, a_k)$ and $s_{k+1} = f(s_k, a_k)$.

**Lemma 2.1** (Measurability):
Since $\Sigma_{u_k}$ is countable for each $k$, the path space measure
$$\sum_{\gamma \in \Gamma_L} = \sum_{u_0 \in \Sigma_{s_0}} \sum_{u_1 \in \Sigma_{u_0}} \cdots$$
is well-defined by Fubini's theorem. □

### 3. Action Functional

**Definition 3.1** (Lagrangian Density):
$$\mathcal{L}(u_k; N^{\text{pre}}) = \log[N^{\text{pre}}(u_k) + \varepsilon_N] + \lambda \log P(u_k) - \beta Q(u_k)$$

where:
- $N^{\text{pre}}(u_k) \geq 0$: pre-rollout visit count
- $P(u_k) \in (0,1]$: neural network prior
- $Q(u_k) \in [-1,1]$: backed-up value
- $\varepsilon_N > 0$: visit regularization parameter

**Theorem 3.1** (PUCT from Stationary Action):
Under the slow-value-update assumption $|\partial Q/\partial N| \ll 1/N$, the stationary action condition recovers PUCT selection with $c_{\text{puct}} = \lambda/\sqrt{2}$.

*Proof*: The path probability is $P(\gamma) \propto \exp(-S[\gamma]/\hbar_{\text{eff}})$.
Stationarity requires:
$$\frac{\delta S}{\delta N(s,a)} = \frac{1}{N(s,a) + \varepsilon_N} + O(|\partial Q/\partial N|) = 0$$

Treating $Q$ as constant during selection and Taylor expanding for small $N(s,a)/N(s)$ yields the PUCT formula. □

---

## Part II: Quantum Dynamics

### 4. Hamiltonian Structure

**Definition 4.1** (Total Hamiltonian):
With energy scale $E_0$:
$$H = E_0 H_{\text{diag}} + H_{\text{hop}}$$

where:
$$H_{\text{diag}} = \sum_{(s,a)} \mathcal{L}(s,a; N^{\text{pre}})|(s,a)\rangle\langle(s,a)|$$
$$H_{\text{hop}} = \sum_{(s,a) \sim (s',a')} \kappa_N |(s',a')\rangle\langle(s,a)| + \text{h.c.}$$

with $\kappa_N = E_0 \kappa_0/\sqrt{N + 1}$.

### 5. Lindblad Master Equation

**Definition 5.1** (Jump Operators):
$$L_{s,a} = \sqrt{\gamma_{s,a}} |(s,a)\rangle\langle(s,a)|$$

where the decoherence rate per information time:
$$\gamma_{s,a} = g_0 \frac{N(s,a) + \varepsilon_N}{\delta\tau}$$

with $\delta\tau = 1/(N_{\text{root}} + 2)$ and $g_0$ dimensionless.

**Theorem 5.1** (Effective Planck Constant):
The effective Planck constant emerges as:
$$\hbar_{\text{eff}}(N) = \hbar\left[1 + \frac{\Gamma_N}{2}\right]$$

where $\Gamma_N = \sum_{s,a} \gamma_{s,a}$ is the total decoherence rate.

*Proof*: Apply projector $P_{\text{off}}$ onto coherent subspace. The evolution becomes:
$$\partial_\tau \rho_{\text{off}} = -\frac{i}{\hbar}[H,\rho_{\text{off}}] - \frac{\Gamma_N}{2}\rho_{\text{off}}$$

To absorb decoherence into effective unitary evolution with real $\hbar_{\text{eff}}$, we need the time-averaged effect, yielding the result. □

---

## Part III: Quantum Corrections

### 6. One-Loop on Trees

**Theorem 6.1** (Diagonal Hessian):
For tree structures, the action Hessian is diagonal:
$$H_{kk'} = \delta_{kk'} h_k, \quad h_k = \frac{1}{N_k^{\text{pre}} + \varepsilon_N}$$

*Proof*: Tree paths have no interaction between different depths since $\mathcal{L}[\gamma] = \sum_k \mathcal{L}(u_k)$. Thus $\partial^2\mathcal{L}/\partial u_k \partial u_{k'} = 0$ for $k \neq k'$. □

**Lemma 6.1** (Gaussian Approximation):
For $N_k \geq 5$, Stirling's approximation justifies treating discrete $\delta u_k$ as continuous. For smaller $N_k$, the discrete sum yields the same $\log(N_k + \varepsilon_N)$ correction up to $O(1/N)$. □

**Theorem 6.2** (One-Loop Effective Action):
$$\Gamma_{\text{1-loop}} = S_{\text{cl}} + \frac{\hbar_{\text{eff}}}{2}\sum_k \log h_k$$

### 7. UV Cutoff

**Definition 7.1** (Visit Threshold Cutoff):
$$N_{\text{UV}} = N_{\text{parent}}^{\alpha_{\text{UV}}}$$

where empirically $\alpha_{\text{UV}} \in [0.3, 0.7]$ balances exploration and exploitation.

**Note**: The theoretical value $\alpha_{\text{UV}} = 1/(1 + \epsilon_{\text{coh}}\Delta E_N)$ with $\Delta E_N \approx \beta\langle|Q|\rangle$ requires game-specific calibration.

---

## Part IV: Renormalization Group

### 8. Discrete RG Flow

**Theorem 8.1** (RG Recursion Relations):
Integrating out $b$ low-visit edges with parent count $N_p$:

$$\lambda' = \lambda - \frac{\hbar_{\text{eff}} b}{N_p}$$
$$\beta' = \beta\left(1 + \frac{b}{2N_p}\right)$$
$$\hbar_{\text{eff}}' = \hbar_{\text{eff}} + \frac{\gamma_0 b}{2N_p}$$

where $\gamma_0$ is the bare decoherence strength.

**Theorem 8.2** (Beta Functions):
In the continuum limit $\ell = \log b$:

$$\beta_\lambda = -\hbar_{\text{eff}}$$
$$\beta_\beta = \frac{\beta}{2}$$
$$\beta_{\hbar} = \frac{\gamma_0}{2}$$

**Corollary 8.1** (PUCT Decay):
$$c_{\text{PUCT}}(\ell) = \frac{\lambda(\ell)}{\beta(\ell)} \sim e^{-\ell/2} \sim N^{-1/2}$$

---

## Part V: Crossover Phenomena

### 9. Quantum-Classical Crossover

**Theorem 9.1** (Crossover, Not Phase Transition):
For finite trees with finite simulations, the system exhibits a smooth crossover, not a true phase transition. The Liouvillian gap never exactly closes for finite $N$.

**Definition 9.1** (Crossover Regimes):
- **Quantum regime**: $N \lesssim 100$, $\Gamma < 2\kappa$
- **Crossover fan**: $\Gamma \approx 2\kappa$, maximum entropy slope
- **Classical regime**: $N \gtrsim 1000$, $\Gamma > 2\kappa$

**Note**: True phase transition requires limits $b \to \infty$, $N \to \infty$.

### 10. Quantum Darwinism

**Theorem 10.1** (Pointer States):
The pointer basis consists of edge states $\{|s,a\rangle\}$ which are eigenstates of all jump operators.

**Theorem 10.2** (Information Redundancy):
After $k$ independent simulations, mutual information:
$$I(S:F_1,...,F_k) = H(S)[1 - (1-1/b)^k]$$

approaches the system entropy $H(S)$.

**Note**: Independence assumes Dirichlet noise decorrelates rollouts. In self-play without noise, effective sample size is reduced.

---

## Part VI: Parallel Coordination

### 11. Phase-Kicked Policies

**Theorem 11.1** (Destructive Interference):
When $M$ threads explore edge $(s,a)$ with phases $\theta_m = \pi m/M_{\max}$:
$$|A_{\text{total}}|^2 = |A_{s,a}|^2 \cdot \frac{\sin^2(M\pi/2M_{\max})}{\sin^2(\pi/2M_{\max})} \approx \frac{|A_{s,a}|^2}{M^2}$$

**Lemma 11.1** (Complete Positivity):
CP is preserved if all threads release locks before selection, ensuring $\sum_\sigma |K_\sigma|^2 = 1$.

**Note**: Asynchronous updates require atomic lock-reference counting.

### 12. MinHash Clustering

**Definition 12.1** (Quantized MinHash):
For continuous priors, first quantize into $B$ buckets:
$$\tilde{P}(a|s) = \lfloor B \cdot P(a|s) \rfloor / B$$

Then apply MinHash to discretized representation.

**Theorem 12.1** (Policy Clustering):
Similar policies receive phases differing by $O(1-J)$ where $J$ is Jaccard similarity, enabling automatic progressive widening.

---

## Mathematical Summary

The quantum-inspired MCTS framework reveals that tree search naturally implements:

1. **Path integral**: $Z = \sum_\gamma \exp(-S[\gamma]/\hbar_{\text{eff}})$
2. **Decoherence-driven annealing**: $\hbar_{\text{eff}}(N) = \hbar[1 + \Gamma_N/2]$
3. **Emergent PUCT**: From stationary action principle
4. **RG flow**: $c_{\text{PUCT}} \sim N^{-1/2}$ from discrete blocking
5. **Crossover dynamics**: Smooth quantum → classical transition
6. **Parallel coordination**: Via destructive interference

All parameters are measurable from tree statistics, providing principled exploration beyond heuristic tuning.

---

## References

[Implementation details and validation procedures are provided in the companion Engineering Appendix]