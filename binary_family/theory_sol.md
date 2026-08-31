# Two-court renormalization curves

Let \(Y_k^c\) be the \(k\)-th token in channel \(c\), and let \(\mathcal F_{k-1}^c\) contain the entire interleaved history immediately before that token. Write

\[
p_k(y)=P^*(Y_k^c=y\mid \mathcal F_{k-1}^c),\qquad
q_k^\pm(y)=R^\pm(Y_k^c=y\mid \mathcal F_{k-1}^c),
\]

and

\[
\ell_k=\log\frac{q_k^+(Y_k^c)}{q_k^-(Y_k^c)},\qquad
\Lambda_n^c=\sum_{k=1}^n\ell_k.
\]

All asymptotics below are per channel token. Since there is one action and one observation per round, this is also the per-round asymptotic scale.

## 1. Shapes

### 1.1 The universal drift–martingale decomposition

**[Proved]** Whenever the relevant probabilities are positive,

\[
m_k:=E_{P^*}[\ell_k\mid\mathcal F_{k-1}^c]
   =D(p_k\Vert q_k^-)-D(p_k\Vert q_k^+).
\]

Consequently,

\[
\Lambda_n^c=A_n+M_n,\qquad
A_n=\sum_{k\le n}m_k,
\]

where \(M_n=\sum_{k\le n}(\ell_k-m_k)\) is a \(P^*\)-martingale. Also,

\[
E_{P^*}\Lambda_n^c
 =\sum_{k\le n}E_{P^*}
 \left[D(p_k\Vert q_k^-)-D(p_k\Vert q_k^+)\right].
\]

This follows directly by expanding the two conditional KL divergences.

Thus the court favors \(R^+\) exactly when, on average under the true conditional law, \(R^+\) is closer in log loss than \(R^-\). There is no general sign under a misspecified \(P^*\).

### 1.2 When the true law is one of the courts

Assume the two court predictions have common support.

**[Proved]** If \(P^*=R^-\), then

\[
Z_n^-:=e^{\Lambda_n^c}
\]

is a nonnegative \(R^-\)-martingale, since

\[
E_{R^-}[e^{\ell_k}\mid\mathcal F_{k-1}^c]
=\sum_y q_k^-(y)\frac{q_k^+(y)}{q_k^-(y)}=1.
\]

By Doob’s nonnegative-martingale convergence theorem—every nonnegative supermartingale converges almost surely to a finite limit—

\[
\Lambda_n^c\longrightarrow L\in\mathbb R
\quad\text{or}\quad
\Lambda_n^c\longrightarrow-\infty
\qquad R^-\text{-a.s.}
\]

Moreover,

\[
E_{R^-}[\ell_k\mid\mathcal F_{k-1}^c]
=-D(q_k^-\Vert q_k^+)\le 0,
\]

so \(E_{R^-}\Lambda_n^c\) is nonincreasing.

**[Proved]** If \(P^*=R^+\), then \(e^{-\Lambda_n^c}\) is a nonnegative \(R^+\)-martingale. Hence

\[
\Lambda_n^c\longrightarrow L\in\mathbb R
\quad\text{or}\quad
\Lambda_n^c\longrightarrow+\infty
\qquad R^+\text{-a.s.},
\]

and

\[
E_{R^+}[\ell_k\mid\mathcal F_{k-1}^c]
=D(q_k^+\Vert q_k^-)\ge 0.
\]

Therefore:

- **[Proved]** Under \(P^*=R^+\), oscillation between arbitrarily large positive and negative values, convergence to \(-\infty\), and a decreasing expected trajectory are impossible.
- **[Proved]** Under \(P^*=R^-\), oscillation between both infinities, convergence to \(+\infty\), and an increasing expected trajectory are impossible.
- **[Proved]** These restrictions disappear when \(P^*\) is neither court.

Individual increments can have either sign even when \(P^*=R^\pm\); the monotonicity statement concerns expectation, not sample paths.

### 1.3 A sharp information criterion under uniform positivity

Suppose the token alphabet is finite and, for some \(\varepsilon>0\),

\[
q_k^\pm(y)\ge\varepsilon
\]

for every reachable history, \(k\), and \(y\). Define

\[
I_n^-=\sum_{k\le n}D(q_k^-\Vert q_k^+),\qquad
I_n^+=\sum_{k\le n}D(q_k^+\Vert q_k^-).
\]

**[Proved]** Under \(R^-\),

\[
\Lambda_n^c=M_n-I_n^-.
\]

Uniform positivity implies, by compactness and the quadratic equivalence of KL divergence, squared Hellinger distance, and log-likelihood variance on the interior simplex,

\[
E_{R^-}\!\left[(\Delta M_k)^2\mid\mathcal F_{k-1}^c\right]
\le C\,D(q_k^-\Vert q_k^+).
\]

It follows that

\[
\begin{cases}
I_\infty^-<\infty
&\Longrightarrow \Lambda_n^c\text{ converges to a finite real limit},\\[2mm]
I_n^-\to\infty
&\Longrightarrow \displaystyle
\frac{\Lambda_n^c}{I_n^-}\to-1\quad\text{a.s.}
\end{cases}
\]

For the second assertion, the martingale strong law gives \(M_n/I_n^-\to0\); its hypothesis follows from the displayed quadratic-variation bound by grouping times at which \(I_n^-\) crosses successive dyadic levels.

**[Proved]** Symmetrically, under \(R^+\),

\[
\begin{cases}
I_\infty^+<\infty
&\Longrightarrow \Lambda_n^c\text{ has a finite limit},\\[2mm]
I_n^+\to\infty
&\Longrightarrow \displaystyle
\frac{\Lambda_n^c}{I_n^+}\to1\quad\text{a.s.}
\end{cases}
\]

Thus finite plateau versus divergence is controlled by cumulative channel information, not merely by whether the one-step predictions eventually merge.

### 1.4 Linear, sublinear, and oscillatory examples

**[Proved: finite nonzero plateau]** Let a static bit \(\theta\in\{0,1\}\) be revealed perfectly by the first observation. Give the readers interior priors \(w^\pm\), and let the actual bit be \(\theta\). After the first observation both readers know \(\theta\), so all later likelihood ratios equal \(1\), while

\[
\Lambda_\infty^x=\log\frac{w^+(\theta)}{w^-(\theta)}.
\]

Either sign is possible.

**[Proved: exactly zero]** If \(q_k^+=q_k^-\) on all reached histories in channel \(c\), then \(\Lambda_n^c\equiv0\). Conversely, if \(\Lambda_n^c=0\) after every channel token, then each realized increment is zero. Under full support this means \(q_k^+(Y_k^c)=q_k^-(Y_k^c)\) almost surely, though it need not imply equality on tokens that never occur.

**[Proved: linear drift]** Let \(R^+\) and \(R^-\) use persistent Bernoulli action policies

\[
q^+(1)=\frac34,\qquad q^-(1)=\frac14,
\]

and let the actual actions be iid Bernoulli\((r)\). Then

\[
\ell_k=(2Y_k-1)\log 3
\]

and the strong law gives

\[
\frac{\Lambda_n^a}{n}\longrightarrow (2r-1)\log3.
\]

In particular the slopes under \(R^+\) and \(R^-\) are respectively

\[
D\!\left(\tfrac34\middle\Vert\tfrac14\right)
=\frac12\log3,
\qquad
-D\!\left(\tfrac14\middle\Vert\tfrac34\right)
=-\frac12\log3.
\]

**[Proved: square-root-scale oscillation]** In the same example, take \(r=1/2\), corresponding to a third actual policy. Then \(\Lambda_n^a/\log3\) is a simple symmetric random walk. The iid law of the iterated logarithm states that for iid centered variables of variance \(\sigma^2\),

\[
\limsup_{n\to\infty}
\frac{\sum_{k\le n}X_k}{\sqrt{2\sigma^2n\log\log n}}=1,
\qquad
\liminf=-1
\quad\text{a.s.}
\]

Its hypotheses hold here with \(\sigma^2=(\log3)^2\). Hence

\[
\limsup\frac{\Lambda_n^a}{\sqrt{2n\log\log n}}=\log3,
\qquad
\liminf\frac{\Lambda_n^a}{\sqrt{2n\log\log n}}=-\log3.
\]

The curve changes sign infinitely often almost surely. Thus “\(\sqrt n\) fluctuations” are possible, although the sharp almost-sure envelope contains the \(\sqrt{\log\log n}\) factor.

**[Proved: stationary finite-state \(\log n\) growth]** Let the observation alphabet contain \(y,\bar y\), and let the \(y\)-labeled transition matrix on \(S=\{1,2,3\}\) be

\[
Q_y=
\begin{pmatrix}
\rho&b&0\\
0&\rho&0\\
0&0&1
\end{pmatrix},
\qquad
0<\rho<1,\quad 0<b\le1-\rho.
\]

Assign each row’s remaining probability to \(\bar y\), so this defines a valid finite-state world. Let the actual world start in state \(3\), where \(y\) is emitted forever. Let \(R^+\) start at \(\delta_1\) and \(R^-\) at \(\delta_2\). Since

\[
Q_y^n=
\begin{pmatrix}
\rho^n&nb\rho^{n-1}&0\\
0&\rho^n&0\\
0&0&1
\end{pmatrix},
\]

the two likelihoods of \(y^n\) are

\[
L_n^+=\rho^n+nb\rho^{n-1},\qquad L_n^-=\rho^n.
\]

Therefore

\[
\Lambda_n^x
=\log\frac{L_n^+}{L_n^-}
=\log\left(1+\frac{nb}{\rho}\right)
=\log n+O(1).
\]

Thus logarithmic divergence occurs even in a stationary finite-state model; here it is caused by boundary-supported initial states and a Jordan-block polynomial prefactor.

**[Proved: arbitrary polynomial sublinear drift with clocked policies]** Suppose two policies may depend on public round number and set

\[
q_n^-(1)=\frac12,\qquad
q_n^+(1)=\frac12+\delta_n,
\qquad
\delta_n=a n^{-(1-\alpha)/2},
\]

where \(0<\alpha<1\) and \(0<a<1/4\). Under \(P^*=R^-\),

\[
D(q_n^-\Vert q_n^+)
=-\frac12\log(1-4\delta_n^2)
=2\delta_n^2+O(\delta_n^4).
\]

Hence

\[
I_n^-\sim \frac{2a^2}{\alpha}n^\alpha,
\qquad
\Lambda_n^a\sim-\frac{2a^2}{\alpha}n^\alpha
\quad\text{a.s.}
\]

Taking \(\alpha=1/2\) produces one-sided \(\sqrt n\) divergence. Taking \(\delta_n=a/\sqrt n\) produces

\[
\Lambda_n^a\sim-2a^2\log n.
\]

The signs reverse under \(R^+\). This construction uses a clocked policy coordinate; the preceding Jordan-block example shows that logarithmic behavior does not require such a clock.

### 1.5 Sign changes and non-monotonicity

**[Proved]** If \(\Lambda_n^c/n\to\gamma\ne0\), then the running total has the sign of \(\gamma\) eventually; infinitely many late sign changes are impossible.

**[Proved]** If \(\Lambda_n^c\to L\ne0\), its sign is also eventually fixed. Infinite sign changes remain possible when the limit is zero or when the curve does not converge.

**[Proved: non-monotone expected trajectory]** In the Bernoulli example above, let \(P^*(Y_k=1)=0.9\) on odd \(k\) and \(0.1\) on even \(k\), independently. Then

\[
E_{P^*}\ell_k=
\begin{cases}
0.8\log3,&k\text{ odd},\\
-0.8\log3,&k\text{ even}.
\end{cases}
\]

Thus \(E\Lambda_n^a\) zigzags. Blocks of \(0.9\) and \(0.1\) whose lengths successively dominate all preceding blocks give a bounded-increment example in which \(\Lambda_n^a/n\) has no limit.

**[Proved: non-monotonicity in channel noise]** Consider one binary state observed through a binary symmetric channel of noise \(q\). Let the readers’ probabilities for state \(1\) be

\[
u_+=0.6,\qquad u_-=0.1,
\]

and let the actual probability be \(u_*=0.33\). Put

\[
p_i(q)=q+(1-2q)u_i,\qquad i\in\{+,-,*\}.
\]

The expected one-token contribution is

\[
F(q)=p_*(q)\log\frac{p_+(q)}{p_-(q)}
 +(1-p_*(q))\log\frac{1-p_+(q)}{1-p_-(q)}.
\]

At \(q=0\),

\[
F(0)=0.33\log6+0.67\log\frac49>0.
\]

Writing \(t=1-2q\), a Taylor expansion at \(q=1/2\) gives

\[
F(q)=-0.04t^2+O(t^3)<0
\]

for sufficiently small positive \(t\), while \(F(1/2)=0\). Therefore \(F\) is not monotone in \(q\). If the world moves after this observation to an absorbing state with reader-independent emissions, \(F(q)\) is the expected final plateau. Hence non-monotone plateau totals in \(q\) are possible but not universal.

### 1.6 Expected and almost-sure behavior need not agree

**[Proved]** Almost-sure convergence does not imply convergence of expectations without uniform integrability.

For an explicit sequential construction, at stage \(n\) generate a reader-neutral gate \(G_n\) with \(P^*(G_n=1)=2^{-n}\). If \(G_n=1\), make two successive channel tokens deterministically equal to \(1\), with likelihood ratios \(e^{b_n}\) and \(e^{-b_n}\); if \(G_n=0\), use ratio \(1\). These ratios can be realized by choosing, on the first token,

\[
q^+(1)=\frac12,\qquad q^-(1)=\frac12e^{-b_n},
\]

and swapping them on the second. With \(b_n=4^n\), Borel–Cantelli applies because \(\sum_n2^{-n}<\infty\), so only finitely many spikes occur almost surely and \(\Lambda_n^c\to0\). At the intermediate token, however,

\[
E\Lambda^c=2^{-n}b_n=2^n,
\]

whereas after the cancelling token its expectation is zero.

### 1.7 Exhaustive qualitative conclusion

**[Proved]** Without positivity, mixing, stationarity, or a court-correct \(P^*\), all of the following occur in valid sequential Bayesian models:

- finite limits of either sign;
- exactly zero curves;
- linear drift of either sign;
- one-sided \(n^\alpha\) or \(\log n\) divergence;
- centered random-walk fluctuations and infinitely many sign changes;
- non-monotone expected paths;
- failure of \(\Lambda_n/n\) to converge;
- disagreement between almost-sure and expected limiting behavior.

**[Proved]** If \(|\ell_k|\le C\), superlinear growth is impossible because \(|\Lambda_n^c|\le Cn\). Without a lower bound on court probabilities, a single token can create an infinite jump, and unbounded log ratios can also produce superlinear finite growth.

## 2. Limits, slopes, rates, and fluctuations

### 2.1 The finite-limit functional

**[Proved]** Whenever the series converges,

\[
\boxed{
\Lambda_\infty^c
=
\sum_{k=1}^{\infty}
\log\frac{q_k^+(Y_k^c)}{q_k^-(Y_k^c)}
=
\log\prod_{k=1}^{\infty}
\frac{q_k^+(Y_k^c)}{q_k^-(Y_k^c)}
}
\]

is the exact functional of the perturbation, the nonlinear reader dynamics, the selected channel, and the realized stream. In general it cannot be reduced to a norm of the initial perturbation or to a single mixing coefficient.

For the entire interleaved record,

\[
\Lambda_n^{a+x}:=\Lambda_n^a+\Lambda_n^x
=\log\frac{dQ_n^+}{dQ_n^-}(h_n),
\]

where \(Q_n^\pm\) are the two readers’ finite-history laws. Each channel factor separately is the likelihood ratio of a hybrid sequential law that substitutes \(R^+\)’s conditional probabilities only in that channel.

### 2.2 Perturbative form

Let \(u\) denote the complete reader state, let

\[
u_{k+1}=\Phi_{Y_k}(u_k),
\]

and let \(g_y^c(u)\) be the relevant channel probability. Start \(R^-\) at \(u\) and \(R^+\) at \(u+\epsilon v\).

**[Proved under the stated regularity assumptions]** Suppose \(\Phi_y\) and \(g_y^c\) are differentiable, probabilities stay bounded away from zero, and the differentiated series is absolutely dominated. Define the tangent recursion

\[
J_0v=v,\qquad
J_{k+1}v=D\Phi_{Y_k}(u_k)J_kv.
\]

Then

\[
\left.\frac{d}{d\epsilon}\Lambda_\infty^c
\right|_{\epsilon=0}
=
\sum_{k\ge1}
D\log g_{Y_k^c}^c(u_k)\,J_kv.
\]

Thus the leading plateau is an accumulated directional score, with the perturbation propagated through the filter cocycle.

**[Proved under standard dominated differentiability and \(L^2\) interchange]** Under the unperturbed law,

\[
E[\Lambda_\infty^c(\epsilon)]
=-\frac{\epsilon^2}{2}\,\mathcal I_c(v)+o(\epsilon^2),
\]

where \(\mathcal I_c(v)\) is the Fisher information of the channel-hybrid path law in direction \(v\). The first-order score has mean zero. Under the perturbed law the sign of the quadratic term reverses.

### 2.3 Approach rate

**[Proved]** Suppose along the common stream the two reader states satisfy

\[
\|u_k^+-u_k^-\|\le C\rho^k,\qquad 0<\rho<1,
\]

the channel predictor is Lipschitz, and all predictive probabilities are at least \(\varepsilon>0\). Then

\[
|\ell_k|\le C'\rho^k
\]

and consequently

\[
|\Lambda_\infty^c-\Lambda_n^c|
\le \frac{C'\rho^{n+1}}{1-\rho}.
\]

This is the usual exponential plateau.

**[Proved]** No rate follows from existence of the limit alone. For independent Bernoulli predictions

\[
q_n^-(1)=\frac12,\qquad
q_n^+(1)=\frac12+\delta_n,
\]

with

\[
\delta_n=\frac{a}{\sqrt n\,\log(n+1)},
\]

one has \(\sum_n\delta_n^2<\infty\), so the likelihood ratio converges to a finite limit. But

\[
\sum_{k>n}\delta_k^2\asymp\frac1{\log n}.
\]

Under \(R^-\), the centered tail therefore has standard deviation of order

\[
\frac1{\sqrt{\log n}},
\]

while its mean tail is of order \(1/\log n\). Choosing a square-summable sequence with a still slower tail gives arbitrarily slow convergence.

### 2.4 Variance of a plateau

**[Proved]** If \(\Lambda_\infty^c\in L^2(P^*)\) and the covariance series is absolutely summable, then

\[
\operatorname{Var}_{P^*}(\Lambda_\infty^c)
=
\sum_{j,k\ge1}\operatorname{Cov}_{P^*}(\ell_j,\ell_k).
\]

More generally,

\[
\operatorname{Var}(\Lambda_\infty^c)
=\lim_{n\to\infty}\operatorname{Var}(\Lambda_n^c).
\]

Thus plateau variance depends both on one-step channel sensitivity and on temporal correlations created by filtering and control. It is not determined solely by the terminal separation of the readers.

### 2.5 Linear slope and central fluctuations

**[Proved]** If the augmented process determining \((p_k,q_k^+,q_k^-)\) is stationary and ergodic and \(\ell_1\in L^1\), Birkhoff’s ergodic theorem gives

\[
\frac{\Lambda_n^c}{n}
\longrightarrow
\gamma_c:=E_{P^*}\ell_1
=
E_{P^*}\!\left[
D(p_1\Vert q_1^-)-D(p_1\Vert q_1^+)
\right]
\quad\text{a.s. and in }L^1.
\]

Birkhoff’s theorem says that time averages of an integrable observable of a stationary ergodic system converge almost surely and in \(L^1\) to its expectation; stationarity, ergodicity, and integrability are exactly the hypotheses used here.

In particular,

\[
\gamma_c=
\begin{cases}
E_{R^+}D(q_1^+\Vert q_1^-),&P^*=R^+,\\[1mm]
-E_{R^-}D(q_1^-\Vert q_1^+),&P^*=R^-.
\end{cases}
\]

**[Proved]** For arbitrary nonstationary \(P^*\), if log ratios are uniformly bounded and

\[
\frac1n\sum_{k\le n}m_k\to\gamma
\quad\text{a.s.},
\]

then the martingale strong law yields

\[
\frac{\Lambda_n^c}{n}\to\gamma
\quad\text{a.s.}
\]

**[Proved under the stated finite-Markov hypothesis]** If the increments are an additive functional of an irreducible, aperiodic finite-state Markov chain, the Markov-chain central limit theorem gives

\[
\frac{\Lambda_n^c-n\gamma_c}{\sqrt n}
\Rightarrow N(0,\sigma_c^2),
\]

where

\[
\sigma_c^2
=
\operatorname{Var}(\ell_0)
+2\sum_{j\ge1}\operatorname{Cov}(\ell_0,\ell_j).
\]

The theorem applies because finite irreducible aperiodic chains are geometrically mixing and every bounded observable has the required moments and summable correlations. The variance may be zero for a coboundary.

## 3. Conditions separating the regimes

### 3.1 Support and boundary effects

**[Proved]** If at some reached history

\[
p_k(y)>0,\qquad q_k^-(y)=0<q_k^+(y),
\]

then observing \(y\) sends \(\Lambda^c\) immediately to \(+\infty\). Interchanging \(+\) and \(-\) gives \(-\infty\).

Interior predictive probabilities rule out such finite-time explosions. Interior initial beliefs alone do not suffice unless the kernels and policies preserve positive predictive support.

**[Proved]** Boundary initialization is neither necessary nor sufficient for divergence:

- It can cause immediate infinity by excluding a possible token.
- Two boundary vertices with persistently different policies give linear drift.
- A boundary state observationally equivalent to the alternative can give \(\Lambda^c\equiv0\).
- The finite-state Jordan example gives only logarithmic divergence.

### 3.2 Interior priors over a common finite latent model

Let \(U\) be a finite latent initial configuration, encompassing world state and, when appropriate, a fixed policy identity. Suppose both readers use the same conditional laws \(P_u\) after \(t_0\), differing only in priors \(w^\pm\), with

\[
w^\pm(u)>0\quad\text{for every }u.
\]

**[Proved]** Both channel curves converge to finite real limits under \(R^-\), and under every \(P^*\ll R^-\).

Indeed, for every finite history,

\[
\frac{dQ_n^+}{dQ_n^-}
=
\frac{\sum_u w^+(u)L_u(h_n)}
     {\sum_u w^-(u)L_u(h_n)}
\]

lies between

\[
m=\min_u\frac{w^+(u)}{w^-(u)}>0,
\qquad
M=\max_u\frac{w^+(u)}{w^-(u)}<\infty.
\]

But

\[
\frac{dQ_n^+}{dQ_n^-}
=e^{\Lambda_n^a}e^{\Lambda_n^x}.
\]

Under \(R^-\), each factor is a nonnegative martingale and therefore has a finite limit. Since their product is bounded below by \(m\), neither limit can be zero. Hence both logarithms converge to finite real values. Absolute continuity transfers this almost-sure conclusion to \(P^*\ll R^-\).

In particular, any actual component \(P_u\) is dominated by \(R^-\), because

\[
P_u(A)\le\frac{1}{w^-(u)}R^-(A).
\]

**[Proved]** This theorem depends on “interior” meaning positive prior mass on every common latent component. An arbitrary interior point of a policy-coordinate simplex need not have this consequence if that coordinate is a physical randomized controller rather than Bayesian uncertainty about a fixed latent identity.

### 3.3 Mixing and filter stability

**[Sketched]** Uniform positivity of the finite-state update matrices is a standard sufficient condition for exponential forgetting. For a positive matrix \(K\), Birkhoff’s projective-contraction theorem says that its normalized action contracts Hilbert’s projective metric by at most

\[
\tanh\!\left(\frac{\Delta(K)}4\right),
\]

where \(\Delta(K)\) is its projective diameter. If all controlled observation matrices \(T^{(a,x)}\) have entries whose positive ratios are uniformly bounded, then their projective diameters are uniformly bounded and the contraction factor is uniformly below \(1\). Equivalence of Hilbert and ordinary norms on a compact interior subset of the simplex then gives

\[
\|\eta_k^+-\eta_k^-\|\le C\rho^k.
\]

If channel predictors are Lipschitz and bounded away from zero, the exponential plateau bound of Section 2 follows.

This argument verifies the theorem’s hypotheses only when every matrix encountered is uniformly positive. Reducibility, deterministic observations, controlled avoidance of mixing actions, or zeros can destroy contraction.

### 3.4 Persistent informativeness

**[Proved]** Under \(R^-\), uniform positivity, and

\[
\lim_{n\to\infty}\frac1n
\sum_{k\le n}D(q_k^-\Vert q_k^+)=d_->0,
\]

one has

\[
\frac{\Lambda_n^c}{n}\to-d_-.
\]

Under \(R^+\), the analogous condition gives positive slope

\[
d_+=\lim_{n\to\infty}\frac1n
\sum_{k\le n}D(q_k^+\Vert q_k^-)>0.
\]

Thus persistent channel identifiability produces linear drift. Vanishing one-step separation can instead give:

\[
\begin{array}{c|c}
\text{cumulative KL information} & \text{court-correct behavior}\\
\hline
\sum_k D_k<\infty & \text{finite plateau}\\
\sum_{k\le n}D_k\asymp\log n & \pm\log n\\
\sum_{k\le n}D_k\asymp n^\alpha & \pm n^\alpha\\
\sum_{k\le n}D_k\asymp n & \text{linear drift}.
\end{array}
\]

### 3.5 Channel informativeness and causal separation

**[Proved]** A channel is silent exactly when the perturbation never changes its predictive distribution on reached histories.

In particular, if the perturbation affects only \(\eta\), the action predictor depends only on \(z\), and the update dynamics never transfer the \(\eta\)-difference into \(z\), then

\[
\Lambda_n^a\equiv0.
\]

The corresponding statement holds for a pure \(z\)-perturbation and the observation channel. Posterior correlation, belief-responsive policies, or coupled state updates can transfer a perturbation across courts and invalidate this zero conclusion.

### 3.6 Injection versus penetration

**[Proved]** If \(P^*=R^+\), every channel has nonnegative expected increments and the positive information criterion above applies. If \(P^*=R^-\), the signs reverse.

**[Proved]** A finite forced-action window contributes only finitely many extra increments. If after that window the actual conditional law really equals \(q_k^-\), the later expected increments are \(-D(q_k^-\Vert q_k^+)\), so the finite intervention can shift the curve but cannot change whether the tail information is finite, sublinear divergent, or linear. If the readers misinterpret the forced tokens and the post-window actual conditional law is not \(q_k^-\), no such sign conclusion follows; the general difference-of-KLs formula must be used.

## 4. Universal invariants versus family-specific facts

**[Proved: universal]**

1. The drift identity

   \[
   E[\ell_k\mid\mathcal F_{k-1}]
   =D(p_k\Vert q_k^-)-D(p_k\Vert q_k^+)
   \]

   is model-free.

2. Under \(R^-\), \(e^{\Lambda_n^c}\) is a likelihood-ratio martingale; under \(R^+\), \(e^{-\Lambda_n^c}\) is one.

3. Consequently, the court-correct signs of expected drift and the court-correct one-sided alternatives “finite or favored infinity” are universal.

4. The full likelihood ratio factorizes as

   \[
   e^{\Lambda_n^a+\Lambda_n^x}.
   \]

5. Zeros in predictive support can create immediate infinite evidence.

6. Exact equality of channel predictions gives an exactly zero curve.

7. Under uniform positivity, cumulative conditional KL information separates finite, sublinear, and linear regimes.

**[Proved: not universal]** None of the following follows merely from “two exact Bayesian readers on a shared stream”:

- exponential approach to a plateau;
- a fixed sign for realized plateaus;
- monotonicity of individual paths;
- monotonicity of plateau size in observation noise;
- equality or opposition of action and observation curves;
- linear drift whenever the initial states differ;
- convergence of expectations from almost-sure convergence;
- existence of a deterministic asymptotic slope under nonergodic \(P^*\).

**[Proved]** The binary model’s particular dependence on flip rate, consequence parameter, roster policies, and noise \(q\) determines its numerical plateau heights and phase boundaries. The qualitative shapes themselves are not binary-specific: each occurs in more general finite or sequential models above.

## 5. Relation to standard theory

### 5.1 Likelihood-ratio martingales and Doob convergence

**[Proved]** The channel exponentials are ordinary sequential likelihood-ratio martingales under the corresponding baseline court. Doob’s nonnegative-martingale convergence theorem supplies the strongest completely model-free shape restriction: under a court-correct law, a channel curve must converge finitely or diverge toward that court.

### 5.2 Kakutani’s product-measure dichotomy

Kakutani’s theorem states that two product measures

\[
\bigotimes_n\mu_n,\qquad \bigotimes_n\nu_n,
\]

whose coordinate measures are mutually absolutely continuous, are equivalent exactly when the product of their Hellinger affinities is positive; otherwise they are mutually singular. For Bernoulli parameters uniformly separated from \(0\) and \(1\), this is equivalent to

\[
\sum_n(p_n-r_n)^2<\infty.
\]

**[Proved]** In the clocked Bernoulli construction, all coordinate probabilities lie in a common compact subinterval of \((0,1)\), so Kakutani’s hypotheses hold. Therefore

\[
\sum_n\delta_n^2<\infty
\]

is exactly the equivalence/finite-log-likelihood regime, while divergence of this sum is the singular regime. The KL calculation refines the theorem by giving the divergence rate.

### 5.3 Blackwell–Dubins merging

The Blackwell–Dubins theorem states that if a probability law \(P\) is absolutely continuous with respect to \(Q\) on the infinite observation \(\sigma\)-field, then their conditional distributions of the future merge in total variation \(P\)-almost surely.

**[Proved application]** Finite interior priors over the same latent components give mutual absolute continuity, so the two readers’ future predictions merge under either law.

**[Proved limitation]** Blackwell–Dubins supplies no general rate. Moreover, mere one-step convergence \(q_k^+-q_k^-\to0\) does not distinguish a finite plateau from \(\log n\), \(\sqrt n\), or other sublinear divergence; cumulative KL or Hellinger information is required.

### 5.4 Ergodic and martingale limit theory

**[Proved]** Birkhoff’s ergodic theorem identifies the linear slope when the augmented prediction process is stationary ergodic and the log ratio is integrable.

**[Proved]** Martingale strong laws identify the slope more generally from the predictable drift \(A_n\), provided martingale fluctuations are negligible relative to the chosen normalization.

**[Proved under the stated Markov hypotheses]** Finite-state Markov additive-functional central limit theorems give Gaussian \(\sqrt n\) fluctuations around a nonzero linear drift, with the Green–Kubo covariance sum as variance.

### 5.5 Nonlinear-filter contraction

**[Sketched]** Positive-matrix projective contraction, coupling, and Doeblin minorization are the appropriate mechanisms for exponential forgetting of \(\eta\). They yield finite observation-court plateaus when the log predictor is Lipschitz. They do not control a persistent identity coordinate unless that coordinate itself mixes or becomes statistically identified.

No additional conjecture is needed for the general classification: beyond the stated domination, positivity, contraction, and ergodicity hypotheses, the explicit counterexamples show that a sharper model-free taxonomy is impossible.
