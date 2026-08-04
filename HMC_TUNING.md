# HMC tuning (MetaField)

Target for a 2nd-order leapfrog on this codebase:

| Metric | Aim |
|--------|-----|
| Acceptance | **~0.50–0.70** |
| Typical \|ΔH\| | **≲ 1** |
| CG residual | already ~1e-10 (leave alone) |

Below ~0.35 accept with mean \|ΔH\| ≳ 1.5 → step is too large. Above ~0.90 with tiny moves → step is too small (wasted compute).

---

## Defaults (v1.59+)

**Dynamical (fermions on):**

```text
--hmc-step 0.0001
--hmc-leapfrog 150
τ = step × leapfrog ≈ 0.015
```

(Previously `0.0002 × 75` produced \|ΔH\| ~ 2–5 and ~20% accept.)

**Quenched:**

```text
--hmc-step 0.012
--hmc-leapfrog 20
```

---

## Recipe when acceptance is still low

1. Halve the step, double leapfrog (keeps τ roughly fixed):

```bash
python meta_field_distributed.py --world-size 1 --diagnostic --continuous \
  --hmc-step 5e-5 --hmc-leapfrog 300
```

2. Watch the first ~15 trajectories. You want mixed ACCEPT/REJECT and `dH` mostly O(1).

3. If still bad, halve step again and double leapfrog.

4. Do **not** chase 95% accept — decorrelation suffers.

Diagnostic runs print a one-line `[HMC tune]` suggestion around traj 14 if the early window looks unhealthy.

---

## Flags

```text
--hmc-step FLOAT       MD step size ε
--hmc-leapfrog INT     number of leapfrog steps L
# trajectory length τ = ε × L
```

CG tolerances (`cg_tol_md`, `cg_tol_action`) are fine at current defaults; large ΔH with tiny residuals is an **integrator / step-size** issue, not a solver issue.
