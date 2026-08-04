# HMC tuning (MetaField)

Target for a 2nd-order leapfrog on this codebase:

| Metric | Aim |
|--------|-----|
| Acceptance | **~0.50–0.70** |
| Typical \|ΔH\| | **≲ 1** |
| CG residual | already ~1e-10 (leave alone) |

Below ~0.35 accept with mean \|ΔH\| ≳ 1.5 → step is too large. Above ~0.90 with tiny moves → step is too small (wasted compute).

---

## Defaults (v1.59.3+)

**Dynamical (fermions on):**

```text
--hmc-step 5e-5
--hmc-leapfrog 300
τ = step × leapfrog ≈ 0.015
```

History on L=4⁴ β=5.5 dynamical:

| step × leapfrog | typical \|ΔH\| | accept |
|-----------------|---------------|--------|
| 0.0002 × 75 | ~2–5 | ~20% |
| 0.0001 × 150 | ~0.5–4 | ~35–45% |
| **5e-5 × 300** | target ≲1 | target ~50–70% |

**Quenched:**

```text
--hmc-step 0.012
--hmc-leapfrog 20
```

---

## Recipe when acceptance is still low

1. Halve the step, double leapfrog (keeps τ roughly fixed):

```bash
python meta_field_distributed.py --diagnostic --continuous \
  --hmc-step 2.5e-5 --hmc-leapfrog 600
```

2. Watch the first ~15 trajectories. Mixed ACCEPT/REJECT and `dH` mostly O(1).

3. If still bad, another half-step.

4. Do **not** chase 95% accept — decorrelation suffers.

Diagnostic runs print `[HMC tune]` around traj 14 if the early window looks unhealthy.

---

## Flags

```text
--hmc-step FLOAT       MD step size ε
--hmc-leapfrog INT     number of leapfrog steps L
# trajectory length τ = ε × L
```

CG is fine at current tols. Large ΔH with tiny residuals ⇒ **integrator / step-size**, not the solver.

`--world-size` defaults to **1**. Multi-rank needs `torchrun` / `RANK`+`WORLD_SIZE`.
