# Schedule A/B: linear vs cubic kappa

Held-out evaluation of two editors trained identically except for the ② schedule --
`kappa_t = t` against `kappa_t = t^3`. Edit Flows states the cubic one in its
experiments (p9) and ships the linear one in its Figure 13 code, so which of the two
our choice deviates from depends on which half of the paper is read.

`schedlin-*` is the linear arm (sky job 76), `schedcub-*` the cubic (job 83). Both
trained 20,000 steps on one A100, so the pair carries no hardware difference.

**The training losses of the two arms are NOT comparable**: the loss weights each
supervised edit by `kappa_dot / (1 - kappa)`, which differs between the schedules at
every t -- 1.333 against 0.190 at t=0.25, a factor of seven. A lower training loss
under cubic is a different objective, not a better model. That is why both arms get a
held-out generation eval, and only these files answer the question.

Scale for reading a difference: two runs of the SAME configuration (`sched-linear` and
the reported `faithful-appendixa`, both linear) differ by 0.0343 in Ty1 MIP agreement.
A schedule effect smaller than that is not distinguishable from run-to-run variation.
