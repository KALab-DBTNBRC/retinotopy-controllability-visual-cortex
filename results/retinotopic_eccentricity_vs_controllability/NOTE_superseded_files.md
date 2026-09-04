# Superseded result tables in this folder

`gradient_linearity_glasser.csv` and `gradient_linearity_schaefer200.csv` report
the **unadjusted** spline-versus-linear comparison. They are retained only so the
adjusted result can be compared against what preceded it.

The apparent departure from linearity they show is a connectivity-strength
effect: it disappears once weighted node degree enters the model, and the spline
never predicted held-out nodes better than a straight line. **The manuscript
reports `gradient_linearity_adjusted_*.csv`, not these files.**

In the adjusted tables the sign of `excess` carries the meaning, and the
permutation test on it is two-sided. A negative excess means the spline gains
*less* on the real data than on spatially matched surrogates, which is evidence
for linearity rather than against it. Reading a small p-value there as
"significant curvature" inverts the result.
