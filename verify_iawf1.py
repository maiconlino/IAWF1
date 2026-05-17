"""
Verification suite for the canonical IAWF1 implementation.

Goals
-----
1. Reproduce Sections 5.1 (GB), 5.2 (MLP), 5.3 (SVM big data) within rounding.
2. Recompute Table 2 under algorithm (A) — the canonical version — and
   show how the numbers change compared to what is currently in the paper.
3. Edge cases: multiclass, zero-size class, perfect classifier, all-wrong.
4. Confirm the metric stays in [0, 1].
"""

import sys
import math
import numpy as np

sys.path.insert(0, "/home/claude/work")
from iawf1_canonical import IAWF1, per_class_f1, class_sizes_from_cm

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"


def check(name, got, expected, tol=1e-3):
    ok = abs(got - expected) <= tol
    color = GREEN if ok else RED
    flag = "OK  " if ok else "FAIL"
    print(f"  {color}[{flag}]{RESET} {name}: got={got:.5f}, expected≈{expected:.4f}, diff={abs(got-expected):.5f}")
    return ok


# =============================================================================
# 1. Reproduce paper Sections 5.1, 5.2, 5.3
# =============================================================================
print("=" * 78)
print("REPRODUCING PAPER SECTIONS 5.1, 5.2, 5.3 UNDER CANONICAL (A)")
print("=" * 78)

# --- Section 5.1: Gradient Boosting on SINAN-TB Amazonas ----------------------
print("\n[5.1] GB Amazonas — paper reports F1_0=0.99238, F1_1=0.832, IAWF1=0.8588")
# Reconstructed sklearn CM where rows=true, cols=predicted, classes ordered [neg=Cured, pos=Death]:
#   true Cured had 6840 predicted Cured  + 71 predicted Death  → total 6911
#   true Death had 34 predicted Cured  + 260 predicted Death  → total 294
cm_51 = np.array([[6840,   71],
                  [  34,  260]])
n = class_sizes_from_cm(cm_51)
f1 = per_class_f1(cm_51)
print(f"  Class sizes from CM (rows): {n}")
print(f"  Per-class F1: {f1}")
m = IAWF1()
s, w = m.score(cm_51)
print(f"  Weights used: {w}")
print(f"  IAWF1 = {s:.5f}")
check("IAWF1 ≈ 0.8588", s, 0.8588)


# --- Section 5.2: MLP on SINAN-TB Amazonas -----------------------------------
print("\n[5.2] MLP Amazonas — paper reports F1_0≈0.9903, F1_1≈0.7990, IAWF1=0.8338")
cm_52 = np.array([[6808,   67],
                  [  66,  264]])
n = class_sizes_from_cm(cm_52)
f1 = per_class_f1(cm_52)
print(f"  Class sizes from CM: {n}")
print(f"  Per-class F1: {f1}")
s, w = IAWF1().score(cm_52)
print(f"  IAWF1 = {s:.5f}")
check("IAWF1 ≈ 0.8338", s, 0.8338)


# --- Section 5.3: SVM on SINAN-TB Brasil -------------------------------------
print("\n[5.3] SVM Brasil — paper reports F1_0=0.9553, F1_1=0.4525, IAWF1=0.5850")
# Paper's CM was [[842762, 75842], [3080, 32749]] with cols=true (transposed)
# Translating to sklearn convention: rows=true, cols=pred → transpose:
#   true Cured(0) → 842762 predicted Cured + 3080 predicted Death  → 845842
#   true Death(1) → 75842 predicted Cured + 32749 predicted Death  → 108591
cm_53 = np.array([[842762,   3080],
                  [ 75842,  32749]])
n = class_sizes_from_cm(cm_53)
f1 = per_class_f1(cm_53)
print(f"  Class sizes: {n}  (paper: 845842 and 108591)")
print(f"  F1: {f1}")
s, w = IAWF1().score(cm_53)
print(f"  IAWF1 = {s:.5f}")
check("IAWF1 ≈ 0.5850", s, 0.5850)

# Custom weights example from 5.3 (w_0=0.2 cured, w_1=0.8 death)
print("\n  With custom weights [0.2, 0.8]:")
s_cust, w_cust = IAWF1(custom_weights=[0.2, 0.8]).score(cm_53)
print(f"  IAWF1 = {s_cust:.5f}, weights = {w_cust}")
check("IAWF1 ≈ 0.5531 (custom)", s_cust, 0.5531)


# =============================================================================
# 2. Recompute Table 2 under canonical (A)
# =============================================================================
print("\n" + "=" * 78)
print("TABLE 2 RECOMPUTED UNDER CANONICAL (A)  vs.  PAPER VALUES")
print("=" * 78)

# Table 2 confusion matrices are in sklearn convention [[TN,FP],[FN,TP]]
table2 = [
    ("Extreme (error in both)",          [[1,     99999], [1,     999999]], 0.0013),
    ("Medium  (error in both)",          [[100,   99999], [10000, 999999]], 0.0845),
    ("Small   (error in both)",          [[1000,  99999], [10000, 999999]], 0.1024),
    ("Extreme (minority error)",         [[10,        0], [40,    500000]], 0.3399),
    ("Balanced (perfect)",               [[500000,    0], [0,     500000]], 1.0000),
]

print(f"\n{'Scenario':<32s} | {'IAWF1 (A) canonical':>20s} | {'IAWF1 paper':>12s}")
print("-" * 78)
for name, cm_list, paper_val in table2:
    cm = np.array(cm_list, dtype=float)
    s, _ = IAWF1().score(cm)
    delta = s - paper_val
    color = GREEN if abs(delta) < 0.05 else YELLOW if abs(delta) < 0.2 else RED
    print(f"{name:<32s} | {color}{s:>20.4f}{RESET} | {paper_val:>12.4f}  (Δ = {delta:+.4f})")


# =============================================================================
# 3. Edge cases
# =============================================================================
print("\n" + "=" * 78)
print("EDGE CASES")
print("=" * 78)

# 3.1 Perfect classifier
print("\n[3.1] Perfect binary classifier")
cm = np.array([[1000, 0], [0, 1000]])
s, _ = IAWF1().score(cm)
print(f"  IAWF1 = {s:.5f}")
check("Perfect → 1.0", s, 1.0)

# 3.2 All-wrong classifier
print("\n[3.2] Inverted classifier")
cm = np.array([[0, 1000], [1000, 0]])
s, _ = IAWF1().score(cm)
print(f"  IAWF1 = {s:.5f}")
check("All wrong → 0.0", s, 0.0)

# 3.3 Multiclass (3 classes, unbalanced)
print("\n[3.3] Multiclass 3-class CM (rows=true)")
cm = np.array([
    [9000, 100,   50],   # class 0: 9150 samples, mostly correct
    [ 200, 300,   10],   # class 1: 510 samples
    [  20,  10,  100],   # class 2: 130 samples
])
n = class_sizes_from_cm(cm)
f1 = per_class_f1(cm)
s, w = IAWF1().score(cm)
print(f"  Class sizes: {n}")
print(f"  Per-class F1: {f1}")
print(f"  Weights used: {w}")
print(f"  IAWF1 = {s:.5f}")
print(f"  Expected behavior: weights should be largest for the smallest class.")
print(f"  Confirmation: argmax(w) = class {np.argmax(w)}, argmin(n) = class {np.argmin(n)}")
assert np.argmax(w) == np.argmin(n), "Largest weight should go to smallest class"
print(f"  {GREEN}[OK]{RESET} largest weight ↔ smallest class")

# 3.4 Class with zero samples
print("\n[3.4] CM with an empty class (class 1 missing)")
cm = np.array([
    [100,   0,  5],
    [  0,   0,  0],
    [  3,   0, 50],
])
n = class_sizes_from_cm(cm)
print(f"  Class sizes: {n}")
s, w = IAWF1().score(cm)
print(f"  Weights: {w} (class 1 should get weight 0)")
print(f"  IAWF1 = {s:.5f}")
assert w[1] == 0.0
print(f"  {GREEN}[OK]{RESET} empty class properly excluded")

# 3.5 Bounded in [0,1]?
print("\n[3.5] Random stress: 200 random binary CMs, all in [0,1]?")
rng = np.random.default_rng(42)
all_ok = True
for _ in range(200):
    cm = rng.integers(0, 10000, size=(2, 2)).astype(float)
    if cm.sum() == 0: continue
    s, _ = IAWF1().score(cm)
    if not (0.0 <= s <= 1.0 + 1e-10):
        all_ok = False
        print(f"  {RED}FAIL{RESET}: IAWF1 = {s} for CM = {cm}")
print(f"  {GREEN if all_ok else RED}[{'OK' if all_ok else 'FAIL'}]{RESET} IAWF1 ∈ [0,1] always")

# 3.6 Sensitivity to weighting alpha (1/n^alpha family) — preview for the rebuttal
print("\n[3.6] 1/n^alpha family — preview for the methodological justification")
# Same CM, alpha varying. alpha=0 → macro. alpha=1 → inverse-frequency.
cm = np.array([[990, 10], [50, 950]])  # mild imbalance
f1 = per_class_f1(cm)
n  = class_sizes_from_cm(cm)
print(f"  CM: {cm.tolist()}")
print(f"  Per-class F1: {f1}")
print(f"  Class sizes: {n}")
print()
print(f"  {'alpha':>8s} | {'w_0':>8s} | {'w_1':>8s} | {'IAWF1(α)':>10s}")
print("  " + "-" * 44)
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    w_init = 1.0 / np.power(n, alpha)
    w = w_init / w_init.sum()
    score = float(np.sum(w * f1))
    print(f"  {alpha:>8.2f} | {w[0]:>8.4f} | {w[1]:>8.4f} | {score:>10.5f}")
print("  → α=0.5 (our IAWF1) sits between F1-macro (α=0) and inverse-frequency (α=1)")

print("\n" + "=" * 78)
print(f"{GREEN}Suite finished.{RESET}")
