"""A24/A25/A26 guard (pre-registered in docs/phase4_notes.md, 2026-10-22): read once.

usage: python a24_gate.py <gold dir>

Variant = (1 - w) x A16-with-cohort (R6 flip, as A22) + w x S6 slot; guard: each >= A16 - 0.003.
"""
import sys
import numpy as np
sys.argv = sys.argv[:2]
import final_evidence as F  # noqa: E402
from a23_gate import a16, n384, cohort, slot  # noqa: E402

A16 = a16(n384, F.r6f)
base = a16(0.5 * F.rk(n384) + 0.5 * F.rk(cohort), F.r6f)
print()
for tag, w in (('A22', 1 / 3), ('A24', 1 / 2), ('A25', 1 / 4), ('A26', 2 / 5)):
    v = (1 - w) * base + w * slot
    d = F.macro(v) - F.macro(A16)
    print('%s w %.3f  gold %.4f  - A16 %+.4f CI %s  rho(., A16) %.3f  GUARD %s' % (
        tag, w, F.macro(v), d, np.round(F.boot(v, A16), 4), F.rho(v, A16), 'PASS' if d >= -0.003 else 'FAIL'))
