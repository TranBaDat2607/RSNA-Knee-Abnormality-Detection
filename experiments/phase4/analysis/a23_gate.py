"""A23 guard (pre-registered in docs/phase4_notes.md, 2026-10-22 02:50 UTC): read once.

usage: python a23_gate.py <gold dir>

A23 = A21 (a21_gate.py) with R6 plain (`gold_r6_plain.csv`) in place of R6 flip.  Guard: A23 >= A16 - 0.003.
"""
import sys
import numpy as np
sys.argv = sys.argv[:2]
import final_evidence as F  # noqa: E402

K = 'rsna-knee-a19-gold58'
n384, cohort = F.leg(K, 'n384_sub.csv'), F.leg(K, 'cohort_sub.csv')
r6p = F.leg('rsna-knee-r6-gold58', 'gold_r6_plain.csv')


def a16(n384_like, r6):
    raptor = 0.65 * F.rk(n384_like) + 0.35 * F.rk(F.n224)
    return F.rk((1 - F.wr) * F.rk(F.w * F.rk(raptor) + (1 - F.w) * F.rk(r6)) + F.wr * F.rk(F.reader))


s6t = F.leg('rsna-knee-s6t-gold58', 'gold_s6t.csv')
slot = F.rk(0.5 * F.rk(F.s6f) + 0.5 * F.rk(s6t))
A16 = a16(n384, F.r6f)
A21 = (2 / 3) * a16(0.5 * F.rk(n384) + 0.5 * F.rk(cohort), F.r6f) + (1 / 3) * slot
A23 = (2 / 3) * a16(0.5 * F.rk(n384) + 0.5 * F.rk(cohort), r6p) + (1 / 3) * slot
d = F.macro(A23) - F.macro(A16)
print()
print('A16 %.4f  A21 %.4f  A23 %.4f  A23 - A16 %+.4f CI %s  A23 - A21 %+.4f  rho(A23, A16) %.3f  rho(r6 plain, flip) %.3f' % (
    F.macro(A16), F.macro(A21), F.macro(A23), d, np.round(F.boot(A23, A16), 4), F.macro(A23) - F.macro(A21),
    F.rho(A23, A16), F.rho(r6p, F.r6f)))
print('A23 GUARD', 'PASS' if d >= -0.003 else 'FAIL')
