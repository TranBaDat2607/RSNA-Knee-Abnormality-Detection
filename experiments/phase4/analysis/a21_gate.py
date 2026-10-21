"""A21 guard (pre-registered in docs/phase4_notes.md, 2026-10-21 16:40 UTC): read once.

usage: python a21_gate.py <gold dir>

A19 = A16 with the 384 slot = rank mean(nartaa 384, cohort), both from the A19 gold kernel (as in a19_gate.py).
A21 = 2/3 x A19 + 1/3 x rank(1/2 rank(s6fall) + 1/2 rank(S6T)).  Guard: A21 >= A16 - 0.003.
"""
import sys
import numpy as np
sys.argv = sys.argv[:2]
import final_evidence as F  # noqa: E402

K = 'rsna-knee-a19-gold58'
n384, cohort = F.leg(K, 'n384_sub.csv'), F.leg(K, 'cohort_sub.csv')


def a16(n384_like):
    raptor = 0.65 * F.rk(n384_like) + 0.35 * F.rk(F.n224)
    return F.rk((1 - F.wr) * F.rk(F.w * F.rk(raptor) + (1 - F.w) * F.rk(F.r6f)) + F.wr * F.rk(F.reader))


A16 = a16(n384)
A19 = a16(0.5 * F.rk(n384) + 0.5 * F.rk(cohort))
s6t = F.leg('rsna-knee-s6t-gold58', 'gold_s6t.csv')
A21 = (2 / 3) * A19 + (1 / 3) * F.rk(0.5 * F.rk(F.s6f) + 0.5 * F.rk(s6t))
d = F.macro(A21) - F.macro(A16)
print()
print('A16 %.4f  A19 %.4f  A21 %.4f  A21 - A16 %+.4f CI %s  A21 - A19 %+.4f' % (
    F.macro(A16), F.macro(A19), F.macro(A21), d, np.round(F.boot(A21, A16), 4), F.macro(A21) - F.macro(A19)))
print('A21 GUARD', 'PASS' if d >= -0.003 else 'FAIL')
