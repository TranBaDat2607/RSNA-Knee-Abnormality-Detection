"""A20 guard (pre-registered in docs/phase4_notes.md, 2026-10-21 16:00 UTC): read once.

usage: python a20_gate.py <gold dir>

A20 = 2/3 x A16 + 1/3 x rank(1/2 rank(s6fall) + 1/2 rank(S6T)); guard A20 >= A16 - 0.003 (regression guard only).
A17 (= 2/3 A16 + 1/3 rank(s6fall)) is printed for reference. Legs and A16 exactly as in final_evidence.py.
"""
import os, sys
import numpy as np
sys.argv = sys.argv[:2]
import final_evidence as F  # noqa: E402  (prints its own A12/A16/A17 lines)

s6t = F.leg('rsna-knee-s6t-gold58', 'gold_s6t.csv')
slot = F.rk(0.5 * F.rk(F.s6f) + 0.5 * F.rk(s6t))
A20 = (2 / 3) * F.A16 + (1 / 3) * slot
A17 = F.C['A17']
print()
print('S6T %.4f  s6fall %.4f  rho(S6T, s6fall) %.3f  rho(S6T, raptor) %.3f  rho(S6T, R6 flip) %.3f' % (
    F.macro(s6t), F.macro(F.s6f), F.rho(s6t, F.s6f), F.rho(s6t, F.raptor), F.rho(s6t, F.r6f)))
d = F.macro(A20) - F.macro(F.A16)
print('A20 %.4f  A16 %.4f  A17 %.4f  A20 - A16 %+.4f CI %s  A20 - A17 %+.4f CI %s' % (
    F.macro(A20), F.macro(F.A16), F.macro(A17), d, np.round(F.boot(A20, F.A16), 4), F.macro(A20) - F.macro(A17),
    np.round(F.boot(A20, A17), 4)))
print('A20 GUARD', 'PASS' if d >= -0.003 else 'FAIL')
