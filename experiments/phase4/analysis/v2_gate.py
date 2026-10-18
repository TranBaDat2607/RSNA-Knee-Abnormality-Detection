"""V2 gate (pre-registered in docs/phase4_notes.md, 2026-10-17, V2 section): read once, after the V2 gold-58 gate kernel.

usage: python v2_gate.py runtime <v2-gold58-dev output dir>              # rule 4 only: choose the eval stride (timing, no scores)
       python v2_gate.py gate <gold_out dir with the phase-3 gold kernels> <v2 gold-58 gate output dir>

1. V2 gold-58 macro AUC >= 0.925
2. mean Spearman (12 findings) <= 0.90 against each of raptor, R7, s6fall
3. A16 = 3/4 A12 + 1/4 rank(V2) >= A12 - 0.003
4. projected A16 = 4.77 h (D1) + V2 stage (gold-58 wall per study per GPU x 1,322 / 2 x 1.15) <= 8.5 h; smallest stride of 1-4
"""
import json, os, sys

A12_H, N_TEST, MARGIN, CAP = 4.77, 1322, 1.15, 8.5


def runtime(dev_dir):
    info = json.load(open(os.path.join(dev_dir, 'v2_gold_info.json')))
    chosen = None
    for s in sorted(info, key=int):
        r = info[s]
        per = max(p['seconds'] / p['studies'] for p in r['parts'])      # slower GPU worker, wall seconds per study
        stage_h = per * N_TEST / 2 * MARGIN / 3600
        total = A12_H + stage_h
        fits = total <= CAP and r['failed'] == 0
        print('stride %s: %.1f s/study/worker (parts %s) -> V2 stage %.2f h, A16 %.2f h %s | shared trunk %s, models %s' % (
            s, per, [(p['studies'], p['seconds']) for p in r['parts']], stage_h, total, 'FITS' if fits else 'no',
            r['shared_trunk'], r['models']))
        if fits and chosen is None:
            chosen = int(s)
    print('RULE 4 stride:', chosen if chosen is not None else 'none fits -> V2 dropped')
    return chosen


def gate(g, s):
    import numpy as np, pandas as pd
    from scipy.stats import rankdata, spearmanr
    from sklearn.metrics import roc_auc_score
    T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
         "Baker's", 'Contusion', 'Fracture']
    y = pd.read_csv(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_root', 'gold_labels.csv'),
                    dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
    Y = (y.values > 0.5).astype(int)
    leg = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[y.index, T].values
    rk = lambda x: rankdata(x, axis=0) / len(x)
    macro = lambda p: float(np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in range(len(T))]))
    rho = lambda a, b: float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(len(T))]))
    n384 = leg(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_acc.csv'))
    n224 = leg(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_eff.csv'))
    r7 = leg(os.path.join(g, 'rsna-knee-oai-gold58-r7', 'gold_r7_plain.csv'))
    s6f = leg(os.path.join(g, 'rsna-knee-oai-gold58-s6fall', 'gold_s6fall.csv'))
    raptor = 0.65 * rk(n384) + 0.35 * rk(n224)
    a12 = (2 / 3) * (0.5 * rk(raptor) + 0.5 * rk(r7)) + (1 / 3) * rk(s6f)
    print('A12 gold-58 %.4f (phase 3: 0.9406)' % macro(a12))
    for f in sorted(f for f in os.listdir(s) if f.startswith('gold_v2_s')) + ['gold_v2.csv']:
        p = leg(os.path.join(s, f))
        print('%-16s macro %.4f | rho raptor %.3f R7 %.3f s6fall %.3f' % (f, macro(p), rho(p, raptor), rho(p, r7), rho(p, s6f)))
    t = leg(os.path.join(s, 'gold_v2.csv'))
    a16 = 0.75 * a12 + 0.25 * rk(t)
    c = {'1 leg >= 0.925': macro(t) >= 0.925,
         '2 rho <= 0.90 vs raptor, R7, s6fall': max(rho(t, raptor), rho(t, r7), rho(t, s6f)) <= 0.90,
         '3 A16 >= A12 - 0.003': macro(a16) >= macro(a12) - 0.003}
    print('A16 gold-58 %.4f (A12 %.4f, delta %+.4f)' % (macro(a16), macro(a12), macro(a16) - macro(a12)))
    per = pd.DataFrame({'V2': [roc_auc_score(Y[:, j], t[:, j]) for j in range(len(T))],
                        'A12': [roc_auc_score(Y[:, j], a12[:, j]) for j in range(len(T))],
                        'A16': [roc_auc_score(Y[:, j], a16[:, j]) for j in range(len(T))]}, index=T).round(3)
    print(per.to_string())
    for k, v in c.items():
        print('rule %-40s %s' % (k, 'PASS' if v else 'FAIL'))
    print('GATE (rules 1-3)', 'PASS' if all(c.values()) else 'FAIL', '| rule 4 from the dev timing run')


if __name__ == '__main__':
    runtime(sys.argv[2]) if sys.argv[1] == 'runtime' else gate(sys.argv[2], sys.argv[3])
