"""Daily public-notebook watch (backlog #3): which public notebooks score at or above our best public LB?

usage: python watch_public.py [our_best=0.957] [--all]

Uses the Kaggle search API (kernel_document.best_public_score), because `kgkit kernels top` prints no score
(UAT issue TranBaDat2607/claude-code-kaggle-grandmaster#14). Prints new candidates (score >= our best, updated in the last 3 days) to audit and fork.
"""
import sys, time
from datetime import datetime, timedelta, timezone
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/../phase2/analysis")
from search import search  # noqa: E402  (experiments/phase2/analysis/search.py)
from kagglesdk.search.types.search_enums import DocumentType, ListSearchContentOrderBy as O  # noqa: E402

best = float(next((a for a in sys.argv[1:] if not a.startswith("--")), 0.957))
seen = {}
# Notebooks attached to the competition count even without "knee" in the title (missed rsna-versia-7, top-1-gold on 10-21).
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
_api = KaggleApi()
_api.authenticate()
comp = set()
for page in range(1, 6):
    ks = _api.kernels_list(competition="rsna-knee-abnormality-detection", sort_by="dateRun", page=page, page_size=100) or []
    comp |= {str(k.ref).lower() for k in ks}
    if len(ks) < 100:
        break
for q in ("rsna knee", "knee", "rsna", "rsna knee 0.957", "rsna knee 0.958", "rsna knee 0.959", "rsna knee 0.96", "knee gold"):
    for order in (O.LIST_SEARCH_CONTENT_ORDER_BY_VOTES, O.LIST_SEARCH_CONTENT_ORDER_BY_DATE_UPDATED):
        for att in range(3):
            try:
                docs = search(q, DocumentType.KERNEL, pages=4, order=order)
                break
            except Exception:
                time.sleep(5)
                docs = []
        for d in docs:
            s = d.kernel_document.best_public_score if d.kernel_document else None
            owner = d.owner_user.user_name if getattr(d, "owner_user", None) else "?"
            ref = owner + "/" + (d.slug or "")
            if s and s < 1.0 and ("knee" in (d.title or "").lower() + (d.slug or "") or ref.lower() in comp):
                seen[ref] = (s, d.update_time, d.title, d.votes)
recent = datetime.now(timezone.utc) - timedelta(days=3)
rows = sorted(seen.items(), key=lambda kv: -kv[1][0])
hits = [(k, v) for k, v in rows if v[0] >= best and v[1] and v[1].replace(tzinfo=timezone.utc) >= recent]
print("public notebooks scoring >= %.3f, updated in the last 3 days: %d" % (best, len(hits)))
for k, (s, t, title, votes) in (rows[:25] if "--all" in sys.argv else hits):
    print("%.3f  %s  %4d votes  %s | %s" % (s, str(t)[:16], votes, k, title))
