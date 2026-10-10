import csv, sys, time
from search import search
from kagglesdk.search.types.search_enums import DocumentType, ListSearchContentOrderBy as O
rows=[r for r in csv.reader(open(sys.argv[1],encoding='utf8',errors='replace')) if r]
out=[]
for r in rows:
    slug=r[0].split('/')[1]
    s=None
    for att in range(3):
        try: ds=search(r[1], DocumentType.KERNEL, pages=1); break
        except Exception as e: time.sleep(5); ds=[]
    for d in ds:
        if d.slug==slug: s=d.kernel_document.best_public_score; break
    out.append((s, r[3][:10], r[0], r[1])); time.sleep(0.5)
for o in sorted(out,key=lambda o:-(o[0] or 0)): print(*o,sep=' | ')
