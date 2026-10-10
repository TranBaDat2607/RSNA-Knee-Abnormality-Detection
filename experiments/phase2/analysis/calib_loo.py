"""Can the 12 report labels jointly predict each expert label better than the matching report label alone?
Leave-one-out on gold-58, ridge-penalised logistic per finding; compares LOO AUC with the raw teach4 column."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
gold = pd.read_csv('C:/Users/Admin/Desktop/RSNA-Knee-Abnormality-Detection/data/gold_annotated.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T].astype(float)
teach = pd.read_csv('teach4.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T].reindex(gold.index)
print('gold positives:', (gold > .5).sum().to_dict())
print('teach4 mean on gold-58 vs gold rate:', {t: (round(teach[t].mean(), 2), round((gold[t] > .5).mean(), 2)) for t in T})
X = np.log(np.clip(teach.values, 1e-3, 1 - 1e-3) / (1 - np.clip(teach.values, 1e-3, 1 - 1e-3)))
for C in (0.05, 0.2, 1.0):
    rows = {}
    for j, t in enumerate(T):
        y = (gold[t] > .5).astype(int).values
        p = np.zeros(len(y))
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            clf = LogisticRegression(C=C, max_iter=2000).fit(X[m], y[m])
            p[i] = clf.predict_proba(X[i:i + 1])[0, 1]
        rows[t] = (roc_auc_score(y, teach[t]), roc_auc_score(y, p))
    raw = np.mean([a for a, _ in rows.values()]); cal = np.mean([b for _, b in rows.values()])
    print(f'C={C}: raw {raw:.4f}  joint-LOO {cal:.4f}', {t: f'{a:.2f}->{b:.2f}' for t, (a, b) in rows.items()})
