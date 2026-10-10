import sys, re, html
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.discussions.types.discussions_api_service import ApiGetTopicRequest, ApiListCommentsRequest
api=KaggleApi(); api.authenticate()
def clean(s): return html.unescape(re.sub(r'<[^>]+>','',s or '')).strip()
def dump(c, depth, out):
    out.append('  '*depth+f"-- [{c.author_name} | {c.votes}v | {str(c.post_date)[:10]}] "+clean(c.content).replace('\n','\n'+'  '*depth))
    for r in (c.replies or []): dump(r, depth+1, out)
with api.build_kaggle_client() as kc:
    cl=kc.discussions.discussion_api_client
    for tid in map(int, sys.argv[1:]):
        out=[]
        r=ApiGetTopicRequest(); r.id=tid; t=cl.get_topic(r).topic
        out.append(f"#### {tid} {t.title} [{t.author_name} {t.votes}v {str(t.post_date)[:10]}]\n{clean(t.content)}")
        tok=None
        while True:
            q=ApiListCommentsRequest(); q.topic_id=tid; q.page_size=100
            if tok: q.page_token=tok
            resp=cl.list_comments(q)
            for c in resp.comments: dump(c,0,out)
            tok=resp.next_page_token
            if not tok: break
        open(f'disc/{tid}.txt','w',encoding='utf8').write('\n'.join(out))
        print(tid, len('\n'.join(out)))
