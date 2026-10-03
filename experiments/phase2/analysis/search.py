import sys, json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.search.types.search_api_service import ListEntitiesRequest, ListEntitiesFilters
from kagglesdk.search.types.search_enums import DocumentType, ListSearchContentOrderBy
api=KaggleApi(); api.authenticate()
def search(query, dtype, pages=5, order=ListSearchContentOrderBy.LIST_SEARCH_CONTENT_ORDER_BY_VOTES):
    docs=[]; tok=None
    with api.build_kaggle_client() as kc:
        for _ in range(pages):
            r=ListEntitiesRequest(); f=ListEntitiesFilters(); f.query=query; f.document_types=[dtype]; r.filters=f; r.page_size=50
            r.canonical_order_by=order
            if tok: r.page_token=tok
            resp=kc.search.search_api_client.list_entities(r)
            docs+=resp.documents; tok=resp.next_page_token
            if not tok: break
    return docs
if __name__=='__main__':
    q=sys.argv[1]; t=sys.argv[2]
    ds=search(q, getattr(DocumentType,t))
    for d in ds:
        extra=''
        if d.kernel_document: extra=f"score={d.kernel_document.best_public_score}"
        if d.discussion_document: extra=f"forum={d.discussion_document.forum_name}"
        print(d.id, d.votes, str(d.update_time)[:10], d.slug, '|', d.title, '|', extra)
