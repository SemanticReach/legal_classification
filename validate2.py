import requests
SERVER = 'http://18.220.128.24:8000'
KEY    = 'hb_6F7LQcFwm_sr7bZ_dqKKjVUrNTr2HNB2Ftp4QhJhv_Q'
DB     = 'fraud_db'
NS     = 'cuad_clauses'

def search(queries, top_k=1):
    r = requests.post(f'{SERVER}/compose/search_slots/{DB}/{NS}',
        headers={'X-API-Key': KEY},
        json={'slot_queries': queries, 'top_k': top_k}, timeout=30)
    return r.json().get('results', [])

clauses = {
    'Nuance/SpinCo Irrevocable': 'Subject to the terms and conditions of this Agreement, as of the Distribution Date, Nuance hereby grants to SpinCo and the members of the SpinCo Group a worldwide, non-exclusive, fully paid-up, perpetual and irrevocable, transferable, sublicensable license under the Nuance Patents, solely to the extent that claims of the Nuance Patents cover products or services of the SpinCo Business in the SpinCo Field of Use.',
    'NCM Covenant Not To Sue': 'NCM shall not engage in any conduct which may place Network Affiliate or any Network Affiliate Mark in a negative light or context, nor shall it contest or assist others in contesting the title or any rights of Network Affiliate in and to any Network Affiliate Mark. Neither party will at any time challenge or otherwise do anything inconsistent with the other partys right, title or interest in its property.',
    'Honeywell Irrevocable': 'Hence, as of the Distribution Date, Honeywell hereby grants to SpinCo and the members of the SpinCo Group a non-exclusive, royalty-free, fully-paid, perpetual, sublicenseable, worldwide license to use and exercise rights under the Honeywell Shared IP, said license being limited to use of a similar type, scope and extent as used in the SpinCo Business prior to the Distribution Date.',
    'AT&T Irrevocable': 'Vendor hereby grants and promises to grant and have granted to AT&T and its Affiliates a royalty-free, nonexclusive, sublicensable, assignable, transferable, irrevocable, perpetual, world-wide license in and to any applicable Intellectual Property Rights of Vendor to use, copy, modify, distribute, display, perform, import, make, sell, offer to sell, and exploit any Intellectual Property Rights of Vendor.',
}

for name, text in clauses.items():
    hits = search({'object': {'query': text[:500], 'weight': 1.0, 'encoding': 'semantic'}})
    pred = hits[0]['data'].get('predicate', 'none') if hits else 'none'
    score = round(hits[0]['_score'], 4) if hits else 0
    print(f'{name}: vanilla={pred} ({score})')
