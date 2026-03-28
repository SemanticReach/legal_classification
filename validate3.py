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
    'Anti-Assignment plain': 'Neither party may assign this Agreement or any rights or obligations hereunder, by operation of law or otherwise, without the prior written consent of the other party, which consent shall not be unreasonably withheld or delayed.',
    'Termination For Convenience plain': 'Either party may terminate this Agreement for any reason or no reason upon thirty (30) days prior written notice to the other party, without liability to the terminating party except for payment of amounts due and owing as of the termination date.',
    'Non-Compete plain': 'During the term of this Agreement and for a period of two (2) years following termination, Employee shall not directly or indirectly engage in any business activity that competes with the Company within any geographic area where the Company conducts business.',
    'Cap On Liability plain': 'In no event shall either party be liable to the other for any indirect, incidental, special, exemplary, or consequential damages, however caused, even if such party has been advised of the possibility of such damages. Each party is liability shall not exceed the total fees paid in the twelve months preceding the claim.',
}

for name, text in clauses.items():
    hits = search({'object': {'query': text[:500], 'weight': 1.0, 'encoding': 'semantic'}})
    pred = hits[0]['data'].get('predicate', 'none') if hits else 'none'
    score = round(hits[0]['_score'], 4) if hits else 0
    print(f'{name}: vanilla={pred} ({score})')
