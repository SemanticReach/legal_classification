import requests
SERVER = 'http://18.220.128.24:8000'
KEY    = 'hb_6F7LQcFwm_sr7bZ_dqKKjVUrNTr2HNB2Ftp4QhJhv_Q'
DB     = 'fraud_db'
NS     = 'cuad_clauses'

def search(queries, top_k=3):
    r = requests.post(f'{SERVER}/compose/search_slots/{DB}/{NS}',
        headers={'X-API-Key': KEY},
        json={'slot_queries': queries, 'top_k': top_k}, timeout=30)
    return r.json().get('results', [])

clauses = {
    'MMT/Pfizer Non-Compete': 'During the Term, MMT shall not Commercialize in any manner any Competing Product in the Field in any country in the Territory; provided, however, the Parties hereby acknowledge that the restrictions set forth in this Section 2.3 shall not apply to any Affiliates of MMT (including Pfizer).',
    'CHT/Ehave Liquidated Damages': 'In addition, CHT may terminate this Agreement and the rights granted hereunder, in whole or in part, and without prejudice to enforcement of any other legal right or remedy, at any time without cause, by providing at least thirty (30) Business Days prior written notice to Ehave, but subject to payment of a termination fee equal to an amount set out in Schedule 6.',
    'Uncapped Liability EXCEPTING': 'EXCEPTING ONLY CLAIMS MADE PURSUANT TO SECTION 12.1, IN NO EVENT WILL EITHER PARTY BE LIABLE FOR ANY INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY, PUNITIVE OR CONSEQUENTIAL DAMAGES OF ANY KIND, INCLUDING ANY LOST PROFITS, LOST REVENUES OR LOST SAVINGS.',
    'XSPA Non-Compete': 'Throughout the Term and for a period of six (6) months after the expiration or termination of this Agreement, neither XSPA nor any of its affiliates shall, directly or indirectly, sell, offer for sale, market or promote any digital meditation or digital sleep products.',
}

for name, text in clauses.items():
    hits = search({'object': {'query': text[:500], 'weight': 1.0, 'encoding': 'semantic'}}, top_k=1)
    pred = hits[0]['data'].get('predicate', 'none') if hits else 'none'
    score = round(hits[0]['_score'], 4) if hits else 0
    print(f'{name}: vanilla={pred} ({score})')
