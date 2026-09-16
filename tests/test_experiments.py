from copy import deepcopy

from aerial_assembly.config import Release
from aerial_assembly.experiments import compare_candidates, summarize


def test_candidate_rejection_does_not_abort_search(bundle, monkeypatch, tmp_path):
    invalid = deepcopy(bundle)
    invalid['asset_hash'] = 'invalid-candidate'
    def fake_batch(candidate, *args, **kwargs):
        if candidate['asset_hash'] == 'invalid-candidate':
            raise ValueError('Impossible flush seating')
        return summarize([{'status':'success','settling_time':.2}])
    monkeypatch.setattr('aerial_assembly.experiments.batch', fake_batch)
    report = compare_candidates([invalid,bundle], [Release()], tmp_path/'search')
    assert len(report['ranking']) == 1
    assert report['ranking'][0]['candidate'] == 1
    assert report['excluded_invalid_candidates'][0]['rejection_reason'] == 'Impossible flush seating'


def test_deterministic_sweep_does_not_claim_binomial_interval():
    result = summarize([{'status':'success','settling_time':.2}], iid=False)
    assert result['wilson95_among_valid'] is None
