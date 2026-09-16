import pytest

from aerial_assembly.onshape import OnshapeClient, merge_variables, signature


def test_preserve_derived_expressions_and_unrelated_variables():
    tables = [{'variables':[{'name':'ramp_angle','type':'ANGLE','expression':'50 deg'},
                            {'name':'socket_width','type':'LENGTH','expression':'#leg_width+2*#socket_clearance'}]}]
    result = merge_variables(tables, {'ramp_angle':'45 deg'}, ['ramp_angle'])
    assert result[0]['expression'] == '45 deg'
    assert result[1] == tables[0]['variables'][1]
    assert tables[0]['variables'][0]['expression'] == '50 deg'
    with pytest.raises(ValueError):
        merge_variables(tables, {'socket_width':'10 mm'}, ['ramp_angle'])
    with pytest.raises(ValueError, match='derived'):
        merge_variables(tables, {'socket_width':'10 mm'}, ['socket_width'])


def test_signature_changes_with_redirect_path():
    args = ('key','secret','Tue, 15 Sep 2026 00:00:00 GMT','0123456789abcdef')
    a = signature('GET','https://cad.onshape.com/api/v16/a?x=1',*args)
    b = signature('GET','https://cad.onshape.com/api/v16/b?x=1',*args)
    assert a['Authorization'] != b['Authorization']
    assert 'secret' not in a['Authorization']


class Response:
    def __init__(self, code, headers=None):
        self.status_code = code
        self.headers = headers or {}


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method,url,kwargs))
        return next(self.responses)


def client(monkeypatch, responses):
    monkeypatch.setenv('ONSHAPE_ACCESS_KEY','test-key')
    monkeypatch.setenv('ONSHAPE_SECRET_KEY','test-secret')
    session = Session(responses)
    return OnshapeClient(session=session), session


def test_signed_redirect_and_no_cross_host_credential_leak(monkeypatch):
    c, s = client(monkeypatch,[Response(307,{'Location':'/api/v16/download'}),Response(200)])
    c.request('GET','/export')
    assert len(s.calls) == 2
    assert s.calls[0][2]['headers']['Authorization'] != s.calls[1][2]['headers']['Authorization']
    c, s = client(monkeypatch,[Response(307,{'Location':'https://untrusted.example/download'})])
    with pytest.raises(ValueError, match='outside trusted'):
        c.request('GET','/export')
    assert len(s.calls) == 1


def test_api_quota_and_retry(monkeypatch):
    c, _ = client(monkeypatch,[Response(402)])
    with pytest.raises(RuntimeError, match='allowance'):
        c.request('GET','/export')
    sleeps = []
    monkeypatch.setattr('aerial_assembly.onshape.time.sleep',sleeps.append)
    c, s = client(monkeypatch,[Response(429,{'Retry-After':'1'}),Response(200)])
    c.request('GET','/export')
    assert sleeps == [1] and len(s.calls) == 2
