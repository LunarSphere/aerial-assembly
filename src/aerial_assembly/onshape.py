"""Small optional Onshape adapter. Nothing calls this from local simulations.

Credentials are obtained from environment variables. Redirects are re-signed only
for explicitly trusted Onshape hosts; API headers never go to arbitrary hosts.
"""
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import time
from urllib.parse import quote, urljoin, urlsplit

import requests

from .config import digest, read_json, write_json


def signature(method, url, access_key, secret_key, date, nonce):
    parsed = urlsplit(url)
    content_type = 'application/json'
    message = '\n'.join([method, nonce, date, content_type, parsed.path, parsed.query, '']).lower()
    value = base64.b64encode(hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()).decode()
    return {'Date': date, 'On-Nonce': nonce, 'Content-Type': content_type,
            'Authorization': f'On {access_key}:HmacSHA256:{value}', 'Accept': '*/*'}


def merge_variables(tables, changes, independent):
    """Read/modify/write all local definitions, preserving derived expressions."""
    if set(changes) - set(independent):
        raise ValueError('Only configured independent variables may be changed')
    if not changes or any(not isinstance(v, str) or not v.strip() for v in changes.values()):
        raise ValueError('Changes must be nonempty expression strings with explicit units')
    local = [table for table in tables if not table.get('isReference', False)]
    if len(local) != 1:
        raise ValueError('Expected exactly one local Variable Studio table; inspect API response')
    fields = ['name', 'type', 'expression', 'description', 'configuredDescription', 'configuredExpression']
    variables = [{k: deepcopy(v[k]) for k in fields if k in v} for v in local[0]['variables']]
    names = [v['name'] for v in variables]
    if len(set(names)) != len(names) or set(changes)-set(names):
        raise ValueError('Unknown or ambiguous variable name')
    for v in variables:
        if v['name'] in changes:
            if v.get('configuredExpression'):
                raise ValueError('Configured variables require configuration APIs')
            if '#' in v.get('expression', ''):
                raise ValueError(f"Refusing to replace derived expression {v['name']}")
            v['expression'] = changes[v['name']]
    return variables


class OnshapeClient:
    def __init__(self, base_url='https://cad.onshape.com', api_version='v16', session=None):
        p = urlsplit(base_url)
        if p.scheme != 'https' or not p.hostname or not (p.hostname == 'cad.onshape.com' or p.hostname.endswith('.onshape.com')):
            raise ValueError('Expected an HTTPS Onshape host')
        self.base = base_url.rstrip('/')
        self.api = api_version
        self.access = os.environ.get('ONSHAPE_ACCESS_KEY', '')
        self.secret = os.environ.get('ONSHAPE_SECRET_KEY', '')
        if not self.access or not self.secret:
            raise ValueError('Set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY outside source code')
        self.session = session or requests.Session()

    def request(self, method, path, params=None, body=None):
        url = requests.Request(method, self.base+'/api/'+self.api+path, params=params).prepare().url
        trusted = {urlsplit(self.base).hostname, 'cad.onshape.com'}
        retries = 0
        for _ in range(12):
            parsed = urlsplit(url)
            if parsed.scheme != 'https' or parsed.hostname not in trusted or parsed.username or parsed.password:
                raise ValueError('Export redirect points outside trusted Onshape hosts; credentials were not sent')
            headers = signature(method, url, self.access, self.secret,
                                datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT'), secrets.token_hex(16))
            response = self.session.request(method, url, headers=headers, json=body,
                                            timeout=(10, 120), allow_redirects=False)
            if response.status_code == 402:
                raise RuntimeError('Onshape API allowance exhausted (402)')
            if response.status_code == 429 and retries < 4:
                delay = response.headers.get('Retry-After', '')
                time.sleep(min(30, float(delay) if delay.isdigit() else 2**retries))
                retries += 1
                continue
            if response.status_code in (307, 308):
                url = urljoin(url, response.headers['Location'])
                continue
            if response.status_code >= 400:
                # Do not expose signed URLs, credentials, or arbitrary error bodies.
                raise RuntimeError(f'Onshape returned HTTP {response.status_code} for {method} {path}')
            if response.status_code >= 300:
                raise RuntimeError('Unexpected Onshape redirect')
            return response
        raise RuntimeError('Onshape retry or redirect limit exceeded')

    def microversion(self, config):
        r = self.request('GET', f"/documents/d/{config['document_id']}/w/{config['workspace_id']}/currentmicroversion").json()
        value = r.get('microversion') if isinstance(r, dict) else None
        if not value:
            raise ValueError('No microversion in currentmicroversion response')
        return value

    def variables(self, config):
        path = f"/variables/d/{config['document_id']}/w/{config['workspace_id']}/e/{config['variable_studio_id']}/variables"
        return self.request('GET', path, params={'includeValuesAndReferencedVariables': 'true'}).json()

    def update_variables(self, config, changes, audit_directory):
        # Explicit future write entrypoint; the CLI requires --apply plus an experiment workspace declaration.
        if not config.get('experiment_workspace'):
            raise ValueError('Declare a dedicated experiment_workspace in Onshape configuration')
        audit = Path(audit_directory)
        audit.mkdir(parents=True, exist_ok=False)
        before_version = self.microversion(config)
        before = self.variables(config)
        payload = merge_variables(before, changes, config['independent_variables'])
        write_json(audit/'before.json', {'microversion': before_version, 'variables': before})
        write_json(audit/'request.json', payload)
        if self.microversion(config) != before_version:
            raise RuntimeError('Workspace changed while preparing variable update')
        path = f"/variables/d/{config['document_id']}/w/{config['workspace_id']}/e/{config['variable_studio_id']}/variables"
        self.request('POST', path, body=payload)
        after = self.variables(config)
        write_json(audit/'after.json', {'microversion': self.microversion(config), 'variables': after})
        return after

    def snapshot(self, config, cache_directory):
        """Export explicitly named parts at one immutable microversion. No CAD writes."""
        before = self.microversion(config)
        variables = self.variables(config)
        if self.microversion(config) != before:
            raise RuntimeError('Workspace changed during variable read; retry when idle')
        key = digest({'config': config, 'microversion': before, 'api': self.api})
        directory = Path(cache_directory)/key
        if (directory/'snapshot.json').exists():
            cached = read_json(directory/'snapshot.json')
            for part in cached['parts']:
                file = directory/part['file']
                if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != part['sha256']:
                    raise ValueError('Cached export is missing or changed; use a new cache directory')
            return directory
        directory.mkdir(parents=True, exist_ok=True)
        exports = []
        # Roles: visual/collision; names are resolved anew for every geometry revision.
        for studio in config['part_studios']:
            base = f"/parts/d/{config['document_id']}/m/{before}/e/{studio['element_id']}"
            parts = self.request('GET', base).json()
            for name in studio['part_names']:
                matches = [p for p in parts if p.get('name') == name]
                if len(matches) != 1:
                    raise ValueError(f'Expected one part named {name!r}, found {len(matches)}')
                pid = quote(matches[0]['partId'], safe='')
                result = self.request('GET', base+f'/partid/{pid}/stl',
                                      params={'mode': 'binary', 'units': 'meter', 'scale': 1,
                                              'chordTolerance': 0.00001, 'angleTolerance': 0.05})
                filename = f'part_{len(exports):04d}.stl'
                (directory/filename).write_bytes(result.content)
                exports.append({'name': name, 'role': studio['role'], 'file': filename,
                                'element_id': studio['element_id'], 'part_id': matches[0]['partId'],
                                'sha256': hashlib.sha256(result.content).hexdigest()})
        write_json(directory/'snapshot.json', {'microversion': before, 'document_id': config['document_id'],
                                               'variables': variables, 'mesh_units': 'm', 'parts': exports,
                                               'note': 'Supply/update target pose, feature metadata, and mass properties before preparation.'})
        return directory
