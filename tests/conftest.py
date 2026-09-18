import pytest
from synthetic import demo_geometry
from aerial_assembly.model import build_model


@pytest.fixture(scope='session')
def bundle(tmp_path_factory):
    return demo_geometry(tmp_path_factory.mktemp('assets')/'demo.json')


@pytest.fixture(scope='session')
def model(bundle):
    return build_model(bundle)[0]
