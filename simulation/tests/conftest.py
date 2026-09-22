import pytest
from synthetic import demo_geometry
from aerial_assembly.config import Physics
from aerial_assembly.model import build_model


@pytest.fixture(scope='session')
def bundle(tmp_path_factory):
    return demo_geometry(tmp_path_factory.mktemp('assets')/'demo.json')


@pytest.fixture(scope='session')
def refined_physics():
    # Preserve the historical accuracy checks independently of preview defaults.
    return Physics(timestep=.0000125, contact_timeconst=.00005)


@pytest.fixture(scope='session')
def model(bundle, refined_physics):
    return build_model(bundle, refined_physics)[0]
