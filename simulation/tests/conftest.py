import pytest
from synthetic import demo_geometry
from aerial_assembly.config import Physics
from aerial_assembly.model import build_model


@pytest.fixture(scope='session')
def bundle(tmp_path_factory):
    return demo_geometry(tmp_path_factory.mktemp('assets')/'demo.json')


@pytest.fixture(scope='session')
def grid_physics():
    # Match the current 27-state experiment configuration.
    return Physics(timestep=.002, contact_timeconst=.005)


@pytest.fixture(scope='session')
def model(bundle, grid_physics):
    return build_model(bundle, grid_physics)[0]
