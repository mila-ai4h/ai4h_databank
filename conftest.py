# pytest calls this file before any test. It is used to define fixtures and hooks.
# pytest_addoption is a hook that allows us to add command line options.
import pytest


def pytest_addoption(parser):
    parser.addoption("--run_expensive", action="store_true", default=False, help="run expensive tests")


@pytest.fixture(scope="module")
def run_expensive(request):
    """This fixture is used to determine if we should run expensive tests.
    Those tests will make API calls if run_expensive is True, otherwise they will be mocked.
    """
    return request.config.getoption("--run_expensive")
