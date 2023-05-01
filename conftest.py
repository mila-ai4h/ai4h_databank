import pytest


def pytest_addoption(parser):
    parser.addoption("--run_expensive", action="store_true", default=False, help="run expensive tests")


@pytest.fixture(scope="module")
def run_expensive(request):
    return request.config.getoption("--run_expensive")
