import pytest
from fastapi.testclient import TestClient

from pyaesthetics.api.run import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
