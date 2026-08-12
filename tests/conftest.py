from ipaddress import ip_address
import os

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from brd_srs_testgen.storage import RunRepository


def _is_local_address(address: object) -> bool:
    if address is None:
        return True
    try:
        return ip_address(str(address)).is_loopback
    except ValueError:
        return False


@pytest.fixture
def test_database_connection() -> psycopg.Connection:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set")
    try:
        database_name = conninfo_to_dict(database_url).get("dbname")
    except psycopg.Error:
        pytest.fail("TEST_DATABASE_URL must identify a dedicated local test database")
    if database_name != "brd_srs_test":
        pytest.fail("TEST_DATABASE_URL must identify a dedicated local test database")

    connection = None
    try:
        connection = psycopg.connect(database_url, autocommit=True)
        live_database, server_address = connection.execute(
            "SELECT current_database(), inet_server_addr()"
        ).fetchone()
    except psycopg.Error:
        if connection is not None:
            connection.close()
        pytest.fail("TEST_DATABASE_URL must identify a dedicated local test database")
    if live_database != "brd_srs_test" or not _is_local_address(server_address):
        connection.close()
        pytest.fail("TEST_DATABASE_URL must identify a dedicated local test database")

    yield connection
    connection.close()


@pytest.fixture
def repository(test_database_connection: psycopg.Connection) -> RunRepository:
    database_url = os.environ["TEST_DATABASE_URL"]

    result = RunRepository(database_url)
    result.initialize()
    test_database_connection.execute("TRUNCATE TABLE runs CASCADE")
    yield result
    test_database_connection.execute("TRUNCATE TABLE runs CASCADE")
