from contextlib import contextmanager

from src.db.factory import make_database

# Global database instance
_database = None


def get_database():
    global _database
    if _database is None:
        _database = make_database()
    return _database


@contextmanager
def get_db_session():
    database = get_database()
    with database.get_session() as session:
        yield session
