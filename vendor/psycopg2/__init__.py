from psycopg2._psycopg import (                     
    BINARY, NUMBER, STRING, DATETIME, ROWID,

    Binary, Date, Time, Timestamp,
    DateFromTicks, TimeFromTicks, TimestampFromTicks,

    Error, Warning, DataError, DatabaseError, ProgrammingError, IntegrityError,
    InterfaceError, InternalError, NotSupportedError, OperationalError,

    _connect, apilevel, threadsafety, paramstyle,
    __version__, __libpq_version__,
)

from psycopg2 import extensions as _ext
_ext.register_adapter(tuple, _ext.SQL_IN)
_ext.register_adapter(type(None), _ext.NoneAdapter)

from decimal import Decimal                         
from psycopg2._psycopg import Decimal as Adapter    
_ext.register_adapter(Decimal, Adapter)
del Decimal, Adapter


def connect(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
    kwasync = {}
    if 'async' in kwargs:
        kwasync['async'] = kwargs.pop('async')
    if 'async_' in kwargs:
        kwasync['async_'] = kwargs.pop('async_')

    dsn = _ext.make_dsn(dsn, **kwargs)
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
    if cursor_factory is not None:
        conn.cursor_factory = cursor_factory

    return conn
