import re as _re

from psycopg2._psycopg import (                            
    BINARYARRAY, BOOLEAN, BOOLEANARRAY, BYTES, BYTESARRAY, DATE, DATEARRAY,
    DATETIMEARRAY, DECIMAL, DECIMALARRAY, FLOAT, FLOATARRAY, INTEGER,
    INTEGERARRAY, INTERVAL, INTERVALARRAY, LONGINTEGER, LONGINTEGERARRAY,
    ROWIDARRAY, STRINGARRAY, TIME, TIMEARRAY, UNICODE, UNICODEARRAY,
    AsIs, Binary, Boolean, Float, Int, QuotedString, )

from psycopg2._psycopg import (                         
    PYDATE, PYDATETIME, PYDATETIMETZ, PYINTERVAL, PYTIME, PYDATEARRAY,
    PYDATETIMEARRAY, PYDATETIMETZARRAY, PYINTERVALARRAY, PYTIMEARRAY,
    DateFromPy, TimeFromPy, TimestampFromPy, IntervalFromPy, )

from psycopg2._psycopg import (                             
    adapt, adapters, encodings, connection, cursor,
    lobject, Xid, libpq_version, parse_dsn, quote_ident,
    string_types, binary_types, new_type, new_array_type, register_type,
    ISQLQuote, Notify, Diagnostics, Column, ConnectionInfo,
    QueryCanceledError, TransactionRollbackError,
    set_wait_callback, get_wait_callback, encrypt_password, )


ISOLATION_LEVEL_AUTOCOMMIT = 0
ISOLATION_LEVEL_READ_UNCOMMITTED = 4
ISOLATION_LEVEL_READ_COMMITTED = 1
ISOLATION_LEVEL_REPEATABLE_READ = 2
ISOLATION_LEVEL_SERIALIZABLE = 3
ISOLATION_LEVEL_DEFAULT = None


STATUS_SETUP = 0
STATUS_READY = 1
STATUS_BEGIN = 2
STATUS_SYNC = 3  
STATUS_ASYNC = 4  
STATUS_PREPARED = 5

STATUS_IN_TRANSACTION = STATUS_BEGIN


POLL_OK = 0
POLL_READ = 1
POLL_WRITE = 2
POLL_ERROR = 3


TRANSACTION_STATUS_IDLE = 0
TRANSACTION_STATUS_ACTIVE = 1
TRANSACTION_STATUS_INTRANS = 2
TRANSACTION_STATUS_INERROR = 3
TRANSACTION_STATUS_UNKNOWN = 4


def register_adapter(typ, callable):
    adapters[(typ, ISQLQuote)] = callable


class SQL_IN:
    def __init__(self, seq):
        self._seq = seq
        self._conn = None

    def prepare(self, conn):
        self._conn = conn

    def getquoted(self):
        pobjs = [adapt(o) for o in self._seq]
        if self._conn is not None:
            for obj in pobjs:
                if hasattr(obj, 'prepare'):
                    obj.prepare(self._conn)
        qobjs = [o.getquoted() for o in pobjs]
        return b'(' + b', '.join(qobjs) + b')'

    def __str__(self):
        return str(self.getquoted())


class NoneAdapter:
    def __init__(self, obj):
        pass

    def getquoted(self, _null=b"NULL"):
        return _null


def make_dsn(dsn=None, **kwargs):
    if dsn is None and not kwargs:
        return ''

    if not kwargs:
        parse_dsn(dsn)
        return dsn

    if 'database' in kwargs:
        if 'dbname' in kwargs:
            raise TypeError(
                "you can't specify both 'database' and 'dbname' arguments")
        kwargs['dbname'] = kwargs.pop('database')

    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}

    if dsn is not None:
        tmp = parse_dsn(dsn)
        tmp.update(kwargs)
        kwargs = tmp

    dsn = " ".join(["{}={}".format(k, _param_escape(str(v)))
        for (k, v) in kwargs.items()])

    parse_dsn(dsn)

    return dsn


def _param_escape(s,
        re_escape=_re.compile(r"([\\'])"),
        re_space=_re.compile(r'\s')):
    if not s:
        return "''"

    s = re_escape.sub(r'\\\1', s)
    if re_space.search(s):
        s = "'" + s + "'"

    return s


from psycopg2._json import register_default_json, register_default_jsonb    

try:
    JSON, JSONARRAY = register_default_json()
    JSONB, JSONBARRAY = register_default_jsonb()
except ImportError:
    pass

del register_default_json, register_default_jsonb

from psycopg2. _range import Range                              
del Range


for k, v in list(encodings.items()):
    k = k.replace('_', '').replace('-', '').upper()
    encodings[k] = v

del k, v
