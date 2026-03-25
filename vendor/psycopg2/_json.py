import json

from psycopg2._psycopg import ISQLQuote, QuotedString
from psycopg2._psycopg import new_type, new_array_type, register_type


JSON_OID = 114
JSONARRAY_OID = 199

JSONB_OID = 3802
JSONBARRAY_OID = 3807


class Json:
    def __init__(self, adapted, dumps=None):
        self.adapted = adapted
        self._conn = None
        self._dumps = dumps or json.dumps

    def __conform__(self, proto):
        if proto is ISQLQuote:
            return self

    def dumps(self, obj):
        return self._dumps(obj)

    def prepare(self, conn):
        self._conn = conn

    def getquoted(self):
        s = self.dumps(self.adapted)
        qs = QuotedString(s)
        if self._conn is not None:
            qs.prepare(self._conn)
        return qs.getquoted()

    def __str__(self):
        return self.getquoted().decode('ascii', 'replace')


def register_json(conn_or_curs=None, globally=False, loads=None,
                  oid=None, array_oid=None, name='json'):
    return register_json(conn_or_curs=conn_or_curs, globally=globally,
        loads=loads, oid=JSON_OID, array_oid=JSONARRAY_OID)


def register_default_jsonb(conn_or_curs=None, globally=False, loads=None):
    return register_json(conn_or_curs=conn_or_curs, globally=globally,
        loads=loads, oid=JSONB_OID, array_oid=JSONBARRAY_OID, name='jsonb')


def _create_json_typecasters(oid, array_oid, loads=None, name='JSON'):
    if loads is None:
        loads = json.loads

    def typecast_json(s, cur):
        if s is None:
            return None
        return loads(s)

    JSON = new_type((oid, ), name, typecast_json)
    if array_oid is not None:
        JSONARRAY = new_array_type((array_oid, ), f"{name}ARRAY", JSON)
    else:
        JSONARRAY = None

    return JSON, JSONARRAY


def _get_json_oids(conn_or_curs, name='json'):
    from psycopg2.extensions import STATUS_IN_TRANSACTION
    from psycopg2.extras import _solve_conn_curs

    conn, curs = _solve_conn_curs(conn_or_curs)

    conn_status = conn.status

    typarray = conn.info.server_version >= 80300 and "typarray" or "NULL"

    curs.execute(
        "SELECT t.oid, %s FROM pg_type t WHERE t.typname = %%s;"
        % typarray, (name,))
    r = curs.fetchone()
    if conn_status != STATUS_IN_TRANSACTION and not conn.autocommit:
        conn.rollback()

    if not r:
        raise conn.ProgrammingError(f"{name} data type not found")

    return r
