def lookup(code):
    from psycopg2._psycopg import sqlstate_errors
    return sqlstate_errors[code]
