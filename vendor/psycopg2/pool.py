
import psycopg2
from psycopg2 import extensions as _ext


class PoolError(psycopg2.Error):
    pass


class AbstractConnectionPool:

    def __init__(self, minconn, maxconn, *args, **kwargs):
        self.minconn = int(minconn)
        self.maxconn = int(maxconn)
        self.closed = False

        self._args = args
        self._kwargs = kwargs

        self._pool = []
        self._used = {}
        self._rused = {}
        self._keys = 0

        for i in range(self.minconn):
            self._connect()

    def _connect(self, key=None):
        conn = psycopg2.connect(*self._args, **self._kwargs)
        if key is not None:
            self._used[key] = conn
            self._rused[id(conn)] = key
        else:
            self._pool.append(conn)
        return conn

    def _getkey(self):
        self._keys += 1
        return self._keys

    def _getconn(self, key=None):
        if self.closed:
            raise PoolError("connection pool is closed")
        if key is None:
            key = self._getkey()

        if key in self._used:
            return self._used[key]

        if self._pool:
            self._used[key] = conn = self._pool.pop()
            self._rused[id(conn)] = key
            return conn
        else:
            if len(self._used) == self.maxconn:
                raise PoolError("connection pool exhausted")
            return self._connect(key)

    def _putconn(self, conn, key=None, close=False):
        if self.closed:
            raise PoolError("connection pool is closed")

        if key is None:
            key = self._rused.get(id(conn))
            if key is None:
                raise PoolError("trying to put unkeyed connection")

        if len(self._pool) < self.minconn and not close:
            if not conn.closed:
                status = conn.info.transaction_status
                if status == _ext.TRANSACTION_STATUS_UNKNOWN:
                    conn.close()
                elif status != _ext.TRANSACTION_STATUS_IDLE:
                    conn.rollback()
                    self._pool.append(conn)
                else:
                    self._pool.append(conn)
        else:
            conn.close()

        if not self.closed or key in self._used:
            del self._used[key]
            del self._rused[id(conn)]

    def _closeall(self):
        if self.closed:
            raise PoolError("connection pool is closed")
        for conn in self._pool + list(self._used.values()):
            try:
                conn.close()
            except Exception:
                pass
        self.closed = True


class SimpleConnectionPool(AbstractConnectionPool):

    getconn = AbstractConnectionPool._getconn
    putconn = AbstractConnectionPool._putconn
    closeall = AbstractConnectionPool._closeall


class ThreadedConnectionPool(AbstractConnectionPool):

    def __init__(self, minconn, maxconn, *args, **kwargs):
        import threading
        AbstractConnectionPool.__init__(
            self, minconn, maxconn, *args, **kwargs)
        self._lock = threading.Lock()

    def getconn(self, key=None):
        self._lock.acquire()
        try:
            return self._getconn(key)
        finally:
            self._lock.release()

    def putconn(self, conn=None, key=None, close=False):
        self._lock.acquire()
        try:
            self._putconn(conn, key, close)
        finally:
            self._lock.release()

    def closeall(self):
        self._lock.acquire()
        try:
            self._closeall()
        finally:
            self._lock.release()
