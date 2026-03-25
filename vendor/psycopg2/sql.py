

import string

from psycopg2 import extensions as ext


_formatter = string.Formatter()


class Composable:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __repr__(self):
        return f"{self.__class__.__name__}({self._wrapped!r})"

    def as_string(self, context):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Composed):
            return Composed([self]) + other
        if isinstance(other, Composable):
            return Composed([self]) + Composed([other])
        else:
            return NotImplemented

    def __mul__(self, n):
        return Composed([self] * n)

    def __eq__(self, other):
        return type(self) is type(other) and self._wrapped == other._wrapped

    def __ne__(self, other):
        return not self.__eq__(other)


class Composed(Composable):
    def __init__(self, seq):
        wrapped = []
        for i in seq:
            if not isinstance(i, Composable):
                raise TypeError(
                    f"Composed elements must be Composable, got {i!r} instead")
            wrapped.append(i)

        super().__init__(wrapped)

    @property
    def seq(self):
        return list(self._wrapped)

    def as_string(self, context):
        rv = []
        for i in self._wrapped:
            rv.append(i.as_string(context))
        return ''.join(rv)

    def __iter__(self):
        return iter(self._wrapped)

    def __add__(self, other):
        if isinstance(other, Composed):
            return Composed(self._wrapped + other._wrapped)
        if isinstance(other, Composable):
            return Composed(self._wrapped + [other])
        else:
            return NotImplemented

    def join(self, joiner):
        if isinstance(joiner, str):
            joiner = SQL(joiner)
        elif not isinstance(joiner, SQL):
            raise TypeError(
                "Composed.join() argument must be a string or an SQL")

        return joiner.join(self)


class SQL(Composable):
    def __init__(self, string):
        if not isinstance(string, str):
            raise TypeError("SQL values must be strings")
        super().__init__(string)

    @property
    def string(self):
        return self._wrapped

    def as_string(self, context):
        return self._wrapped

    def format(self, *args, **kwargs):
        rv = []
        autonum = 0
        for pre, name, spec, conv in _formatter.parse(self._wrapped):
            if spec:
                raise ValueError("no format specification supported by SQL")
            if conv:
                raise ValueError("no format conversion supported by SQL")
            if pre:
                rv.append(SQL(pre))

            if name is None:
                continue

            if name.isdigit():
                if autonum:
                    raise ValueError(
                        "cannot switch from automatic field numbering to manual")
                rv.append(args[int(name)])
                autonum = None

            elif not name:
                if autonum is None:
                    raise ValueError(
                        "cannot switch from manual field numbering to automatic")
                rv.append(args[autonum])
                autonum += 1

            else:
                rv.append(kwargs[name])

        return Composed(rv)

    def join(self, seq):
        rv = []
        it = iter(seq)
        try:
            rv.append(next(it))
        except StopIteration:
            pass
        else:
            for i in it:
                rv.append(self)
                rv.append(i)

        return Composed(rv)


class Identifier(Composable):
    def __init__(self, *strings):
        if not strings:
            raise TypeError("Identifier cannot be empty")

        for s in strings:
            if not isinstance(s, str):
                raise TypeError("SQL identifier parts must be strings")

        super().__init__(strings)

    @property
    def strings(self):
        return self._wrapped

    @property
    def string(self):
        if len(self._wrapped) == 1:
            return self._wrapped[0]
        else:
            raise AttributeError(
                "the Identifier wraps more than one than one string")

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(repr, self._wrapped))})"

    def as_string(self, context):
        return '.'.join(ext.quote_ident(s, context) for s in self._wrapped)


class Literal(Composable):
    @property
    def wrapped(self):
        return self._wrapped

    def as_string(self, context):
        if isinstance(context, ext.connection):
            conn = context
        elif isinstance(context, ext.cursor):
            conn = context.connection
        else:
            raise TypeError("context must be a connection or a cursor")

        a = ext.adapt(self._wrapped)
        if hasattr(a, 'prepare'):
            a.prepare(conn)

        rv = a.getquoted()
        if isinstance(rv, bytes):
            rv = rv.decode(ext.encodings[conn.encoding])

        return rv


class Placeholder(Composable):

    def __init__(self, name=None):
        if isinstance(name, str):
            if ')' in name:
                raise ValueError(f"invalid name: {name!r}")

        elif name is not None:
            raise TypeError(f"expected string or None as name, got {name!r}")

        super().__init__(name)

    @property
    def name(self):
        return self._wrapped

    def __repr__(self):
        if self._wrapped is None:
            return f"{self.__class__.__name__}()"
        else:
            return f"{self.__class__.__name__}({self._wrapped!r})"

    def as_string(self, context):
        if self._wrapped is not None:
            return f"%({self._wrapped})s"
        else:
            return "%s"


NULL = SQL("NULL")
DEFAULT = SQL("DEFAULT")
