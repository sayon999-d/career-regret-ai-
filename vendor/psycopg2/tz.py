
import datetime
import time

ZERO = datetime.timedelta(0)


class FixedOffsetTimezone(datetime.tzinfo):
    _name = None
    _offset = ZERO

    _cache = {}

    def __init__(self, offset=None, name=None):
        if offset is not None:
            if not isinstance(offset, datetime.timedelta):
                offset = datetime.timedelta(minutes=offset)
            self._offset = offset
        if name is not None:
            self._name = name

    def __new__(cls, offset=None, name=None):
        key = (offset, name)
        try:
            return cls._cache[key]
        except KeyError:
            tz = super().__new__(cls, offset, name)
            cls._cache[key] = tz
            return tz

    def __repr__(self):
        return "psycopg2.tz.FixedOffsetTimezone(offset=%r, name=%r)" \
            % (self._offset, self._name)

    def __eq__(self, other):
        if isinstance(other, FixedOffsetTimezone):
            return self._offset == other._offset
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, FixedOffsetTimezone):
            return self._offset != other._offset
        else:
            return NotImplemented

    def __getinitargs__(self):
        return self._offset, self._name

    def utcoffset(self, dt):
        return self._offset

    def tzname(self, dt):
        if self._name is not None:
            return self._name

        minutes, seconds = divmod(self._offset.total_seconds(), 60)
        hours, minutes = divmod(minutes, 60)
        rv = "%+03d" % hours
        if minutes or seconds:
            rv += ":%02d" % minutes
            if seconds:
                rv += ":%02d" % seconds

        return rv

    def dst(self, dt):
        return ZERO


STDOFFSET = datetime.timedelta(seconds=-time.timezone)
if time.daylight:
    DSTOFFSET = datetime.timedelta(seconds=-time.altzone)
else:
    DSTOFFSET = STDOFFSET
DSTDIFF = DSTOFFSET - STDOFFSET


class LocalTimezone(datetime.tzinfo):
    def utcoffset(self, dt):
        if self._isdst(dt):
            return DSTOFFSET
        else:
            return STDOFFSET

    def dst(self, dt):
        if self._isdst(dt):
            return DSTDIFF
        else:
            return ZERO

    def tzname(self, dt):
        return time.tzname[self._isdst(dt)]

    def _isdst(self, dt):
        tt = (dt.year, dt.month, dt.day,
              dt.hour, dt.minute, dt.second,
              dt.weekday(), 0, -1)
        stamp = time.mktime(tt)
        tt = time.localtime(stamp)
        return tt.tm_isdst > 0


LOCAL = LocalTimezone()

