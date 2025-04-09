import time
from datetime import datetime, timedelta

from beartype.typing import Union


class Timer:
    def __init__(self) -> None:
        self.dtime_format = "%Y-%m-%d %H:%M:%S.%f"

    def datetime_diff(
        self,
        dtime1: Union[str, datetime],
        dtime2: Union[str, datetime],
        offset: float = 0,
    ):
        """
        Compute time difference between two datetimes. The offset potentially accounts
        for datetimes coming from different timezone which user wants to disregard. Typical
        example is streaming from UK to AssemblyAI server which is 1 hrs behind.

        The exact formula is:
        diff = dtime1 - dtime2 - offset

        Args
        :dtime1: datetime
        :dtime2: datetime
        :offset: offset in seconds to account for different timezone for example

        """
        if isinstance(dtime1, str):
            dtime1 = datetime.strptime(dtime1, self.dtime_format)
        if isinstance(dtime2, str):
            dtime2 = datetime.strptime(dtime2, self.dtime_format)

        return (dtime1 - dtime2).total_seconds() - offset

    def now(self) -> datetime:
        return datetime.now()

    def datetime(self) -> str:
        """
        Return a datetime to record timestamps of events.
        """
        return datetime.fromtimestamp(time.time()).strftime(self.dtime_format)

    def increment_datetime(self, dtime, offset: int) -> datetime:
        """
        Incrememnt datetime by offset in seconds
        """
        return datetime.strptime(dtime, self.dtime_format) + timedelta(seconds=offset)
