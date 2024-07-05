import os
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import lru_cache
from utils import months_between
from tqdm import tqdm


class DescriptorReader:
    patterns_iter = [
        re.compile(
            "^router (?P<nickname>\S*) (?P<address>\S*) (?P<or_port>\d*) (?P<socks_port>\d*) (?P<dir_port>\d*)$\n",
            re.MULTILINE,
        ),
        re.compile("^platform (?P<platform>.*)$\n", re.MULTILINE),
        re.compile(
            "^bandwidth (?P<bandwidth_avg>\d*) (?P<bandwidth_burst>\d*) (?P<bandwidth_observed>\d*)$\n",
            re.MULTILINE,
        ),
        re.compile("^contact (?P<contact>.*)$\n", re.MULTILINE),
        re.compile("^published (?P<published>.*)$\n", re.MULTILINE),
        re.compile("^fingerprint (?P<fingerprint>.*)$\n", re.MULTILINE),
        re.compile("^uptime (?P<uptime>\d*)$\n", re.MULTILINE),
    ]

    patterns_all = {
        "exit_policy": re.compile("(^(?:reject|accept) \S*$)\n", re.MULTILINE),
        "family": re.compile("^family (.*$\n(?:^\$.*$\n)*)", re.MULTILINE),
        "proto": re.compile("^proto (?P<proto>.*)$\n", re.MULTILINE),
    }

    def __init__(
        self,
        start: datetime,
        end: datetime,
        folder: str = "cache",
        bridge: bool = False,
    ):
        self._folder = folder
        self._start = start
        self._end = end
        self._bridge = bridge

    @staticmethod
    def _get_descriptor_data_from_file(file_path: str) -> str | None:
        """
        Read the data from the given file and returns a dict with the keys passed in as argument (see regex on top)
        :param file_path: descriptor file to read
        :return: the content of the file
        """
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            return f.read()

    def get_relays_timespan(self, start: datetime, end: datetime):
        """
        Returns a dict with the relays found in the given timespan
        :param start: start time
        :param end: end time
        :return: list with the relays found in the given timespan
        """
        if end is None:
            end = start
        if not (self._start <= start <= end <= self._end):
            raise ValueError(
                "start and end must be between {} and {}".format(self._start, self._end)
            )

        months = months_between(start, end)
        sub_dir = "bridge_descriptors" if self._bridge else "relay_descriptors"
        month_dir = (
            "bridge-server-descriptors-{}" if self._bridge else "server-descriptors-{}"
        )
        dir_path = os.path.join(self._folder, sub_dir, month_dir)
        dir_paths = [dir_path.format(month.strftime("%Y-%m")) for month in months]

        for d in dir_paths:
            if not os.path.exists(d):
                raise ValueError("Missing descriptors for {}".format(d))

            for descriptor_file in tqdm(os.listdir(d)):
                yield self._get_descriptor_data_from_file(
                    os.path.join(d, descriptor_file)
                )

    @lru_cache(maxsize=1000)
    def get_descriptor(
        self, descriptor_hash: str, date: datetime, keys: tuple[str] | None = None
    ) -> dict | None:
        sub_dir = "bridge_descriptors" if self._bridge else "relay_descriptors"
        month_dir = (
            "bridge-server-descriptors-{}" if self._bridge else "server-descriptors-{}"
        ).format(date.strftime("%Y-%m"))
        dir_path = os.path.join(
            self._folder, sub_dir, month_dir, descriptor_hash.lower()
        )

        descriptor = self._get_descriptor_data_from_file(dir_path)
        if descriptor is None:
            return None
        result = dict()
        for pattern in DescriptorReader.patterns_iter:
            for match in pattern.finditer(descriptor):
                result |= {
                    k: v
                    for k, v in match.groupdict().items()
                    if keys is None or k in keys
                }

        for key, pattern in [
            (k, p)
            for k, p in DescriptorReader.patterns_all.items()
            if keys is None or k in keys
        ]:
            if key == "family" or key == "proto":
                found = pattern.findall(descriptor)
                # flatten list of lists
                found = [x[1:] for xs in [item.split() for item in found] for x in xs]
                result |= {key: found}
            else:
                result |= {key: pattern.findall(descriptor)}

        return result

    def get_most_recent_descriptor(
        self, descriptor_hash: str, date: datetime, month_back: int
    ) -> dict | None:
        """
        Returns the most recent descriptor for the given relay
        :param descriptor_hash: relay's hash
        :param date: date to search for the descriptor
        :param month_back: number of months to go back
        :return: dict with the descriptor's data
        """
        oldest_month = date - relativedelta(months=month_back)
        months = months_between(oldest_month, date)

        for m in reversed(months):
            descriptor = self.get_descriptor(descriptor_hash, m)
            if descriptor is not None:
                return descriptor
        return None
