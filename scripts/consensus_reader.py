import os
import datetime
import re
from typing import List
from dateutil.relativedelta import relativedelta
from functools import lru_cache


class ConsensusReader:
    # Regex to parse a router entry in the consensus.  Named group to get only relevant info when needed
    pattern = re.compile(
        '(^r (?P<nickname>\S*) (?P<id>\S*) (?P<digest>\S*) (?P<publication>\S* \S*) (?P<ip>\S*) (?P<orport>\S*) (?P<dirport>\S*)$\n)'
        '(^a (?P<ipv6>\S*)$\n)?'
        '(^s (?P<flags>(\S ?)*)$\n)'
        '(^v (?P<version>.*)$\n)'
        '(^pr .*$\n)?'
        '(?P<weight>^w (Bandwidth=(?P<bandwidth>\d*)).*$\n)'
        '(?P<ports>^p .*$)', re.MULTILINE)

    bridge_pattern = re.compile(
        '(^r (?P<nickname>\S*) (?P<id>\S*) (?P<digest>\S*) (?P<publication>\S* \S*) (?P<ip>\S*) (?P<orport>\S*) (?P<dirport>\S*)$\n)'
        '(^a (?P<ipv6>\S*)$\n)?'
        '(^s (?P<flags>(\S ?)*)$\n)'
        '(?P<weight>^w (Bandwidth=(?P<bandwidth>\d*)).*$\n)'
        '(?P<ports>^p .*$)', re.MULTILINE)

    default_ttl = 70

    def __init__(self,
                 start: datetime.datetime,
                 end: datetime.datetime,
                 folder: str = 'cache',
                 lookahead: int = 1000,
                 bridge: bool = False,
                 keys: List[str] | None = None):
        self._lookahead = lookahead
        self._start = start
        self._end = end
        self._cache = dict()
        self._keys = keys if keys is not None else []
        self._bridge = bridge
        sub_folder = 'bridge-statuses' if self._bridge else 'consensuses'
        self._folder = os.path.join(folder, sub_folder)

    def _load_from_files(self, start: datetime.datetime, end: datetime.datetime):
        """"
        Loads consensus data from disk if they are between start time and end time (plus some lookahead)
        Uses multiprocessing to make it faster
        """
        end = min(self._end, end + relativedelta(hours=self._lookahead))

        times = list()
        while start <= end:
            times.append(start)
            start += relativedelta(hours=1)

        if self._bridge:
            file_name_prefix = "{:%Y%m%d-%H}"
            missing = [file_name for time in times for file_name in
                       find_all_files_from_prefix(file_name_prefix.format(time), self._folder)]
        else:
            filename = "{}-00-00-consensus"
            missing = [c for c in [filename.format(time.strftime('%Y-%m-%d-%H')) for time in times] if
                       c not in self._cache]

        files = [os.path.join(self._folder, item) for item in missing]

        results = [ConsensusReader._get_consensus_data_from_file(file, self._keys, self._bridge) for file in files]

        d = dict(zip([os.path.basename(r) for r in files], results))
        self._cache = self._cache | d

    @staticmethod
    def _get_consensus_data_from_file(file_path: str, keys, bridge: bool) -> dict:
        """
        Read the data from the given file and returns a dict with the keys passed in as argument (see regex on top)
        :param file_path: consensus file to read
        :param keys: keys to keep (w.r.t named groups in the regex on top of the file)
        :return: dict with the properties specified in keys
        """
        if not os.path.exists(file_path):
            return {'relays': list(),
                    'ttl': ConsensusReader.default_ttl,
                    'bandwidth-weights': {}}

        result = dict()
        result['relays'] = list()
        pattern = ConsensusReader.bridge_pattern if bridge else ConsensusReader.pattern
        with open(file_path, 'r') as f:
            for match in pattern.finditer(f.read()):
                if not keys:
                    result['relays'].append(match.groupdict())
                else:
                    result['relays'].append({k: v for k, v in match.groupdict().items() if k in keys})

            # Get the footer weight from the consensus file
            f.seek(0)
            for line in f:
                if line.startswith("bandwidth-weights"):
                    weights_line = set(line.split())
                    weights_line.remove("bandwidth-weights")
                    weights_line = [p.split('=') for p in weights_line]
                    bandwidth_weights = {weight: int(value) for weight, value in weights_line}
        result['bandwidth-weights'] = bandwidth_weights
        result['ttl'] = ConsensusReader.default_ttl

        return result

    def get_relays_timespan(self, start: datetime.datetime, end: datetime.datetime | None = None) -> dict:
        """
        Get a list of relays for the specified timespan.  Uses caching to avoid reading from disk too much
        :param start: start of the timespan (inclusive)
        :param end: end of the timespan (inclusive)
        :return: dict with the keys specified when creating self
        """
        if end is not None and end > self._end:
            raise ValueError("End is after the timespan covered by the provider")
        if start < self._start:
            raise ValueError("Start is before the timespan covered by the provider")
        if end is not None and end < start:
            raise ValueError("Start must be before or equal to end")

        if end is None:
            end = start

        times = list()
        lower = start
        while lower <= end:
            times.append(lower)
            lower += relativedelta(hours=1)

        if self._bridge:
            file_name_prefix = "{:%Y%m%d-%H}"
            consensuses_prefix = [file_name_prefix.format(time) for time in times]
            consensuses = []

            for prefix in consensuses_prefix:
                consensuses.extend(find_all_files_from_prefix(prefix, self._folder))
        else:
            filename = "{}-00-00-consensus"
            consensuses = [filename.format(time.strftime('%Y-%m-%d-%H')) for time in times]

        if len([c for c in consensuses if c not in self._cache]) > 0:
            self._load_from_files(start, end)

        result = {c: v['relays'] for c, v in self._cache.items() if c in consensuses}

        self._cache = {k: v for k, v in self._cache.items() if v['ttl'] > 1}
        for k, v in self._cache.items():
            v['ttl'] -= 1
        return result

    def get_b_weights_timespan(self, time: datetime.datetime) -> dict:
        """
        Get a list of bandwidth-weights for the specified consensus file.
        :param time: start of the timespan (inclusive)
        :return: dict with the keys corresponding to the weights in the footer of the consensus
        """
        if time < self._start or time > self._end:
            raise ValueError("Time is not in the timespan covered by the provider")

        filename = f"{time.strftime('%Y-%m-%d-%H')}-00-00-consensus"

        if filename not in self._cache:
            self._load_from_files(time, time)
        return self._cache.get(filename, {}).get('bandwidth-weights')

    def info(self):
        print(len(self._cache))


@lru_cache(maxsize=1000)
def cached_listdir(folder: str) -> list[str]:
    return os.listdir(folder)


def find_all_files_from_prefix(prefix: str, search_folder: str) -> list[str]:
    """
    Find all files that start with the given prefix
    :param prefix: prefix to look for
    :param search_folder: folder to search for files
    :return: list of files that start with the given prefix
    """
    return [f for f in cached_listdir(search_folder) if
            f.startswith(prefix) and os.path.isfile(os.path.join(search_folder, f))]

