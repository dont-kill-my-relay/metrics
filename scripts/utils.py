import datetime
import io
import itertools
import os
import shutil
import tarfile

import requests
from dateutil.relativedelta import relativedelta


def delta_between(start: datetime.datetime, end: datetime.datetime, delta: relativedelta) -> list[datetime.datetime]:
    """
    Generate the list of dates between two dates (start and end inclusive) with a given delta
    :param start: first date
    :param end: last date
    :param delta: delta between two dates
    :return: list of dates between start and end
    """
    assert start <= end
    dates = []
    while start <= end:
        dates.append(start)
        start += delta

    if dates[-1] != end:
        dates.append(end)

    return dates


def months_between(start: datetime.datetime, end: datetime.datetime) -> list[datetime.date]:
    """
    Generate the list of months between two dates (start and end inclusive)
    :param start: first month
    :param end: last month
    :return: list of months between start and end
    """
    return list({datetime.date(year=d.year, month=d.month, day=1) for d
                 in delta_between(start, end, relativedelta(months=1))})

def validate_timespan(start, end) -> (datetime.datetime, datetime.datetime | None):
    try:
        start = datetime.datetime.strptime(str(start), "%Y-%m-%dT%H")
        end = (
            datetime.datetime.strptime(str(end), "%Y-%m-%dT%H")
            if end is not None
            else None
        )
    except ValueError as err:
        print(str(err))
        exit(-1)
    if end is not None and start > end:
        print("start must be before end")
        exit(-1)
    return start, end


def download_and_untar_xz(url: str, cache: str):
    """
    Utility function to download the file at url and untar it
    :param url: url where to get the file
    :param cache: path where to decompress the file
    """
    response = requests.get(url)
    if response.status_code == 200:
        f = io.BytesIO(response.content)
        tar = tarfile.open(fileobj=f)
        tar.extractall(cache)


def flatten_dir(path: str):
    """
    Utility function to flatten a directory
    :param path: path to the directory to flatten
    """
    all_files = []
    for root, _dirs, files in itertools.islice(os.walk(path), 1, None):
        for filename in files:
            all_files.append(os.path.join(root, filename))
    for filename in all_files:
        shutil.move(filename, path)
    for directory in [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]:
        shutil.rmtree(directory)
