import datetime
import multiprocessing
import os
import shutil
from multiprocessing import Pool

import fire

from utils import flatten_dir, validate_timespan, months_between, download_and_untar_xz


def download_consensus(
        start: datetime.datetime,
        base_url: str,
        output_folder: str,
        end: datetime.datetime | None = None,
        folder: str = "cache",
        force: bool = False,
):
    """
    Download monthly consensus archive from Collector website and uncompress them

    :param start: start of the timespan to consider
    :param base_url: base url of the archive
    :param output_folder: folder name where the descriptors are download
    :param end: end of the timespan to consider, defaults to now if None
    :param folder: folder where the consensuses are downloaded (optional)
    :param force: deletes folder before downloading if it exists
    """
    destination = os.path.join(folder, output_folder)
    if os.path.exists(destination) and os.listdir(destination) and not force:
        print(
            f"{destination} is not empty. Not downloading. Use --force to delete and proceed"
        )
        exit(-1)
    if os.path.exists(destination) and os.listdir(destination) and force:
        print(f"{destination} exists but --force is set, deleting folder")
        shutil.rmtree(destination)

    if end is None:
        today = datetime.date.today()
        end = datetime.date(today.year, today.month, 1)
    else:
        end = datetime.date(end.year, end.month, end.day)

    start = datetime.date(start.year, start.month, 1)

    dates = months_between(start, end)
    print(
        f"About to download {len(dates)} monthly archives to {destination} from collector"
    )

    url = base_url + "{}.tar.xz"

    with Pool(multiprocessing.cpu_count()) as p:
        p.starmap(
            download_and_untar_xz,
            list(
                zip(
                    [url.format(date.strftime("%Y-%m")) for date in dates],
                    [destination] * len(dates),
                )
            ),
        )

    flatten_dir(destination)


def download_command(target: str, start: str, end: str | None = None, folder: str = "cache", force: bool = False):
    start, end = validate_timespan(start, end)
    relay_base_url = "https://collector.torproject.org/archive/relay-descriptors/consensuses/consensuses-"
    bridge_base_url = "https://collector.torproject.org/archive/bridge-descriptors/statuses/bridge-statuses-"
    relay_folder = "consensuses"
    bridge_folder = "bridge-statuses"

    match target:
        case "relay":
            download_consensus(start, relay_base_url, relay_folder, end, folder, force)
        case "bridge":
            download_consensus(start, bridge_base_url, bridge_folder, end, folder, force)
        case _:
            print("Invalid target, use 'relay' or 'bridge'")
            exit(-1)


if __name__ == '__main__':
    fire.Fire(download_command)
