import multiprocessing
import os
import shutil
from datetime import datetime
from multiprocessing import Pool

import fire

from utils import (
    months_between,
    validate_timespan,
    download_and_untar_xz,
    flatten_dir,
)


def download_descriptors(
        start: datetime,
        base_url: str,
        output_folder: str,
        end: datetime | None = None,
        folder: str = "cache",
        force: bool = False,
):
    """
    Download monthly server descriptor archive from Collector website and uncompress them
    :param start: first timespan to consider
    :param base_url: base url of the archive
    :param output_folder: folder name where the descriptors are download
    :param end: last timespan to consider, defaults to now if None
    :param folder: folder where the descriptors are download, './cache/' by default
    :param force: deletes folder before downloading if it exists
    """
    destination = os.path.join(folder, output_folder)
    if os.path.exists(destination) and os.listdir(destination) and not force:
        print(
            f"Folder {destination} already exists and is not empty, skipping download, use --force to overwrite"
        )
        exit(-1)
    if os.path.exists(destination) and os.listdir(destination) and force:
        print(f"Folder {destination} already exists and is not empty, deleting it")
        shutil.rmtree(destination)

    os.makedirs(destination, exist_ok=True)

    if end is None:
        today = datetime.now()
        end = datetime(today.year, today.month, 1)

    dates = months_between(start, end)

    print(f"About to download {len(dates)} months of descriptors")

    url = base_url + "server-descriptors-{}.tar.xz"

    with Pool(multiprocessing.cpu_count()) as p:
        p.starmap(
            download_and_untar_xz,
            [(url.format(date.strftime("%Y-%m")), destination) for date in dates],
        )

    print("flattening directory")
    folders = [os.path.join(destination, folder) for folder in os.listdir(destination)]
    with Pool(multiprocessing.cpu_count()) as p:
        p.map(flatten_dir, folders)


def download_command(target: str, start: str, end: str | None = None, folder: str = "cache", force: bool = False):
    start, end = validate_timespan(start, end)
    match target:
        case "relay":
            base_url = "https://collector.torproject.org/archive/relay-descriptors/server-descriptors/"
            output_folder = "relay_descriptors"
        case "bridge":
            base_url = "https://collector.torproject.org/archive/bridge-descriptors/server-descriptors/bridge-"
            output_folder = "bridge_descriptors"
        case _:
            print("Invalid target, use 'relay' or 'bridge'")
            exit(-1)
    download_descriptors(start, base_url, output_folder, end, folder, force)


if __name__ == "__main__":
    fire.Fire(download_command)
