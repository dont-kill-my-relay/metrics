import base64
import datetime
import json
import fire
from consensus_reader import ConsensusReader
from descriptor_reader import DescriptorReader


def calculate_relay_weights_timespan(
        date: datetime.datetime,
        excluded_ids: str,
        folder: str = "cache"
) -> dict:
    cr = ConsensusReader(start=date, end=date, keys=["id", "digest", "flags", "ip", "bandwidth", "weight"],
                         folder=folder)
    dr = DescriptorReader(start=date, end=date, folder=folder)

    with open(excluded_ids, 'r') as file:
        excluded_list = [line.strip() for line in file]

    b_weights = cr.get_b_weights_timespan(date)

    consensuses_files: dict = cr.get_relays_timespan(date, date)
    if len(consensuses_files) > 1:
        raise ValueError("Only one consensus file at a time is supported")

    return calculate_relay_weights(consensuses_files, excluded_list, b_weights, dr, date)


def calculate_relay_weights(con_file: dict, excluded_list: list, b_weights: dict, dr: DescriptorReader,
                            date: datetime.datetime) -> dict:
    g_weights = dict()
    e_weights = dict()
    d_weights = dict()
    excluded_g = list()
    excluded_e = list()
    excluded_d = list()

    all_g = list()
    all_e = list()

    metric = {}
    relays = list(con_file.values())[0]
    for r in relays:
        if is_guard_only(r):
            g_weights[r['id']] = int(r['bandwidth']) * (b_weights['Wgg'] / 10000)
            all_g.append(r)
            if base64.b64decode(r['id'] + '=').hex().upper() in excluded_list:
                excluded_g.append(r)
        if is_exit_only(r):
            e_weights[r['id']] = int(r['bandwidth']) * (b_weights['Wee'] / 10000)
            all_e.append(r)
            if base64.b64decode(r['id'] + '=').hex().upper() in excluded_list:
                excluded_e.append(r)
        if is_middle_only(r) and base64.b64decode(r['id'] + '=').hex().upper() in excluded_list:
            metric[r['id']] = 0
        if is_dual(r):
            d_weights[r['id']] = {'guard': int(r['bandwidth']) * (b_weights['Wgd'] / 10000),
                                  'exit': int(r['bandwidth']) * (b_weights['Wed'] / 10000)}
            if base64.b64decode(r['id'] + '=').hex().upper() in excluded_list:
                excluded_d.append(r)

    g_band_sum = sum_weights(g_weights) + sum_weights({d: b['guard'] for d, b in d_weights.items()})
    e_band_sum = sum_weights(e_weights) + sum_weights({d: b['exit'] for d, b in d_weights.items()})

    # Build the list of excluded guards and their possible exits and of excluded exits and their possible guards
    g_e_filtered = filter_relays(excluded_g, all_e, dr, date)
    e_g_filtered = filter_relays(excluded_e, all_g, dr, date)
    gd_e_filtered = filter_relays(excluded_d, all_e, dr, date)
    ed_g_filtered = filter_relays(excluded_d, all_g, dr, date)

    # Calculate relay metrics for guards
    for g, exits in g_e_filtered.items():
        sum = 0
        for e in exits:
            sum += e_weights[e] / e_band_sum
        metric[g] = int(1e9 * ((g_weights[g] / g_band_sum) * sum))

    # Calculate relay metrics for guards
    for e, guards in e_g_filtered.items():
        sum = 0
        for g in guards:
            sum += g_weights[g] / g_band_sum
        metric[e] = int(1e9 * ((e_weights[e] / e_band_sum) * sum))

    for d in excluded_d:
        sum_e = 0
        for e in gd_e_filtered[d['id']]:
            sum_e += e_weights[e] / e_band_sum
        metric[d['id']] = int(1e9 * ((d_weights[d['id']]['guard'] / g_band_sum) * sum_e))

        sum_g = 0
        for g in ed_g_filtered[d['id']]:
            sum_g += g_weights[g] / g_band_sum
        metric[d['id']] += int(1e9 * ((d_weights[d['id']]['exit'] / e_band_sum) * sum_g))

    return metric


def is_dual(r):
    return "Guard" in r["flags"] and "Exit" in r["flags"]


def is_middle_only(relay):
    return "Guard" not in relay["flags"] and "Exit" not in relay["flags"]


def is_exit_only(relay):
    return "Exit" in relay["flags"] and "Guard" not in relay["flags"]


def is_guard_only(relay):
    return "Guard" in relay["flags"] and "Exit" not in relay["flags"]


def sum_weights(weights: dict) -> int:
    g_band_sum = 0
    for w in weights.values():
        g_band_sum += w
    return g_band_sum


def filter_relays(source_list: list, dest_list: list, dr: DescriptorReader, date: datetime.datetime) -> dict:
    source_dest_filtered = dict()
    source_descriptors = {
        s['id']: dr.get_descriptor(base64.b64decode(s['digest'] + '=').hex().lower(),
                                   date,
                                   keys=('family',))
        for s in source_list
    }

    source_families = {
        s: set(descriptor['family']) if descriptor is not None else set()

        for s, descriptor in source_descriptors.items()
    }

    for s in source_list:
        filtered_dest = list()
        for d in dest_list:
            if base64.b64decode(d['id'] + '=').hex().lower() not in source_families[s['id']] and can_connect(s, d):
                filtered_dest.append(d['id'])
        source_dest_filtered[s['id']] = filtered_dest

    return source_dest_filtered


def can_connect(guard, ext) -> bool:
    # IP address and subnet mask
    g_ip_address = guard['ip']
    e_ip_address = ext['ip']

    # Split the IP address and subnet mask
    g_ip_parts = g_ip_address.split('.')[:2]
    e_ip_parts = e_ip_address.split('.')[:2]
    return not g_ip_parts == e_ip_parts


def calculate_command(date: str, excluded_list: str, folder: str = 'cache'):
    date = datetime.datetime.strptime(str(date), "%Y-%m-%dT%H")
    result = calculate_relay_weights_timespan(date, excluded_list, folder)

    print("Saving result to file")
    with open(f'relay_metric_{date.strftime("%Y-%m-%dT%H")}.json', 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    fire.Fire(calculate_command)
