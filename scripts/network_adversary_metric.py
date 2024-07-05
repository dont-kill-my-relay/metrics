import base64
import json
import multiprocessing
import time
from functools import lru_cache
from itertools import product

import fire
from numba import njit

from consensus_reader import ConsensusReader
from descriptor_reader import DescriptorReader
from utils import validate_timespan


def build_inference_file(consensus_time: str, ip_file: str, folder: str, nb_clients: int = 100):
    """
    Create a file that is similar to TorPS output but tweaked for our purposes. This file should be fed to the AS path
    inference script.
    On top of the TorPS-like, a mapping between the TorPS-like output and the ID of the relays is written to disk.
    :param consensus_time: time of the consensus to use for metric computation e.g. 2023-05-30T12
    :param ip_file: file with destination IPs, one per line, IPv4
    :param folder: folder where the consensus are stored
    :param nb_clients: number of clients to generate for the inference step
    """

    consensus_time, _ = validate_timespan(consensus_time, None)

    have_exit_flag, have_guard_flag, _, _ = get_consensus_data(consensus_time, folder)

    with open(ip_file, 'r') as f:
        destinations = f.readlines()

    destinations = [d.strip() for d in destinations]
    client_guard_pairs = list(product(have_guard_flag, range(nb_clients)))
    exit_destination_pairs = list(product(have_exit_flag, destinations))

    mapping = {
        'client_guard': list(),
        'exit_destination': list()
    }

    with open(f'to_infer-{consensus_time.strftime("%Y-%m-%dT%H")}.txt', 'w') as output:
        output.write('Sample\tTimestamp\tGuard IP\tMiddle IP\tExit IP\tDestination IP\n')
        for i in range(max(len(client_guard_pairs), len(exit_destination_pairs))):
            if i < len(client_guard_pairs):
                guard, client = client_guard_pairs[i]
                mapping['client_guard'].append(guard['id'])
            else:
                guard = {'ip': '0.0.0.0', 'id': 'None'}
                client = -1
            if i < len(exit_destination_pairs):
                exit_, dest = exit_destination_pairs[i]
                mapping['exit_destination'].append(exit_['id'])

            else:
                exit_ = {'ip': '0.0.0.0', 'id': 'None'}
                dest = '0.0.0.0'
            output.write(f"{client}\t{i}\t{guard['ip']}\t{guard['id']}-{exit_['id']}\t{exit_['ip']}\t{dest}\n")
    with open(f'mapping_infer-{consensus_time.strftime("%Y-%m-%dT%H")}.json', 'w') as j:
        json.dump(mapping, j)


def get_consensus_data(consensus_time, folder):
    cr = ConsensusReader(start=consensus_time, end=consensus_time, folder=folder, lookahead=0)
    relays = cr.get_relays_timespan(consensus_time, consensus_time)
    relays = list(relays.values())[0]
    have_guard_flag = [r for r in relays if 'Guard' in r['flags']]
    have_exit_flag = [r for r in relays if 'Exit' in r['flags']]
    footer_weights = cr.get_b_weights_timespan(consensus_time)
    return have_exit_flag, have_guard_flag, relays, footer_weights


def add_or_increment_subkey(data, relay_id, asn):
    if relay_id not in data:
        data[relay_id] = {asn: 1}

    elif asn in data[relay_id]:
        data[relay_id][asn] += 1

    else:
        data[relay_id][asn] = 1


def add_or_increment_subkey_many(data, relay_id, asns):
    for asn in asns:
        add_or_increment_subkey(data, relay_id, asn)


def divide_subkeys(data, divide_by):
    for relay_id, asns in data.items():
        for asn, count in asns.items():
            data[relay_id][asn] = count / divide_by


def extract_pag_pae_from_inference(inferred_file: str, mapping_file: str):
    """
    Once inference is done on the output of build_inference_file(), this function computes the following probabilities:
    - Pr[ag]: probability of finding the AS a before the guard g
    - Pr[ae]: probability of finding the AS a after the exit e

    The function also returns the set of all ASes we've seen in the ingerred file
    """
    nb_clients = 0
    nb_destinations = 100
    all_ases = set()

    pag = dict()
    # pag = {
    #     'guard_id': {
    #         'as1': count,
    #         'as2': count
    #     }
    # }

    pae = dict()
    # pae = {
    #     'exit_id': {
    #         'as1': count,
    #         'as2': count
    #     }
    # }

    with open(mapping_file, 'r') as m:
        mapping = json.load(m)

    with open(inferred_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            client_n, idx, c2g, g2c, e2d, d2e = line.split()

            client_n = int(client_n)
            idx = int(idx)

            # Count ASes between guard and clients
            cg = set(c2g.split('-') + g2c.split('-'))
            if 'None' in cg:
                cg.remove('None')
            if idx < len(mapping['client_guard']):
                add_or_increment_subkey_many(pag, mapping['client_guard'][idx], cg)

            # Count ASes between exit and destinations
            ed = set(e2d.split('-') + d2e.split('-'))
            if 'None' in ed:
                ed.remove('None')
            if idx < len(mapping['exit_destination']):
                add_or_increment_subkey_many(pae, mapping['exit_destination'][idx], ed)

            nb_clients = max(client_n, nb_clients)
            all_ases.update(cg)
            all_ases.update(ed)

    nb_clients += 1

    # Divide count to get probability
    # If a path could not be inferred for ASes, then it is considered as magic black hole: there is a path between
    # the node and the other end without any ASes (not even the source and dest ASes).  This could be changed if needed
    divide_subkeys(pag, nb_clients)
    divide_subkeys(pae, nb_destinations)

    return pag, pae, all_ases


@njit
def bw_e(relay_bw: int, relay_flags: str, wee: int, wed: int) -> float:
    bw = relay_bw / 1_000
    if is_exit_only(relay_flags):
        return bw * wee
    elif is_dual(relay_flags):
        return bw * wed
    else:
        return 0


@njit
def bw_g(relay_bw: int, relay_flags: str, wgg: int, wgd: int) -> float:
    bw = relay_bw / 1_000
    if is_guard_only(relay_flags):
        return bw * wgg
    elif is_dual(relay_flags):
        return bw * wgd
    else:
        return 0


@njit
def is_guard_only(relay: str) -> bool:
    return 'Guard' in relay and 'Exit' not in relay


@njit
def is_exit_only(relay: str) -> bool:
    return 'Guard' not in relay and 'Exit' in relay


@njit
def is_dual(relay: str) -> bool:
    return 'Guard' in relay and 'Exit' in relay


def reachable_from(source_relay: dict, destination_list: list) -> list:
    """
    From a source relay descriptor and a list of destination descriptors, return the subset of the destinations that
    can be reached by the source by checking the subnet and family restrictions
    """
    source_subnet = source_relay['address'].split('.')[:2]
    source_family = {x.replace(' ', '').upper() for x in list(source_relay['family']) + [source_relay['fingerprint']]}
    source_id = source_relay['fingerprint'].replace(' ', '').upper()
    result = list()

    for dest_descriptor in destination_list:
        dest_subnet = dest_descriptor['address'].split('.')[:2]
        dest_family = {x.replace(' ', '').upper() for x in
                       list(dest_descriptor['family']) + [dest_descriptor['fingerprint']]}
        dest_id = dest_descriptor['fingerprint'].replace(' ', '').upper()

        if source_id not in dest_family and \
                dest_id not in source_family and \
                source_subnet != dest_subnet:
            result.append(dest_descriptor)

    return result


def compute_network_adversary_metric(
        consensus_time: str,
        inferred_file: str,
        mapping_file: str,
        exclusion_file: str,
        folder: str):
    consensus_time, _ = validate_timespan(consensus_time, None)
    have_exit_flag, have_guard_flag, all_relays, footer_weights = get_consensus_data(consensus_time, folder)

    pag, pae, all_ases = extract_pag_pae_from_inference(inferred_file, mapping_file)

    dr = DescriptorReader(start=consensus_time, end=consensus_time, folder=folder)
    have_exit_flag = [dr.get_descriptor(base64.b64decode(r['digest'] + '=').hex().lower(), consensus_time) | r for r in
                      have_exit_flag]

    have_guard_flag = [dr.get_descriptor(base64.b64decode(r['digest'] + '=').hex().lower(), consensus_time) | r for r in
                       have_guard_flag]

    with open(exclusion_file, 'r') as f:
        excluded_fprints = [line.strip() for line in f.readlines()]

    excluded_relays = [r for r in all_relays if base64.b64decode(r['id'] + '=').hex().upper() in excluded_fprints]
    excluded_relays = [dr.get_descriptor(base64.b64decode(r['digest'] + '=').hex().lower(), consensus_time) | r for r in
                       excluded_relays]

    print(f"Found {len(excluded_relays)} out of the {len(excluded_fprints)} excluded fingerprints")
    print(f"{len(all_ases)} ASes")
    result_metrics = dict()

    with multiprocessing.Pool(14) as pool:
        pool_results = pool.starmap(compute_metric_for_relay,
                                    ((r, have_exit_flag, have_guard_flag, all_ases, footer_weights, pae, pag)
                                     for r in excluded_relays))
    for r in pool_results:
        result_metrics |= r

    print("Saving results to file")
    with open(f'net_adv_metric-{consensus_time.strftime("%Y-%m-%dT%H")}.json', 'w') as f:
        json.dump(result_metrics, f)


def compute_metric_for_relay(relay, have_exit_flag, have_guard_flag, all_ases, footer_weights, pae, pag):
    result_metrics = dict()
    start = time.time()
    if is_guard_only(relay['flags']):
        print(f"Guard {relay['nickname']}")
        m = metric_guard(
            current_guard=relay,
            all_ases=all_ases,
            have_exit_flag=have_exit_flag,
            footer_weights=footer_weights,
            pae=pae,
            pag=pag
        )
        result_metrics[relay['id']] = m
        print(f"Done guard {relay['nickname']} in {round(time.time() - start, 2)} s")

    elif is_exit_only(relay['flags']):
        print(f"Exit {relay['nickname']}")
        m = metric_exit(
            current_exit=relay,
            all_ases=all_ases,
            have_exit_flag=have_exit_flag,
            have_guard_flags=have_guard_flag,
            footer_weights=footer_weights,
            pae=pae,
            pag=pag
        )
        result_metrics[relay['id']] = m
        print(f"Done exit {relay['nickname']} in {round(time.time() - start, 2)} s")

    elif is_dual(relay['flags']):
        print(f"Dual {relay['nickname']}")
        m = metric_dual(
            current_dual=relay,
            all_ases=all_ases,
            have_exit_flag=have_exit_flag,
            have_guard_flags=have_guard_flag,
            footer_weights=footer_weights,
            pae=pae,
            pag=pag
        )
        result_metrics[relay['id']] = m
        print(f"Done dual {relay['nickname']} in {round(time.time() - start, 2)} s")
    else:
        print(f"Nothing to do for {relay['nickname']}")
    return result_metrics


def metric_guard(current_guard, all_ases, have_exit_flag, footer_weights, pae, pag) -> float:
    bw_guard = bw_g(int(current_guard['bandwidth']), current_guard['flags'], footer_weights['Wgg'],
                    footer_weights['Wgd'])

    sum_ = 0
    for as_ in all_ases:
        pag_value = pag.get(current_guard['id'], {}).get(as_, 0)
        if pag_value != 0:
            paeg_value = paeg(current_guard=current_guard,
                              have_exit_flag=have_exit_flag,
                              footer_weights=footer_weights,
                              current_as=as_,
                              pae=pae)

            sum_ += 1 - (pag_value * paeg_value)
        else:
            sum_ += 1

    return bw_guard * sum_


def metric_exit(current_exit, all_ases, have_exit_flag, have_guard_flags, footer_weights, pae, pag) -> float:
    bw_exit = bw_e(int(current_exit['bandwidth']), current_exit['flags'], wee=footer_weights['Wee'],
                   wed=footer_weights['Wed'])

    sum_ = 0
    for as_ in all_ases:
        pae_value = pae.get(current_exit['id'], {}).get(as_, 0)
        if pae_value != 0:
            page_value = page(current_exit=current_exit,
                              have_guard_flag=have_guard_flags,
                              have_exit_flag=have_exit_flag,
                              footer_weights=footer_weights,
                              current_as=as_,
                              pag=pag)

            sum_ += 1 - (pae_value * page_value)
        else:
            sum_ += 1

    return bw_exit * sum_


def metric_dual(current_dual, all_ases, have_exit_flag, have_guard_flags, footer_weights, pae, pag) -> float:
    metric_dual_as_guard = metric_guard(current_guard=current_dual,
                                        all_ases=all_ases,
                                        have_exit_flag=have_exit_flag,
                                        footer_weights=footer_weights,
                                        pae=pae,
                                        pag=pag)
    metric_dual_as_exit = metric_exit(current_exit=current_dual,
                                      all_ases=all_ases,
                                      have_exit_flag=have_exit_flag,
                                      have_guard_flags=have_guard_flags,
                                      footer_weights=footer_weights,
                                      pae=pae,
                                      pag=pag)
    return metric_dual_as_guard + metric_dual_as_exit


@lru_cache(maxsize=4_000)
def peg(current_exit_bw: int, current_exit_flags: str, wee: int, wed: int, in_reach: tuple) -> float:
    bw_exit = bw_e(current_exit_bw, current_exit_flags, wee=wee, wed=wed)

    sum_bw_e_e = peg_sum_fast(list(in_reach), wee=wee, wed=wed)
    return bw_exit / sum_bw_e_e


@njit
def peg_sum_fast(in_reach, wee, wed):
    return sum([bw_e(bw, flags, wee=wee, wed=wed) for bw, flags in in_reach])


@lru_cache(maxsize=4_000)
def pg(current_guard_bw: int, current_guard_flags: str, have_guard_flag: tuple, wgg: int, wgd: int) -> float:
    bw_guard = bw_g(current_guard_bw, current_guard_flags, wgg, wgd)
    sum_bw_g_g = pg_sum_fast(list(have_guard_flag), wgd, wgg)
    return bw_guard / sum_bw_g_g


@njit
def pg_sum_fast(have_guard_flag, wgd, wgg):
    return sum([bw_g(bw, flags, wgg, wgd) for bw, flags in have_guard_flag])


@lru_cache(maxsize=4_000)
def pe(current_exit: tuple, have_exit_flag: tuple, have_guard_flag: tuple, footer_weights: tuple) -> float:
    have_exit_flag = [dict(v) for v in have_exit_flag]
    have_guard_flag = [dict(v) for v in have_guard_flag]

    footer_weights = dict(footer_weights)
    current_exit = dict(current_exit)

    result = 0
    have_guard_flag_tuple = [(int(g['bandwidth']), g['flags']) for g in have_guard_flag]
    for guard in reachable_from(current_exit, have_guard_flag):
        pg_value = pg(int(guard['bandwidth']), guard['flags'], tuple(have_guard_flag_tuple), wgg=footer_weights['Wgg'],
                      wgd=footer_weights['Wgd'])

        in_reach = [(int(r['bandwidth']), r['flags']) for r in reachable_from(guard, have_exit_flag)]
        peg_value = peg(current_exit_bw=int(current_exit['bandwidth']), current_exit_flags=current_exit['flags'],
                        wee=footer_weights['Wee'], wed=footer_weights['Wed'], in_reach=tuple(in_reach))

        result += (pg_value * peg_value)
    return result


@lru_cache(maxsize=4_000)
def pge(current_guard_t: tuple, current_exit_t: tuple, have_guard_flag_t: tuple, have_exit_flag_t: tuple,
        footer_weights_t: tuple) -> float:
    footer_weights = dict(footer_weights_t)
    current_guard = dict(current_guard_t)
    current_exit = dict(current_exit_t)
    have_exit_flag = [dict(v) for v in have_exit_flag_t]
    have_guard_flag = [dict(v) for v in have_guard_flag_t]

    in_reach = [(int(r['bandwidth']), r['flags']) for r in reachable_from(current_guard, have_exit_flag)]
    peg_value = peg(current_exit_bw=int(current_exit['bandwidth']), current_exit_flags=current_exit['flags'],
                    wee=footer_weights['Wee'], wed=footer_weights['Wed'], in_reach=tuple(in_reach))

    have_guard_flag_tuple = [(int(g['bandwidth']), g['flags']) for g in have_guard_flag]
    pg_value = pg(int(current_guard['bandwidth']), current_guard['flags'], tuple(have_guard_flag_tuple),
                  wgg=footer_weights['Wgg'], wgd=footer_weights['Wgd'])

    pe_value = pe(current_exit_t, have_exit_flag_t, have_guard_flag_t, tuple(footer_weights.items()))
    return (peg_value * pg_value) / pe_value


def cast_dict(data: list[dict]) -> tuple:
    result = list()
    for d in data:
        result.append((('fingerprint', d['fingerprint']),
                       ('address', d['address']),
                       ('family', tuple(d['family'])),
                       ('bandwidth', d['bandwidth']),
                       ('flags', d['flags'])))
    return tuple(result)


def cast_single_relay(relay: dict) -> tuple:
    return (('fingerprint', relay['fingerprint']),
            ('address', relay['address']),
            ('family', tuple(relay['family'])),
            ('bandwidth', relay['bandwidth']),
            ('flags', relay['flags']))


def page(current_exit, have_guard_flag, have_exit_flag, footer_weights, current_as, pag) -> float:
    result = 0
    for guard in have_guard_flag:
        pag_value = pag.get(guard['id'], {}).get(current_as, 0)
        if pag_value != 0:
            pge_value = pge(current_guard_t=cast_single_relay(guard),
                            current_exit_t=cast_single_relay(current_exit),
                            have_guard_flag_t=cast_dict(have_guard_flag),
                            have_exit_flag_t=cast_dict(have_exit_flag),
                            footer_weights_t=tuple(footer_weights.items())
                            )
            result += (pge_value * pag_value)
    return result


def paeg(current_guard, have_exit_flag, footer_weights, current_as, pae) -> float:
    result = 0
    in_reach = [(int(r['bandwidth']), r['flags']) for r in reachable_from(current_guard, have_exit_flag)]

    for exit_ in have_exit_flag:
        pae_value = pae.get(exit_['id'], {}).get(current_as, 0)
        if pae_value != 0:
            peg_value = peg(current_exit_bw=int(exit_['bandwidth']), current_exit_flags=exit_['flags'],
                            wee=footer_weights['Wee'], wed=footer_weights['Wed'], in_reach=tuple(in_reach))
            result += (peg_value * pae_value)
    return result


if __name__ == '__main__':
    fire.Fire({
        'build-inference': build_inference_file,
        'metric': compute_network_adversary_metric,
    })
