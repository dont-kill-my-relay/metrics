# Compute anonymity metrics for Tor relays

This repository holds the code to compute relay adversary metric and network adversary.

## Install

1. Clone this repository
2. Install the requirements: `pip install -r requirements.txt`

## Usage

In the remaining, we show an example describing how to compute both metrics for the exclusion of May 2023.

## Download the data you need

To compute the metrics, we need consensus and descriptor data. 
Download the data needed using the commands below. 
If the exclusion is at the very beginning of the month, you may need to download the previous month of descriptors as 
well.
In this case, just adapt the beginning of the period to download.

```bash
python scripts/consensus_mining.py relay 2023-05-01T00 2023-05-30T00 cache
python scripts/descriptor_mining.py relay 2023-05-01T00 2023-05-30T00 cache
```

## Relay adversary metric

The relay metric is the more straight forward to compute.
You only need the data downloaded in the previous step, plus the list of excluded fingerprint. 
One fingerprint with upper-case hexadecimal fingerprint per-line as listed on 
[Tor Gitlab, under March/April 2023](https://gitlab.torproject.org/tpo/network-health/team/-/wikis/Relay-EOL-policy#marchapril-2023).

```bash
python scripts/relay_metric_calculator.py 2023-05-15T12 exclusion-list cache
```


## Network adversary metric

The network adversary metric is more involved to compute. 

First, you need to build a list of possible paths between the guards and a sample of clients as well as between the 
exits and a sample of clients. 
The following command generate two files. The first will be fed to the AS inference script.  The second will allow 
further processing to match the path with the correct relay.

The file `2023-05-31-ips.txt``contains the top IP address that we want to clients to connect to. 
Create the file by using the IP you want, or use pre-build lists like: https://tranco-list.eu/

```bash
python scripts/network_adversary_metric.py build-inference 2023-05-15T12 2023-05-31-ips.txt cache
```

Feed the `to_infer-2023-05-15T12.txt` to the AS inference script.  The result of this inference is then used below.

Second, you need to actually compute the metric. 
Use the same time as in the first command (here 2023-05-15T12), the file with the inferred AS path, the mapping produced
by the previous command and the list of excluded fingerprint (same as in relay metric).

```bash
python scripts/network_adversary_metric.py metric 2023-05-15T12 2023-05-15-asinfer.txt mapping_infer-2023-05-15T12.json exclusion-list cache
```
