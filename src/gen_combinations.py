#!/usr/bin/env python3

import argparse
import itertools
import logging
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Tuple

import yaml

from utils import NamedDict, setup_logging


class Combination:
    def __init__(self, name: str, pairs: Dict[str, object]):
        self.name = name
        self.pairs = pairs

    def __repr__(self) -> str:
        return f"Combination(name={self.name}, pairs={self.pairs})"


class Placeholder:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "<placeholder>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def get_combinations(combi_config: Dict[str, object]):
    found_combis = []

    def find_recursive(
        partial_config: Dict[str, object] | List[object],
    ) -> List[Tuple[str, Tuple[str, ...]]]:
        values = partial_config.values() if isinstance(partial_config, dict) else partial_config
        for value in values:
            if isinstance(value, Combination):
                found_combis.append(value)
                find_recursive(value.pairs)
            elif isinstance(value, (dict, list)):
                find_recursive(value)

    find_recursive(combi_config)
    name_key_combis = {(combi.name, tuple(combi.pairs.keys())) for combi in found_combis}
    assert len(name_key_combis) == len({name for name, _ in name_key_combis}), (
        f"Combination names must be unique across different key sets. Found: \n{pformat(name_key_combis)}"
    )
    return list(name_key_combis)


def set_combination(combi_config: Dict[str, object], combi_dict: Dict[str, str]):
    def set_recursive(partial_config: Dict[str, object] | List[object]):
        if isinstance(partial_config, dict):
            for key, value in partial_config.items():
                if isinstance(value, Combination):
                    set_recursive(value.pairs)
                    selected_key = combi_dict[value.name]
                    partial_config[key] = value.pairs[selected_key]
                elif isinstance(value, (dict, list)):
                    set_recursive(value)
        elif isinstance(partial_config, list):
            for idx, value in enumerate(partial_config):
                if isinstance(value, Combination):
                    set_recursive(value.pairs)
                    selected_key = combi_dict[value.name]
                    partial_config[idx] = value.pairs[selected_key]
                elif isinstance(value, (dict, list)):
                    set_recursive(value)

    combi_config_copy = deepcopy(combi_config)
    set_recursive(combi_config_copy)
    return combi_config_copy


def purge_placeholder(combi_config: Dict[str, object]):
    def purge_recursive(partial_config: Dict[str, object] | List[object]):
        if isinstance(partial_config, dict):
            to_purge = []
            for key, value in partial_config.items():
                if isinstance(value, Placeholder):
                    to_purge.append(key)
                elif isinstance(value, (dict, list)):
                    purge_recursive(value)
            for key in to_purge:
                del partial_config[key]
        elif isinstance(partial_config, list):
            to_purge = []
            for idx, value in enumerate(partial_config):
                if isinstance(value, Placeholder):
                    to_purge.append(idx)
                elif isinstance(value, (dict, list)):
                    purge_recursive(value)
            for idx in reversed(to_purge):
                del partial_config[idx]
    purge_recursive(combi_config)
    return combi_config


def main():
    setup_logging()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    yaml.add_constructor(
        "!combination",
        lambda loader, node: Combination(**loader.construct_mapping(node)),
        Loader=yaml.SafeLoader,
    )

    yaml.add_constructor(
        "!placeholder",
        lambda loader, node: Placeholder(),
        Loader=yaml.SafeLoader,
    )

    combi_config = NamedDict.from_yaml(args.infile).to_dict()
    combinations = sorted(get_combinations(combi_config), key=lambda x: x[0])
    combination_keys, combination_values = zip(*combinations)

    for combi_value in itertools.product(*combination_values):
        combi_dict = dict(zip(combination_keys, combi_value))
        config = NamedDict.from_dict(
            purge_placeholder(
                set_combination(
                    combi_config, combi_dict
        )))
        config_name = "-".join(f"{k}={v}" for k, v in zip(combination_keys, combi_value))
        outpath = args.outdir / f"{config_name}.yaml"
        config.to_yaml(outpath)
        logging.info(f"Wrote config to {outpath}")
    logging.info("Done generating config combinations.")


if __name__ == "__main__":
    main()
