"""Validate the shipped calibration catalog independently of local overrides."""
import math
from pathlib import Path

import pytest
import yaml


class UniqueKeysLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate camera configuration key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeysLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def catalog():
    path = Path(__file__).resolve().parents[1] / 'camera_config.yaml'
    return yaml.load(path.read_text(), Loader=UniqueKeysLoader)['cameras']


def test_catalog_has_valid_calibration_and_source_for_every_model():
    for model, cfg in catalog().items():
        width, height, x, y = cfg['default_roi']
        assert all(type(v) is int for v in (width, height, x, y)), model
        assert width > 0 and height > 0 and x >= 0 and y >= 0, model
        pitch = cfg['pixel_size']
        assert isinstance(pitch, float) and math.isfinite(pitch), model
        assert 0 < pitch < 100e-6, model
        assert cfg['source'].startswith((
            'https://softwareservices.flir.com/',
            'https://docs.baslerweb.com/')), model


def test_blackfly_122s6_names_share_the_verified_sensor_calibration():
    models = catalog()
    for name in ('BFS-U3-122S6M', 'BFS-U3-122S6M-C', 'Blackfly S BFS-U3-122S6M'):
        assert models[name]['default_roi'] == [4096, 3000, 0, 0]
        assert models[name]['pixel_size'] == pytest.approx(3.45e-6)


def test_existing_blackfly_output_configuration_is_preserved():
    models = catalog()
    for name in ('BFS-U3-31S4M', 'BFS-U3-31S4M-C', 'Blackfly S BFS-U3-31S4M'):
        assert models[name]['sync'] == {
            'exposure_line': 'Line1', 'user_line': 'Line2',
            'user_output': 'UserOutput0'}
