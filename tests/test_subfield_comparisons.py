import io
import operator
import struct

import numpy as np
import pytest

import laspy
from laspy.point.dims import SubFieldView


@pytest.mark.parametrize("mask", [1, 2, 4, 8, 7, 15, 31, 16, 32, 48, 56, 64, 128, 240])
@pytest.mark.parametrize(
    "compare", [operator.lt, operator.le, operator.ge, operator.gt]
)
@pytest.mark.parametrize(
    "scalar_type",
    [
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_subfield_integer_thresholds(mask, compare, scalar_type):
    # Decompose bytes independently of SubFieldView's masking implementation.
    packed = np.arange(256, dtype=np.uint8)
    original = packed.copy()
    shift = (mask & -mask).bit_length() - 1
    maximum = mask >> shift
    decoded = [int(byte) // (2**shift) % (maximum + 1) for byte in packed]
    thresholds = [-129, -2, -1, 0, 1, maximum, maximum + 1, 127, 128, 255, 256]
    if scalar_type is int:
        thresholds.extend([-(2**100), 2**100])
    else:
        limits = np.iinfo(scalar_type)
        thresholds.extend([int(limits.min), int(limits.max)])
        thresholds = [t for t in thresholds if limits.min <= t <= limits.max]

    field = SubFieldView(packed, mask)
    for threshold in thresholds:
        expected = [compare(value, threshold) for value in decoded]
        actual = compare(field, scalar_type(threshold))
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.dtype(bool)
    np.testing.assert_array_equal(packed, original)


@pytest.mark.parametrize(
    "compare", [operator.lt, operator.le, operator.ge, operator.gt]
)
@pytest.mark.parametrize("threshold", [-2, 0, 1, 2, 256])
def test_subfield_comparison_preserves_shape(compare, threshold):
    packed = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
    field = SubFieldView(packed, 128)
    expected = np.array(
        [[compare(int(byte) // 128, threshold) for byte in row] for row in packed]
    )
    np.testing.assert_array_equal(compare(field, np.int16(threshold)), expected)
    assert compare(field, threshold).shape == packed.shape
    assert compare(SubFieldView(packed[:0], 128), threshold).shape == (0, 3)
    scalar = SubFieldView(np.array(128, dtype=np.uint8), 128)
    assert bool(compare(scalar, threshold)) == compare(1, threshold)


def test_subfield_array_threshold_and_equality_controls():
    packed = np.arange(256, dtype=np.uint8)
    field = SubFieldView(packed, 48)
    decoded = np.array([int(byte) // 16 % 4 for byte in packed])
    thresholds = np.arange(256, dtype=np.int64) % 6 - 1
    for compare in [operator.lt, operator.le, operator.ge, operator.gt]:
        np.testing.assert_array_equal(
            compare(field, thresholds), compare(decoded, thresholds)
        )
    np.testing.assert_array_equal(field == 2, decoded == 2)
    np.testing.assert_array_equal(field != 2, decoded != 2)


@pytest.mark.parametrize("point_format", [0, 3, 6, 10])
@pytest.mark.parametrize("scalar_type", [int, np.uint8, np.int64])
def test_subfield_filter_keeps_points_below_exclusive_domain_bound(
    point_format, scalar_type
):
    las = laspy.create(point_format=point_format)
    las.X = [11, 22, 33, 44]
    las.return_number = [1, 2, 3, 1]
    las.number_of_returns = [1, 3, 3, 2]
    las.point_source_id = [51, 52, 53, 54]
    bound = 8 if point_format < 6 else 16
    filtered = las[las.number_of_returns < scalar_type(bound)]
    destination = io.BytesIO()
    filtered.write(destination)
    data = destination.getvalue()
    # Inspect standard little-endian point count and X bytes without laspy's reader.
    count_offset = 107 if point_format < 6 else 247
    count_format = "<I" if point_format < 6 else "<Q"
    assert struct.unpack_from(count_format, data, count_offset)[0] == 4
    offset = struct.unpack_from("<I", data, 96)[0]
    size = struct.unpack_from("<H", data, 105)[0]
    assert [struct.unpack_from("<i", data, offset + i * size)[0] for i in range(4)] == [
        11,
        22,
        33,
        44,
    ]
    destination.seek(0)
    reread = laspy.read(destination)
    np.testing.assert_array_equal(reread.points.array, las.points.array)
