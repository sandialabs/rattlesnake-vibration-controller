import numpy as np

# region Templates
_NETCDF_SKIP_ATTRIBUTES = {"control_python_script"}


def diff_netcdf_groups(original_group, saved_group, prefix=""):
    """
    Recursively compares attributes, variables, and subgroups between two
    netCDF4 groups, returning a list of human-readable differences.

    Attributes that only appear in ``saved_group`` are ignored, since example
    files may predate fields that were added later (the loaders fall back to
    a default in that case). Attributes only in ``original_group`` indicate
    real data loss and are reported.
    """
    differences = []

    original_attrs = set(original_group.ncattrs()) - _NETCDF_SKIP_ATTRIBUTES
    saved_attrs = set(saved_group.ncattrs()) - _NETCDF_SKIP_ATTRIBUTES

    for name in sorted(original_attrs - saved_attrs):
        differences.append(f"{prefix}{name}: present in example, missing after save")

    for name in sorted(original_attrs & saved_attrs):
        original_value = getattr(original_group, name)
        saved_value = getattr(saved_group, name)
        if isinstance(original_value, np.ndarray) or isinstance(
            saved_value, np.ndarray
        ):
            equal = np.array_equal(original_value, saved_value, equal_nan=True)
        else:
            equal = original_value == saved_value
        if not equal:
            differences.append(f"{prefix}{name}: {original_value!r} != {saved_value!r}")

    original_vars = set(original_group.variables)
    saved_vars = set(saved_group.variables)
    for name in sorted(original_vars ^ saved_vars):
        differences.append(f"{prefix}variables/{name}: only present in one file")

    for name in sorted(original_vars & saved_vars):
        original_data = np.ma.filled(original_group.variables[name][...], np.nan)
        saved_data = np.ma.filled(saved_group.variables[name][...], np.nan)
        if not np.array_equal(original_data, saved_data, equal_nan=True):
            differences.append(f"{prefix}variables/{name}: values differ")

    original_groups = set(original_group.groups)
    saved_groups = set(saved_group.groups)
    for name in sorted(original_groups & saved_groups):
        differences.extend(
            diff_netcdf_groups(
                original_group.groups[name],
                saved_group.groups[name],
                prefix=f"{prefix}{name}/",
            )
        )
    for name in sorted(original_groups ^ saved_groups):
        differences.append(f"{prefix}groups/{name}: only present in one file")

    return differences


def _normalize_cell_value(value):
    """Treats a blank cell and an empty string as equivalent."""
    return "" if value is None else value


def _is_comment_cell(value):
    """
    Identifies human-readable documentation cells (e.g. '# ...') that are
    regenerated fresh by ``save_metadata_to_worksheet`` and aren't expected to
    match older example files that predate a given field.
    """
    return isinstance(value, str) and value.strip().startswith("#")


def diff_worksheets(original_worksheet, saved_worksheet):
    """
    Compares two worksheets cell by cell, returning a list of
    ``(row, col, original_value, saved_value)`` tuples for values that differ.

    Documentation/comment cells are ignored, as are rows whose label mentions
    a "script" or "file", since those are always overwritten with an absolute
    path after loading rather than sourced from the worksheet's saved value.
    """
    differences = []
    max_row = max(original_worksheet.max_row, saved_worksheet.max_row)
    max_col = max(original_worksheet.max_column, saved_worksheet.max_column)

    for row in range(1, max_row + 1):
        row_label = str(
            original_worksheet.cell(row, 1).value
            or saved_worksheet.cell(row, 1).value
            or ""
        ).lower()
        if "script" in row_label or "file" in row_label:
            continue
        for col in range(1, max_col + 1):
            original_value = _normalize_cell_value(
                original_worksheet.cell(row, col).value
            )
            saved_value = _normalize_cell_value(saved_worksheet.cell(row, col).value)
            if _is_comment_cell(original_value) or _is_comment_cell(saved_value):
                continue
            if original_value != saved_value:
                differences.append((row, col, original_value, saved_value))

    return differences


# endregion


# region Metadata
def _is_plain_object(value):
    """True for objects with their own attribute dict (excludes enums, which
    also expose __dict__ but should be compared by value, not recursed into)."""
    return hasattr(value, "__dict__") and not isinstance(value, type)


def diff_metadata_objects(original, saved, prefix=""):
    """
    Recursively compares the attributes of two metadata objects (e.g. two
    ``HardwareMetadata``/``EnvironmentMetadata`` instances), returning a list
    of human-readable differences. Metadata classes don't implement
    ``__eq__``, so comparison is attribute-based via ``vars()``.

    Intended to be used alongside ``diff_netcdf_groups``/``diff_worksheets``
    to isolate save/load bugs: use this to see which attribute changed across
    a round trip, then the file-diffing functions to see whether the value
    was ever written correctly.
    """
    differences = []

    original_attrs = vars(original)
    saved_attrs = vars(saved)

    for name in sorted(set(original_attrs) ^ set(saved_attrs)):
        differences.append(f"{prefix}{name}: present in one object only")

    for name in sorted(set(original_attrs) & set(saved_attrs)):
        original_value = original_attrs[name]
        saved_value = saved_attrs[name]
        child_prefix = f"{prefix}{name}."

        if isinstance(original_value, np.ndarray) or isinstance(
            saved_value, np.ndarray
        ):
            if not np.array_equal(original_value, saved_value, equal_nan=True):
                differences.append(
                    f"{prefix}{name}: {original_value!r} != {saved_value!r}"
                )
        elif isinstance(original_value, (list, tuple)) and isinstance(
            saved_value, (list, tuple)
        ):
            if len(original_value) != len(saved_value):
                differences.append(
                    f"{prefix}{name}: length {len(original_value)} != {len(saved_value)}"
                )
            else:
                for index, (original_item, saved_item) in enumerate(
                    zip(original_value, saved_value)
                ):
                    if _is_plain_object(original_item) and _is_plain_object(saved_item):
                        differences.extend(
                            diff_metadata_objects(
                                original_item,
                                saved_item,
                                prefix=f"{child_prefix}{index}.",
                            )
                        )
                    elif original_item != saved_item:
                        differences.append(
                            f"{child_prefix}{index}: {original_item!r} != {saved_item!r}"
                        )
        elif _is_plain_object(original_value) and _is_plain_object(saved_value):
            differences.extend(
                diff_metadata_objects(original_value, saved_value, prefix=child_prefix)
            )
        elif original_value != saved_value:
            differences.append(f"{prefix}{name}: {original_value!r} != {saved_value!r}")

    return differences


# endregion
