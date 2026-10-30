"""Stream metadata builders for the examples.

Each ``stream_metadata_*`` function builds a `StreamMetadata` for one
`StreamType`. `example_registry.STREAM_DICT` maps each `StreamType` to its
builder. Every builder takes the same arguments, so callers can use
``STREAM_DICT[stream_type](environment_name, output_directory=...)`` for any
stream type.

Builders that stream write ``example_stream.nc4``. They place it in
``output_directory``, or in the current working directory when
``output_directory`` is None. They never write inside the package tree.
"""

import os

from rattlesnake.process.streaming import StreamType, StreamMetadata

STREAM_FILE_NAME = "example_stream.nc4"


def _stream_file(output_directory: str | os.PathLike[str] | None) -> str:
    """Build the stream file path.

    Parameters
    ----------
    output_directory : str or os.PathLike or None
        Directory that holds the stream file. None selects the current working
        directory.

    Returns
    -------
    str
        Path to ``example_stream.nc4`` inside the chosen directory.
    """
    if output_directory is None:
        output_directory = os.getcwd()
    return os.path.join(output_directory, STREAM_FILE_NAME)


def stream_metadata_no(
    environment_name: str,
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> StreamMetadata:
    """Build metadata that does not stream.

    This function writes no file, so it ignores ``output_directory``. It accepts
    the argument only so that every builder in ``STREAM_DICT`` has the same
    signature and callers can invoke any of them uniformly.

    Parameters
    ----------
    environment_name : str
        Name of the environment. Ignored.
    output_directory : str or os.PathLike or None, optional
        Ignored.

    Returns
    -------
    StreamMetadata
        Metadata with `StreamType.NO_STREAM`.
    """
    stream_type = StreamType.NO_STREAM
    metadata = StreamMetadata(stream_type=stream_type)

    return metadata


def stream_metadata_immediate(
    environment_name: str,
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> StreamMetadata:
    """Build metadata that starts streaming when acquisition starts.

    Parameters
    ----------
    environment_name : str
        Name of the environment. Ignored.
    output_directory : str or os.PathLike or None, optional
        Directory that holds the stream file. None selects the current working
        directory.

    Returns
    -------
    StreamMetadata
        Metadata with `StreamType.IMMEDIATELY`.
    """
    stream_type = StreamType.IMMEDIATELY
    metadata = StreamMetadata(
        stream_type=stream_type, stream_file=_stream_file(output_directory)
    )

    return metadata


def stream_metadata_manual(
    environment_name: str,
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> StreamMetadata:
    """Build metadata that streams when the user starts it.

    Parameters
    ----------
    environment_name : str
        Name of the environment. Ignored.
    output_directory : str or os.PathLike or None, optional
        Directory that holds the stream file. None selects the current working
        directory.

    Returns
    -------
    StreamMetadata
        Metadata with `StreamType.MANUAL`.
    """
    stream_type = StreamType.MANUAL
    metadata = StreamMetadata(
        stream_type=stream_type, stream_file=_stream_file(output_directory)
    )

    return metadata


def stream_metadata_test_level(
    environment_name: str,
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> StreamMetadata:
    """Build metadata that streams while an environment runs at test level.

    Parameters
    ----------
    environment_name : str
        Name of the environment whose test level triggers streaming.
    output_directory : str or os.PathLike or None, optional
        Directory that holds the stream file. None selects the current working
        directory.

    Returns
    -------
    StreamMetadata
        Metadata with `StreamType.TEST_LEVEL`.
    """
    stream_type = StreamType.TEST_LEVEL
    metadata = StreamMetadata(
        stream_type=stream_type,
        stream_file=_stream_file(output_directory),
        test_level_environment_name=environment_name,
    )

    return metadata


def stream_metadata_profile(
    environment_name: str,
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> StreamMetadata:
    """Build metadata that streams on profile instructions.

    Parameters
    ----------
    environment_name : str
        Name of the environment. Ignored.
    output_directory : str or os.PathLike or None, optional
        Directory that holds the stream file. None selects the current working
        directory.

    Returns
    -------
    StreamMetadata
        Metadata with `StreamType.PROFILE_INSTRUCTION`.
    """
    stream_type = StreamType.PROFILE_INSTRUCTION
    metadata = StreamMetadata(
        stream_type=stream_type, stream_file=_stream_file(output_directory)
    )

    return metadata
