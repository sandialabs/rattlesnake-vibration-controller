import rattlesnake.examples.defaults as defaults

from rattlesnake.process.streaming import StreamType, StreamMetadata

STREAM_FILE = defaults.DIRECTORY + "/example_stream.nc4"


def stream_metadata_no(environment_name):
    stream_type = StreamType.NO_STREAM
    metadata = StreamMetadata(stream_type=stream_type)

    return metadata


def stream_metadata_immediate(environment_name):
    stream_type = StreamType.IMMEDIATELY
    metadata = StreamMetadata(stream_type=stream_type, stream_file=STREAM_FILE)

    return metadata


def stream_metadata_manual(environment_name):
    stream_type = StreamType.MANUAL
    metadata = StreamMetadata(stream_type=stream_type, stream_file=STREAM_FILE)

    return metadata


def stream_metadata_test_level(environment_name):
    stream_type = StreamType.TEST_LEVEL
    metadata = StreamMetadata(
        stream_type=stream_type,
        stream_file=STREAM_FILE,
        test_level_environment_name=environment_name,
    )

    return metadata


def stream_metadata_profile(environment_name):
    stream_type = StreamType.PROFILE_INSTRUCTION
    metadata = StreamMetadata(stream_type=stream_type, stream_file=STREAM_FILE)

    return metadata
