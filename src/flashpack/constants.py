import struct

MAGIC = b"FLASHPK\0"  # 8 bytes
U64LE = struct.Struct("<Q")  # little-endian uint64

FILE_FORMAT_V3 = "flashpack_v3"
FILE_FORMAT_V4 = "flashpack_v4"

DEFAULT_ALIGN_BYTES = 128
DEFAULT_NUM_WRITE_WORKERS = 32
DEFAULT_NUM_STREAMS = 4
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB
