import struct

MAGIC = b"FLASHPK\0"  # 8 bytes
U64LE = struct.Struct("<Q")  # little-endian uint64

FILE_FORMAT_V3 = "flashpack_v3"
FILE_FORMAT_V4 = "flashpack_v4"

DEFAULT_ALIGN_BYTES = 128
DEFAULT_NUM_WRITE_WORKERS = 32
DEFAULT_NUM_STREAMS = 4
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB

# Sharded distributed read: shard boundaries are aligned to this, which is the
# reader's O_DIRECT alignment and a multiple of every element size flashpack
# stores (so a shard always starts on an element boundary).
SHARD_ALIGN_BYTES = 4096

# Bytes each rank reads per superwindow, i.e. how much one AllGather moves.
# Swept on an 8xH200 node against a 14.5 GB fp8 pack (8 reader threads/rank):
# 64 MiB 3.80s, 128 MiB 3.25s, 256 MiB 3.32s, 512 MiB 4.03s -- flat between
# 128 and 256 MiB, so this sits in that band. Smaller wastes collectives on
# small messages; larger coarsens the tail and the work in flight.
DEFAULT_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB
