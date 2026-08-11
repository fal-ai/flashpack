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
# Override with FLASHPACK_SHARD_BYTES.
DEFAULT_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB

# How a sharded distributed read divides the payload and replicates it.
#
#   "contiguous"  one contiguous length/world slab per rank, replicated with
#                 world owner-broadcasts. One long sequential read per rank
#                 per block.
#   "windows"     interleaved superwindows (rank r takes shard r of every
#                 world*DEFAULT_SHARD_BYTES window), replicated with one
#                 in-place AllGather per window. Equal shards make the single
#                 AllGather legal; it is one collective per window instead of
#                 world, and measured 1.15x faster than the broadcasts at
#                 moving the same bytes (31.8 ms vs 36.5 ms for 12.9 GB across
#                 8 H200s). Bounds bytes in flight per collective regardless of
#                 pack size.
#
# End-to-end on an 8xH200 node with a 14.5 GB fp8 pack the two are within noise
# (0.65 s vs 0.66 s) because the load is read-bound there, so neither is known
# to dominate across pack sizes, world sizes and fabrics -- hence both ship.
# "contiguous" is the default only because it is the longer-exercised shape;
# measure with scripts/ before changing it for a given deployment. Override
# with FLASHPACK_SHARD_STRATEGY.
SHARD_STRATEGY_CONTIGUOUS = "contiguous"
SHARD_STRATEGY_WINDOWS = "windows"
SHARD_STRATEGIES = (SHARD_STRATEGY_CONTIGUOUS, SHARD_STRATEGY_WINDOWS)
DEFAULT_SHARD_STRATEGY = SHARD_STRATEGY_CONTIGUOUS
