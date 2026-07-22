import struct

MAGIC = b"FLASHPK\0"  # 8 bytes
U64LE = struct.Struct("<Q")  # little-endian uint64

FILE_FORMAT_V3 = "flashpack_v3"
FILE_FORMAT_V4 = "flashpack_v4"

DEFAULT_ALIGN_BYTES = 128
DEFAULT_NUM_WRITE_WORKERS = 32
DEFAULT_NUM_STREAMS = 4
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB

# fpz split-plane zstd compression (V4-additive; opt-in via pack_to_file(compress=...)).
# bf16 macroblocks only: viewed as little-endian uint16, the low byte plane
# (mantissa LSB) is incompressible and stored raw, while the high byte plane
# (sign+exponent) compresses ~2.5x with zstd. A compressed macroblock's payload
# is a sequence of frames covering the block in FPZ_FRAME_UNCOMPRESSED_BYTES
# (uncompressed) steps; each frame is [lo raw bytes][hi zstd bytes] with the
# frame payload start 4096-byte aligned relative to the macroblock start.
FPZ_COMPRESS_BF16 = "fpz-bf16"
FPZ_CODEC_SPLITPLANE_V1 = "zstd-splitplane-v1"
FPZ_FRAME_UNCOMPRESSED_BYTES = 64 * 1024 * 1024  # 64 MiB
FPZ_FRAME_ALIGN_BYTES = 4096
DEFAULT_ZSTD_LEVEL = 3
