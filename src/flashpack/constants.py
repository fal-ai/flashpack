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
# v2 chunks each frame's high plane into many small independent zstd frames so a
# GPU decoder (nvcomp) gets its native many-chunk batch shape. A single-frame
# (v1) high plane is ONE nvcomp chunk and decodes serially on the GPU (~0.6
# GB/s); v2's chunks decode in parallel. In a v2 frame the payload is
# [lo raw bytes][hi zstd chunk 0][hi zstd chunk 1]... and the frame record
# carries "hi_chunks": the per-chunk compressed byte lengths (each chunk's
# uncompressed size is FPZ_HI_CHUNK_UNCOMPRESSED_BYTES except the frame's last).
FPZ_CODEC_SPLITPLANE_V2 = "zstd-splitplane-v2"
FPZ_FRAME_UNCOMPRESSED_BYTES = 64 * 1024 * 1024  # 64 MiB
FPZ_FRAME_ALIGN_BYTES = 4096
# Per-chunk uncompressed size for the v2 high plane. Matches nvcomp's default
# uncomp_chunk_size (65536) and divides the 32 MiB half-frame evenly (512
# chunks), so full frames have no odd-sized tail chunk.
FPZ_HI_CHUNK_UNCOMPRESSED_BYTES = 64 * 1024
# Byte alignment of each v2 compressed chunk's START within the frame payload
# (chunks are padded to this; "hi_chunks" still records true zstd lengths and
# the reader recomputes padded offsets from the block's "hi_align" footer
# field, absent = 1 for pre-alignment packs). 16 covers nvcomp's batched
# decompressor input-alignment requirement -- its C API rejects unaligned
# device chunk pointers with nvcompErrorAlignment, and the reference callers
# align inputs to max(16, queried requirement). Cost: <= 15 pad bytes per
# chunk (~0.001% at 1 MiB chunks).
FPZ_HI_CHUNK_ALIGN_BYTES = 16
DEFAULT_ZSTD_LEVEL = 3
