# Benchmarks

OmniMem ships the evaluation code, but not benchmark datasets or model weights.

For Mem-Gallery, use any one of these layouts:

1. Set `OMNIMEM_MEMGALLERY_DIR=/path/to/Mem-Gallery`.
2. Pass `--data-dir /path/to/Mem-Gallery`.
3. Place or symlink the checkout at `benchmarks/Mem-Gallery`.

The directory must contain `data/dialog/*.json` and the referenced images.
