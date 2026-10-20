# Third-party notices

MinSync is MIT licensed (see `LICENSE`). The files listed here contain code
derived from other projects under their own licenses, and the dependencies
below are linked into the `minsync` executable under their own licenses.

## fastembed-rs: Qwen3 model code

`src/embedder/qwen3/` is derived from `src/models/qwen3.rs` of
[fastembed-rs](https://github.com/Anush008/fastembed-rs) 7.0.1.

- Copyright 2026 Anush008 and the fastembed-rs contributors
- License: Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>)

Changes made in MinSync: the text-embedding path only (no vision-language
code); per-device attention backends; candle's fused RMSNorm; host-side padding
mask and rotary tables; pooling and normalization on the last token in f32.

## kiwi-rs: Korean BM25 tokenizer binding

The `ko` and `multilingual` BM25 tokenizers call the [Kiwi](https://github.com/bab2min/Kiwi)
Korean morphological analyzer through the
[`kiwi-rs`](https://crates.io/crates/kiwi-rs) crate
(source: <https://github.com/JAICHANGPARK/kiwi-rs>), which is statically linked
into the `minsync` executable. `kiwi-rs` is not modified.

- Crate: `kiwi-rs` 2026.8.26 (`Cargo.lock`)
- License: GNU Lesser General Public License, version 2.1 or later
  (<https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>)
- Copyright: the `kiwi-rs` authors

The Kiwi native library (`libkiwi`) and its model are **not** part of MinSync.
They are separate files that `kiwi-rs` loads at run time (`KIWI_LIBRARY_PATH`,
`KIWI_MODEL_PATH`). Kiwi 0.24.0 and later, and its models, are Apache-2.0;
Kiwi 0.23.2 and earlier are LGPL.

### Relinking against a modified `kiwi-rs` (LGPL-2.1 section 6)

MinSync is MIT licensed and its complete source is public, so the executable can
be rebuilt against a modified `kiwi-rs`. Check out the MinSync tag that matches
the executable and point Cargo at your copy of the crate:

```bash
git clone https://github.com/NomaDamas/MinSync && cd MinSync
git checkout <tag>   # e.g. v0.4.11
cargo build --release \
  --config 'patch.crates-io.kiwi-rs.path="/path/to/your/kiwi-rs"'
```

Do not add `--locked`: a `[patch]` changes the lock file. Users may modify
`kiwi-rs` for their own use and reverse engineer the result to debug those
modifications.
