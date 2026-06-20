# Multi-Lattice Separation: Reproducing the Training Pipeline

## Background

Around late 2024 / early 2025, we built an ML pipeline to separate
overlapping lattice contributions in diffraction patterns. The idea: simulate
many single-lattice monochromatic diffraction images, run DIALS spot-finding to
get per-lattice masks, then combine random subsets into synthetic multi-lattice
images with known ground-truth labels. A U-Net-style model learns to predict
which pixels belong to which lattice.

The commands below are adapted from the original notes. **The CLI has changed
since then** -- in particular, the old `--mono` and `--randomizerot` flags no
longer exist. The current equivalents are noted below.

---

## Step 1: Simulate monochromatic single-lattice patterns

The original command was:

```bash
# OLD (circa late 2024, flags no longer exist):
# datsimx.sim mono_1ksims --numimg 1000 --cuda --mono --oversample 1 --mosDoms 1 --ndev 4 --randomizerot
```

You will need to adapt this to the current CLI. Check `datsimx.sim --help` for
the up-to-date flags. Key things to replicate:

- **Monochromatic**: use `--monoEnergy <eV>` (default 7120 eV) and do NOT
  provide a `--specFile`.
- **Static / stills**: use `--static` so each image is a still (no phi
  rotation during exposure). The originals used random orientations per shot;
  with `--static` and a large `--numimg` over a full rotation range, you get
  diverse orientations. You may also want `--totalDeg 360` so the orientations
  span a full sphere.
- **Fast sims**: `--oversample 1 --mosDoms 1` keeps things fast.
- **GPU**: `--cuda --ndev 4` (adjust ndev to however many GPUs you have).
- **Output**: the first positional arg is the output directory, e.g.
  `mono_1ksims`.

A reasonable starting point:

```bash
datsimx.sim mono_1ksims \
    --numimg 1000 \
    --static \
    --totalDeg 360 \
    --oversample 1 \
    --mosDoms 1 \
    --cuda \
    --ndev 4
```

This will produce a Nexus master file (`.nxs`) and HDF5 data files in
`mono_1ksims/`.

---

## Step 2: Run DIALS spot-finding with `dials.stills_process`

The training data assembler (`mlat_combine.py`) expects
`proc/*strong.refl` files -- these are DIALS reflection tables from spot
finding. The original command used `dials.stills_process` with indexing
disabled (we only need spot-finding, not indexing/integration):

```bash
mpirun -n 64 dials.stills_process \
    mono_1ksims/run_1_master.h5 \
    dispatch.index=False \
    output.strong_filename="%s_strong.refl" \
    output.experiments_filename="%s_strong.expt" \
    mp.method=mpi \
    output_dir=mono_1ksims/proc
```

Key parameters:
- **Input**: the Nexus/HDF5 master file produced by `datsimx.sim` (e.g.
  `run_1_master.h5` -- check what's in the output directory).
- `dispatch.index=False`: skip indexing, we only want strong spots.
- `output.strong_filename` / `output.experiments_filename`: the `%s` template
  produces one `.refl`/`.expt` pair per image.
- `mp.method=mpi`: parallelize with MPI (adjust `-n` to your core count).
- `output_dir`: where the per-image `.refl`/`.expt` files are written.

---

## Step 3: Assemble training data (mlat_combine.py)

This script reads the DIALS spot-finding results, extracts per-image strong-spot
masks and raw images, downsamples them, and writes everything to an HDF5 file.

```bash
# Run with MPI for parallelism (adjust -n to your core count)
mpirun -n 64 python datsimx/mlat_combine.py \
    "mono_1ksims/proc/*strong.refl" \
    mono_1ksims/train_masks.h5
```

The output HDF5 (`train_masks.h5`) contains two datasets:
- `raw_images`: downsampled diffraction images (float32, 832x832 by default)
- `masks`: boolean spot masks for each image

You'll want to generate a separate test set too (e.g., simulate a second batch
and assemble into `test_masks.h5`).

---

## Step 4: Train the model

The training script lives at `datsimx/ml_sep/train.py`. For multi-lattice
separation, use `--goal sepMulti`:

```bash
python datsimx/ml_sep/train.py \
    mono_1ksims/train_masks.h5 \
    mono_1ksims/test_masks.h5 \
    sep_mlatts_output \
    --bs 12 \
    --nwork 12 \
    --lr 1e-2 \
    --nep 1000 \
    --goal sepMulti \
    --savefreq 1
```

Key args:
- Positional: `<train.h5> <test.h5> <output_dir>`
- `--goal sepMulti`: multi-lattice separation task (predicts per-lattice masks)
- `--bs`: batch size
- `--nwork`: DataLoader workers
- `--lr`: learning rate (the original used 1e-2 with SGD)
- `--adam`: use Adam instead of SGD
- `--nep`: number of epochs
- `--savefreq`: save model checkpoint every N epochs

The model architecture (`arch.py`) uses an `FCN50` (ResNet50-based FCN) or a
custom ResNet U-Net (`resnetU`). The dataset loader (`dset_loader.py`,
`MlatData` class) randomly combines 1-3 single-lattice masks at training time
to create synthetic multi-lattice examples on the fly.

The loss function is a dice loss that finds the best assignment between
predicted masks and ground-truth lattice masks (permutation-invariant).

Feel free to use `train.py` as a reference or write your own training loop.

---

## Supplementing with Rust-simulated patterns

If you are simulating diffraction patterns in Rust, you can supplement (or
replace) the training data as long as the final HDF5 file follows the same
format that `MlatData` in `dset_loader.py` expects:

- Dataset `raw_images`: shape `(N, 832, 832)`, dtype `float32`
  (only needed if you modify training to use raw images as input)
- Dataset `masks`: shape `(N, 832, 832)`, dtype `bool`
  (each image is a binary mask of strong spots for a single lattice)

The training loader randomly combines these single-lattice masks into
multi-lattice examples, so each entry should be one lattice only.

The image dimensions (832x832) come from the default `--cropdim` in
`mlat_combine.py` and can be changed, but must be consistent between data
generation and training.

Mixing Rust-generated and nanoBragg-generated training data could be a good way
to increase diversity and volume of training examples.

---

## File reference

| File | Purpose |
|------|---------|
| `datsimx/mx_simulate.py` | `datsimx.sim` entry point -- simulates diffraction images |
| `datsimx/mlat_combine.py` | Assembles single-lattice DIALS results into training HDF5 |
| `datsimx/ml_sep/train.py` | Training script (supports `sepMulti`, `predMulti`, `peakWave`) |
| `datsimx/ml_sep/arch.py` | Model architectures (FCN50, resnetU, resnetDown, predictMulti) |
| `datsimx/ml_sep/dset_loader.py` | PyTorch Dataset classes (MlatData, NlattsData, WaveData) |
| `datsimx/ml_sep/notes_found.txt` | Original command history from ~late 2024 |
