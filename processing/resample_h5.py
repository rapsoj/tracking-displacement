import h5py
import numpy as np

inp = "train_data_labelling.h5"
out = "train_data_labelling_balanced.h5"

rng = np.random.default_rng(42)  # fixed seed for reproducibility

with h5py.File(inp, "r") as f:
    labels = f["label"]
    n = labels.shape[0]

    # A tile is "null" if all label pixels are zero
    is_null = np.all(labels[:] == 0, axis=(1, 2))
    is_non_null = ~is_null

    null_idx = np.where(is_null)[0]
    non_null_idx = np.where(is_non_null)[0]

    # keep 25% of null tiles, randomly
    keep_null_n = int(0.25 * len(null_idx))
    keep_null_idx = rng.choice(null_idx, size=keep_null_n, replace=False)

    # keep all non-null tiles
    keep_idx = np.sort(np.concatenate([non_null_idx, keep_null_idx]))

    print(f"Original tiles: {n}")
    print(f"Non-null kept : {len(non_null_idx)}")
    print(f"Null kept     : {len(keep_null_idx)}")
    print(f"Final tiles   : {len(keep_idx)}")

    with h5py.File(out, "w") as g:
        for name, d in f.items():
            g.create_dataset(
                name,
                shape=(len(keep_idx), *d.shape[1:]),
                maxshape=(None, *d.shape[1:]),
                chunks=d.chunks,
                dtype=d.dtype,
                compression=d.compression,
            )
            g[name][:] = d[keep_idx]

print("Saved train_data_labelling_balanced.h5")