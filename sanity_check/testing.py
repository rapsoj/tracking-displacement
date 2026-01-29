#!/usr/bin/env python3
"""
inspect_h5_tiles.py

Inspect tiles inside an HDF5 produced by your pipeline. The file layout assumed:
 - /feature/tile_XXXXX
 - /label/tile_XXXXX
 - /prewar/tile_XXXXX   (optional; may not exist for every tile)

Usage examples:
  python inspect_h5_tiles.py final_data_labelling.h5 --first 10 --show
  python inspect_h5_tiles.py final_data_labelling.h5 --indices 0,1,5 --outdir out
  python inspect_h5_tiles.py final_data_labelling.h5 --range 0-20 --outdir out
  python inspect_h5_tiles.py final_data_labelling.h5 --random 20 --show
"""
import argparse
import os
import random
from typing import List, Tuple
from scipy.ndimage import gaussian_filter

import h5py
import matplotlib.pyplot as plt
import numpy as np

import json
import h5py


def list_tile_names(h5path: str) -> list[str]:
    """Return a stable ordered list of tile 'names'.

    Supports two HDF5 layouts:
      - per-tile datasets under /feature (group) with names like tile_00000
      - single chunked dataset /feature shaped (N,H,W) where we synthesize tile names
    """
    with h5py.File(h5path, "r") as f:
        feat = f["feature"]
        if isinstance(feat, h5py.Group):
            names = sorted(list(feat.keys()))
        elif isinstance(feat, h5py.Dataset):
            n = feat.shape[0]
            names = [f"tile_{i:05d}" for i in range(n)]
        else:
            raise RuntimeError("Unknown /feature object type in HDF5")
    return names


def read_tile(h5path: str, tile_name: str):
    """
    Return (feature, label, prewar_or_None, attrs_dict)

    Works with both per-tile-group layout and chunked-dataset layout.
    For chunked-dataset layout, metadata is read from /meta if present (expects JSON strings).
    """
    with h5py.File(h5path, "r") as f:
        feat_obj = f["feature"]
        # decide layout
        if isinstance(feat_obj, h5py.Group):
            # per-tile datasets
            feat_ds = feat_obj.get(tile_name)
            if feat_ds is None:
                raise KeyError(f"{tile_name} not present under /feature")
            feature = np.array(feat_ds)

            label = None
            if "label" in f:
                lab_ds = f["label"].get(tile_name)
                label = np.array(lab_ds) if lab_ds is not None else None

            prewar = None
            if "prewar" in f:
                pre_ds = f["prewar"].get(tile_name)
                prewar = np.array(pre_ds) if pre_ds is not None else None

            # collect attrs from whichever dataset exists (feat preferred)
            attrs = {}
            for ds in (feat_ds, lab_ds if 'lab_ds' in locals() else None, pre_ds if 'pre_ds' in locals() else None):
                if ds is None:
                    continue
                for k, v in ds.attrs.items():
                    attrs.setdefault(k, v)
        else:
            # chunked dataset layout: /feature is Dataset indexed by integer
            # tile_name must encode the index as tile_{idx:05d}
            try:
                idx = int(tile_name.split("_")[-1])
            except Exception:
                raise KeyError("Invalid tile_name for dataset layout; expected tile_##...")

            feature = np.array(feat_obj[idx])

            label = None
            if "label" in f and isinstance(f["label"], h5py.Dataset):
                label = np.array(f["label"][idx])

            prewar = None
            if "prewar" in f and isinstance(f["prewar"], h5py.Dataset):
                prewar = np.array(f["prewar"][idx])

            # metadata: prefer /meta dataset (vlen JSON or strings), fallback to dataset attrs (unlikely)
            attrs = {}
            if "meta" in f and isinstance(f["meta"], h5py.Dataset):
                try:
                    meta_raw = f["meta"][idx]
                    # h5py may return bytes or str
                    if isinstance(meta_raw, bytes):
                        meta_raw = meta_raw.decode("utf-8", errors="ignore")
                    if isinstance(meta_raw, str) and meta_raw:
                        try:
                            attrs = json.loads(meta_raw)
                        except Exception:
                            # not JSON: store raw as 'meta'
                            attrs = {"meta": meta_raw}
                except Exception:
                    attrs = {}
            else:
                # try global attrs on dataset (not per-tile, but might exist)
                for k, v in feat_obj.attrs.items():
                    attrs.setdefault(k, v)

    return feature, label, prewar, attrs


def plot_feature_label_prewar(axs, feature, label, prewar):
    """
    Left: feature with label overlaid as red dots
    Right: prewar
    """
    for ax in axs:
        ax.clear()
        ax.axis("off")

    # panel 1: feature + label
    if feature.ndim == 2:
        axs[0].imshow(feature, cmap="gray", interpolation="nearest")
    else:
        axs[0].imshow(feature.astype(np.uint8))

    if label is not None:
        sigma = 5.0  # <-- use the same sigma as training
        blurred = gaussian_filter(label.astype(np.float32), sigma=sigma)
        axs[0].imshow(blurred, cmap="hot", alpha=0.5)

    # panel 2: prewar
    if prewar is None:
        axs[1].text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
    else:
        if prewar.ndim == 2:
            axs[1].imshow(prewar, cmap="gray", interpolation="nearest")
        else:
            axs[1].imshow(prewar.astype(np.uint8))


def expand_indices(names: List[str], indices_arg: str | None, first: int | None, range_arg: str | None) -> List[int]:
    if indices_arg:
        parts = [p.strip() for p in indices_arg.split(",") if p.strip()]
        out = []
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(p))
        return sorted(set(out))
    if range_arg:
        a, b = range_arg.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if first is not None:
        return list(range(first))
    return []


def interactive_viewer(h5path: str, tile_indices: List[Tuple[int, str]]):
    """
    Interactive viewer: left/right arrows move, q or Esc closes.
    Loads tiles lazily (reads from HDF5 per index).
    """
    if not tile_indices:
        print("No tiles to show.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(top=0.88)

    state = {"pos": 0, "n": len(tile_indices), "fig": fig, "axes": axes, "h5path": h5path}

    def show_pos():
        idx, name = tile_indices[state["pos"]]
        feature, label, prewar, attrs = read_tile(h5path, name)
        title = f"{name}  idx={idx}"
        if "origin_date" in attrs:
            title += f"  date={attrs.get('origin_date')}"
        plot_feature_label_prewar(axes, feature, label, prewar)
        fig.suptitle(title)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "pagedown"):
            state["pos"] = (state["pos"] + 1) % state["n"]
            show_pos()
        elif event.key in ("left", "pageup"):
            state["pos"] = (state["pos"] - 1) % state["n"]
            show_pos()
        elif event.key in ("q", "escape"):
            plt.close(state["fig"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_pos()
    plt.show()  # blocks until figure closed


def main():
    p = argparse.ArgumentParser(description="Inspect tiles in an HDF5 labeling file")
    p.add_argument("h5file", help="Path to final_data_labelling.h5")
    p.add_argument("--indices", help="Comma separated tile indices (e.g. 0,1,5)")
    p.add_argument("--range", dest="rng", help="Range of indices e.g. 0-20")
    p.add_argument("--first", type=int, help="Plot the first N tile indices", default=None)
    p.add_argument("--random", type=int, help="Plot N random tile indices (mutually exclusive with --first/--indices/--range)", default=None)
    p.add_argument("--seed", type=int, help="Optional RNG seed for --random", default=None)
    p.add_argument("--outdir", help="If set, save PNGs to this directory", default=None)
    p.add_argument("--show", action="store_true", help="Show plots interactively (use arrow keys)")
    p.add_argument("--list", action="store_true", help="List available tile names and exit")
    args = p.parse_args()

    h5path = args.h5file
    if not os.path.exists(h5path):
        raise SystemExit(f"HDF5 file not found: {h5path}")

    names = list_tile_names(h5path)
    if args.list:
        print("Found tile names (example):")
        for i, n in enumerate(names[:200]):
            print(f"{i:04d}: {n}")
        print(f"... total {len(names)} tiles")
        return

    # ensure mutual exclusion
    selection_args = [bool(args.indices), bool(args.rng), args.first is not None, args.random is not None]
    if sum(selection_args) > 1:
        raise SystemExit("Provide only one of --indices, --range, --first or --random.")

    if args.random is not None:
        # sample n distinct indices
        n = args.random
        if n <= 0:
            raise SystemExit("--random must be a positive integer")
        total = len(names)
        k = min(n, total)
        if args.seed is not None:
            random.seed(args.seed)
        sampled = sorted(random.sample(range(total), k))
        selected_indices = sampled
    else:
        selected_indices = expand_indices(names, args.indices, args.first, args.rng)

    if not selected_indices:
        raise SystemExit("No indices selected. Use --first, --indices, --range or --random. Use --list to inspect names.")

    tile_indices: List[Tuple[int, str]] = []
    for idx in selected_indices:
        if idx < 0 or idx >= len(names):
            print(f"warning: index {idx} out of range (0..{len(names)-1}), skipping")
            continue
        tile_indices.append((idx, names[idx]))

    if not tile_indices:
        raise SystemExit("No valid indices to process after filtering.")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    # Save PNGs if requested, otherwise optionally open interactive viewer
    for idx, name in tile_indices:
        feature, label, prewar, attrs = read_tile(h5path, name)
        title = f"{name}  idx={idx}"
        if "origin_date" in attrs:
            title += f"  date={attrs.get('origin_date')}"

        if args.outdir:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            plot_feature_label_prewar(axes, feature, label, prewar)
            fig.suptitle(title)
            outpath = os.path.join(args.outdir, f"{name}.png")
            fig.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {outpath}")

    if args.show:
        # for interactive viewing we pass the same ordered list
        interactive_viewer(h5path, tile_indices)

    print("done")


if __name__ == "__main__":
    main()
