from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from pandas.errors import EmptyDataError

from src.map_registration import MapRegistrar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic minimap->global map registration and trajectory mapping")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv", default="outputs/trajectory.csv")
    parser.add_argument("--map_image", default="configs/map.jpg")
    parser.add_argument("--minimap_image", default="configs/minimap_ref.jpg")
    parser.add_argument("--matrix", default="configs/homography_matrix.json")
    parser.add_argument("--trajectory_json", default="outputs/trajectory.json")
    parser.add_argument("--output_csv", default="outputs/mapped_trajectory.csv")
    parser.add_argument("--debug_match_png", default="outputs/registration_matches.png")
    parser.add_argument("--force_reregister", action="store_true")
    return parser.parse_args()


def load_final_team_map(path: str) -> dict[int, str]:
    import json

    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    tracks = payload.get("tracks", {})
    out: dict[int, str] = {}
    for k, v in tracks.items():
        try:
            tid = int(v.get("track_id", int(k)))
        except Exception:
            continue
        out[tid] = str(v.get("final_team", "unknown"))
    return out


def main() -> None:
    args = parse_args()
    registrar = MapRegistrar()

    minimap_path = Path(args.minimap_image)
    if not minimap_path.exists():
        print(f"[map_mapper] minimap image not found: {minimap_path}")
        print(f"[map_mapper] auto extracting from video using config: {args.config}")
        generated = registrar.extract_minimap_reference_from_config(args.config, str(minimap_path))
        print(f"[map_mapper] saved minimap reference: {generated}")

    matrix_path = Path(args.matrix)
    if args.force_reregister or not matrix_path.exists():
        minimap_img = cv2.imread(str(minimap_path))
        map_img = cv2.imread(str(Path(args.map_image)))
        if minimap_img is None or map_img is None:
            raise FileNotFoundError("Cannot open minimap image or map image for registration")

        result = registrar.register(minimap_img, map_img)
        registrar.save_matrix(result.homography, str(matrix_path))
        if result.debug_image is not None:
            registrar.save_debug(result.debug_image, args.debug_match_png)
        print(f"[map_mapper] registration done | matches={result.matches} inliers={result.inliers} score={result.score:.3f}")
    else:
        print(f"[map_mapper] using existing matrix: {matrix_path}")
        H = registrar.load_matrix(str(matrix_path))
        registrar.save_matrix(H, str(matrix_path))

    H = registrar.load_matrix(str(matrix_path))

    traj_path = Path(args.trajectory_csv)
    if not traj_path.exists():
        raise FileNotFoundError(f"[map_mapper] trajectory csv not found: {traj_path}")

    try:
        df = pd.read_csv(traj_path)
        if len(df) == 0:
            print(f"[map_mapper] warning: trajectory csv is empty: {traj_path}")
    except EmptyDataError:
        print(f"[map_mapper] warning: trajectory csv has no columns/content: {traj_path}")

    out_csv = registrar.map_trajectory_csv(args.trajectory_csv, args.output_csv, H)

    # append team columns by track_id
    team_map = load_final_team_map(args.trajectory_json)
    try:
        mdf = pd.read_csv(out_csv)
        if len(mdf) > 0 and "track_id" in mdf.columns:
            if "team_color" not in mdf.columns:
                mdf["team_color"] = "unknown"
            mdf["final_team"] = mdf["track_id"].astype(int).map(lambda x: team_map.get(int(x), "unknown"))
            mdf["team"] = mdf["final_team"]
            mdf.to_csv(out_csv, index=False)
    except Exception:
        pass

    print(f"[map_mapper] mapped trajectory saved: {out_csv}")


if __name__ == "__main__":
    main()
