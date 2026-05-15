from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose FPS trajectory pipeline outputs")
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    out = Path(args.output_dir)
    traj_csv = out / "trajectory.csv"
    traj_json = out / "trajectory.json"
    mapped_csv = out / "mapped_trajectory.csv"
    events_csv = out / "events.csv"

    print("=== Diagnose Report ===")
    print(f"output_dir: {out.resolve()}")

    df_traj = safe_read_csv(traj_csv)
    print(f"trajectory.csv exists={traj_csv.exists()} rows={len(df_traj)} cols={list(df_traj.columns)}")
    if len(df_traj) > 0 and "track_id" in df_traj.columns:
        print(f"trajectory unique tracks={df_traj['track_id'].nunique()}")

    df_map = safe_read_csv(mapped_csv)
    print(f"mapped_trajectory.csv exists={mapped_csv.exists()} rows={len(df_map)}")

    df_evt = safe_read_csv(events_csv)
    print(f"events.csv exists={events_csv.exists()} rows={len(df_evt)}")

    if traj_json.exists():
        payload = json.loads(traj_json.read_text(encoding="utf-8"))
        tracks = payload.get("tracks", {})
        print(f"trajectory.json tracks={len(tracks)}")
        team_cnt = {"white": 0, "red": 0, "green": 0, "purple": 0, "unknown": 0}
        for _, t in tracks.items():
            ft = str(t.get("final_team", "unknown")).lower()
            team_cnt[ft] = team_cnt.get(ft, 0) + 1
        print(f"final_team distribution={team_cnt}")

        # top 15 debug rows
        printed = 0
        for k, t in tracks.items():
            if printed >= 15:
                break
            print(
                f"track={k} final_team={t.get('final_team')} color_history_len={t.get('color_history_len')} "
                f"num_points={t.get('statistics', {}).get('num_points')}"
            )
            printed += 1
    else:
        print("trajectory.json exists=False")


if __name__ == "__main__":
    main()
