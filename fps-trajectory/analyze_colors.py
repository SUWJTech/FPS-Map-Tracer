#!/usr/bin/env python3
import pandas as pd
from collections import Counter

df = pd.read_csv('configs/outputs/trajectory.csv')

# 计算每条轨迹的颜色变化次数
color_stability = {}
for track_id in df['track_id'].unique():
    track_data = df[df['track_id'] == track_id]
    colors = track_data['team_color'].values
    
    # 计算颜色变化次数
    color_changes = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i-1])
    color_stability[track_id] = {
        'total_points': len(colors),
        'color_changes': color_changes,
        'color_change_ratio': color_changes / (len(colors) - 1) if len(colors) > 1 else 0,
        'primary_color': Counter(colors).most_common(1)[0][0]
    }

# 统计
total_color_changes = sum(v['color_changes'] for v in color_stability.values())
avg_changes = total_color_changes / len(color_stability)
avg_change_ratio = sum(v['color_change_ratio'] for v in color_stability.values()) / len(color_stability)

print(f"总轨迹数: {len(color_stability)}")
print(f"总颜色变化次数: {total_color_changes}")
print(f"平均颜色变化次数/轨迹: {avg_changes:.2f}")
print(f"平均颜色变化率: {avg_change_ratio:.4f}")
print()
print("颜色变化最多的5条轨迹:")
sorted_by_changes = sorted(color_stability.items(), key=lambda x: x[1]['color_changes'], reverse=True)
for track_id, info in sorted_by_changes[:5]:
    print(f"  TrackID {track_id}: {info['color_changes']} changes in {info['total_points']} points, primary={info['primary_color']}")
