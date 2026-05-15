"""
Team Estimation Layer - Decoupled from Tracking

Maintains independent color memory for each track using EMA fusion.
Colors are estimated AFTER track creation, not before.
Track ID stability is completely independent of color changes.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from .color_classifier import ColorVotingEngine, normalize_color_name
except ImportError:  # pragma: no cover
    from color_classifier import ColorVotingEngine, normalize_color_name


logger = logging.getLogger("fps_trajectory")


@dataclass
class ColorMemory:
    """Per-track color memory using high-inertia temporal fusion."""

    track_id: int
    color_votes: Dict[str, float] = field(default_factory=dict)
    color_history: Deque[str] = field(default_factory=lambda: deque(maxlen=60))
    stable_team: str = "unknown"
    color_locked: bool = False
    consistent_frames: int = 0
    conflict_frames: int = 0
    last_observed_color: str = "unknown"
    frames_since_color_change: int = 0
    ema_decay: float = 0.97
    ema_update: float = 0.03

    def fuse_observation(self, observed_color: str, confidence: float) -> None:
        """EMA fusion: memory = 0.97*old + 0.03*current."""
        observed = normalize_color_name(observed_color)

        for color in list(self.color_votes.keys()):
            self.color_votes[color] *= self.ema_decay

        if observed == "unknown" or confidence <= 0.0:
            return

        if observed not in self.color_votes:
            self.color_votes[observed] = 0.0
        self.color_votes[observed] += self.ema_update * float(confidence)

        self.color_history.append(observed)
        self.last_observed_color = observed

    def get_dominant_color(self) -> str:
        """Get color with highest weight."""
        if not self.color_votes:
            return "unknown"
        return max(self.color_votes.items(), key=lambda x: x[1])[0]


class TeamEstimator:
    """
    Decoupled Team Estimation Layer
    
    Maintains color memory for each track independently.
    No coupling with tracking logic.
    Colors are estimated AFTER track is created.
    Color changes do NOT trigger track reassignment.
    """
    
    def __init__(
        self,
        voting_window: int = 60,
        color_lock_frames: int = 10,
        color_change_threshold: float = 0.95,
        ema_decay: float = 0.97,
        ema_update: float = 0.03,
        unlock_required_frames: int = 30,
    ) -> None:
        self.voting_window = max(3, int(voting_window))
        self.color_lock_frames = max(1, int(color_lock_frames))
        self.color_change_threshold = float(color_change_threshold)
        self.ema_decay = float(ema_decay)
        self.ema_update = float(ema_update)
        self.unlock_required_frames = max(1, int(unlock_required_frames))
        
        # Per-track color memory
        self.track_colors: Dict[int, ColorMemory] = {}
        
        # Voting engines for color stability
        self.voting_engines: Dict[int, ColorVotingEngine] = {}
    
    def initialize_track_color(self, track_id: int) -> None:
        """Initialize color tracking for a new track."""
        if track_id not in self.track_colors:
            memory = ColorMemory(
                track_id=track_id,
                ema_decay=self.ema_decay,
                ema_update=self.ema_update,
            )
            self.track_colors[track_id] = memory
            
            voting_engine = ColorVotingEngine(
                voting_window=self.voting_window,
                color_lock_frames=self.color_lock_frames,
                color_change_threshold=self.color_change_threshold,
            )
            self.voting_engines[track_id] = voting_engine
    
    def update_track_color(
        self,
        track_id: int,
        observed_color: str,
        color_confidence: float,
        allow_update: bool = True,
    ) -> Tuple[str, float]:
        """
        Update color estimate for a track.
        
        Args:
            track_id: Track ID
            observed_color: Observed color from classifier
            color_confidence: Confidence of color observation
            
        Returns:
            (stable_team, team_confidence)
        """
        if track_id not in self.track_colors:
            self.initialize_track_color(track_id)
        
        memory = self.track_colors[track_id]
        voting_engine = self.voting_engines[track_id]
        
        # Occlusion protection: if occluded, keep old color and skip updates.
        if allow_update:
            memory.fuse_observation(observed_color, color_confidence)

            observed_norm = normalize_color_name(observed_color)
            (
                memory.stable_team,
                memory.color_locked,
                memory.consistent_frames,
                memory.conflict_frames,
                _,
            ) = voting_engine.update(
                history=memory.color_history,
                observed_color=observed_norm,
                stable_color=memory.stable_team,
                color_locked=memory.color_locked,
                consistent_frames=memory.consistent_frames,
                conflict_frames=memory.conflict_frames,
            )

            # Locked color switch is heavily restricted.
            if memory.color_locked and memory.conflict_frames >= self.unlock_required_frames:
                dominant = memory.get_dominant_color()
                if dominant != "unknown" and dominant != memory.stable_team:
                    memory.stable_team = dominant
                    memory.color_locked = False
                    memory.consistent_frames = 0
                    memory.conflict_frames = 0
                    memory.frames_since_color_change = 0

        if memory.stable_team == memory.last_observed_color:
            memory.frames_since_color_change += 1
        else:
            if memory.stable_team != "unknown":
                memory.last_observed_color = memory.stable_team
            memory.frames_since_color_change = 0

        team_confidence = self._compute_color_confidence(memory)
        return memory.stable_team, team_confidence
    
    def get_stable_team(self, track_id: int) -> Tuple[str, float]:
        """
        Get current stable team estimate for a track.
        
        Returns:
            (stable_team, team_confidence)
        """
        if track_id not in self.track_colors:
            return "unknown", 0.0
        
        memory = self.track_colors[track_id]
        confidence = self._compute_color_confidence(memory)
        return memory.stable_team, confidence
    
    def _compute_color_confidence(self, memory: ColorMemory) -> float:
        """
        Compute confidence of stable team estimate.
        
        Higher confidence when:
        - Dominant color has high weight
        - Not recently conflicted
        - Color is locked
        """
        if memory.stable_team == "unknown":
            return 0.0
        
        # Weight dominance (0 to 1)
        dominant_weight = memory.color_votes.get(memory.stable_team, 0.0)
        total_weight = sum(memory.color_votes.values())
        if total_weight <= 1e-6:
            return 0.0
        dominance = dominant_weight / total_weight
        
        # Lock bonus (locked colors are more stable)
        lock_bonus = 0.1 if memory.color_locked else 0.0
        
        # Conflict penalty
        conflict_penalty = 0.1 * memory.conflict_frames / max(1, self.color_lock_frames)
        conflict_penalty = min(0.3, conflict_penalty)  # cap penalty
        
        confidence = dominance + lock_bonus - conflict_penalty
        return max(0.0, min(1.0, confidence))
    
    def cleanup_track(self, track_id: int) -> None:
        """Clean up color memory when track is removed."""
        self.track_colors.pop(track_id, None)
        self.voting_engines.pop(track_id, None)
    
    def get_all_teams(self) -> Dict[int, Tuple[str, float]]:
        """Get all track teams and confidences."""
        result = {}
        for track_id in self.track_colors:
            team, conf = self.get_stable_team(track_id)
            result[track_id] = (team, conf)
        return result
    
    def debug_track_colors(self, track_id: int) -> Dict[str, Any]:
        """Debug info for a specific track's color state."""
        if track_id not in self.track_colors:
            return {"track_id": track_id, "error": "track not found"}
        
        memory = self.track_colors[track_id]
        team, conf = self.get_stable_team(track_id)
        
        return {
            "track_id": track_id,
            "stable_team": team,
            "team_confidence": conf,
            "color_weights": dict(memory.color_votes),
            "color_locked": memory.color_locked,
            "consistent_frames": memory.consistent_frames,
            "conflict_frames": memory.conflict_frames,
            "last_observed_color": memory.last_observed_color,
            "frames_since_color_change": memory.frames_since_color_change,
            "history_len": len(memory.color_history),
        }
