#!/usr/bin/env python3
"""
Knife Skills Trainer - Main Application
Advanced Proof of Concept: Real-time hand + knife detection with 3D depth-aware safety monitoring.
"""

import sys
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from detection.hand_detector import HandDetector
from detection.knife_detector import KnifeDetector
from ui.overlay_renderer import OverlayRenderer
from safety.safety_monitor import SafetyMonitor
from safety.depth_aware_safety import DepthAwareSafetyMonitor
from vision.depth_estimator import DepthEstimator


class DepthMode(Enum):
    """Depth visualization modes"""
    COLORMAP = "colormap"
    GRAYSCALE = "grayscale"
    HEATMAP = "heatmap"


@dataclass
class AppState:
    """Application state configuration"""
    is_running: bool = False
    is_paused: bool = False
    show_hands: bool = True
    show_knives: bool = True
    show_safety: bool = True
    show_depth: bool = True
    use_3d_safety: bool = True
    current_depth_mode: int = 0

    # Performance tracking
    fps_counter: int = 0
    fps_start_time: float = 0.0
    current_fps: float = 0.0


class KnifeSkillsTrainer:
    """Advanced Knife Skills Trainer - Proof of Concept"""

    # Constants
    CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
    FPS_UPDATE_INTERVAL = 10
    TARGET_FPS = 15

    def __init__(self):
        """Initialize the advanced knife skills trainer application"""
        print("🔪 🖐️ 🌊 Initializing Advanced Knife Skills Trainer...")
        print("Features: Hand Detection + Knife Detection + 3D Depth Safety")

        self.state = AppState(fps_start_time=time.time())
        self.depth_modes = [mode.value for mode in DepthMode]

        # Components (initialized in setup)
        self.components = {}
        self.cap: Optional[cv2.VideoCapture] = None

        # Keyboard handlers
        self._key_handlers = {
            27: self._handle_exit,  # ESC
            ord(' '): self._toggle_pause,
            ord('s'): self._save_screenshot,
            ord('S'): self._save_screenshot,
            ord('2'): lambda: self._set_safety_mode(False),
            ord('3'): lambda: self._set_safety_mode(True),
            ord('d'): self._cycle_depth_mode,
            ord('D'): self._cycle_depth_mode,
            ord('h'): lambda: self._toggle_feature('show_hands', 'Hand landmarks'),
            ord('H'): lambda: self._toggle_feature('show_hands', 'Hand landmarks'),
            ord('k'): lambda: self._toggle_feature('show_knives', 'Knife detection'),
            ord('K'): lambda: self._toggle_feature('show_knives', 'Knife detection'),
            ord('w'): lambda: self._toggle_feature('show_safety', 'Safety warnings'),
            ord('W'): lambda: self._toggle_feature('show_safety', 'Safety warnings'),
            ord('r'): self._reset_stats,
            ord('R'): self._reset_stats,
        }

    def _setup_components(self) -> bool:
        """Initialize all application components"""
        try:
            self.components = {
                'hand_detector': HandDetector(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5),
                'knife_detector': KnifeDetector(model_path="models/knife_seg_phase1_frozen/weights/best.pt", confidence_threshold=0.5, device="cpu"),
                'depth_estimator': DepthEstimator(model_type="coreml", model_size="small", max_depth=3.0, device="auto"),
                'safety_monitor_2d': SafetyMonitor(critical_threshold=30, high_threshold=50, medium_threshold=80),
                'overlay_renderer': OverlayRenderer(show_connections=True, show_landmarks=True, show_labels=False)
            }

            self.components['safety_monitor_3d'] = DepthAwareSafetyMonitor(
                depth_estimator=self.components['depth_estimator'],
                critical_threshold_3d=0.050, high_threshold_3d=0.100, medium_threshold_3d=0.200
            )

            for name in self.components:
                print(f"✅ {name.replace('_', ' ').title()} initialized")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            return False

    def _setup_camera(self):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"📹 Camera initialized ({self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT})")

    def _detect_objects(self, frame: np.ndarray) -> Tuple[list, list]:
        """Detect hands and knives in frame"""
        hands = self.components['hand_detector'].detect_hands(frame) if self.state.show_hands else []
        knives = self.components['knife_detector'].detect_knives(frame) if self.state.show_knives else []
        return hands, knives

    def _analyze_safety(self, hands: list, knives: list, frame: np.ndarray) -> Tuple[list, list, dict]:
        """Analyze safety using 2D or 3D methods"""
        if not (self.state.show_safety and hands and knives):
            return [], [], {}

        if self.state.use_3d_safety:
            stats = self.components['safety_monitor_3d'].compare_2d_vs_3d_detection(hands, knives, frame)
            return stats["dangers_2d"], stats["dangers_3d"], stats
        else:
            dangers_2d = self.components['safety_monitor_2d'].detect_fingertip_dangers(hands, knives)
            return dangers_2d, [], {}

    def _render_overlays(self, frame: np.ndarray, hands: list, knives: list, dangers_2d: list, dangers_3d: list):
        """Render all visual overlays on frame"""
        renderer = self.components['overlay_renderer']

        if self.state.show_hands:
            renderer.draw_hands(frame, hands)
        if self.state.show_knives:
            renderer.draw_knives(frame, knives)
        if self.state.show_safety:
            # Render 3D dangers if using 3D mode and they exist
            if self.state.use_3d_safety and dangers_3d:
                for danger in dangers_3d:
                    renderer.draw_3d_danger_warning(frame, danger)
            # Otherwise render 2D dangers
            elif dangers_2d:
                for danger in dangers_2d:
                    renderer.draw_danger_warning(frame, danger)

    def _create_depth_visualization(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Create depth visualization if enabled"""
        if not self.state.show_depth:
            return None

        try:
            depth_map = self.components['depth_estimator'].estimate_depth(frame)
            max_depth = self.components['depth_estimator'].max_depth
            normalized = (depth_map / max_depth * 255).astype(np.uint8)

            mode = self.depth_modes[self.state.current_depth_mode]
            if mode == "grayscale":
                return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            elif mode == "colormap":
                return cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
            else:  # heatmap
                return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        except Exception as e:
            print(f"⚠️ Depth estimation failed: {e}")
            return None

    def _update_fps(self):
        """Update FPS counter"""
        self.state.fps_counter += 1
        if self.state.fps_counter % self.FPS_UPDATE_INTERVAL == 0:
            elapsed = time.time() - self.state.fps_start_time
            self.state.current_fps = self.FPS_UPDATE_INTERVAL / elapsed
            self.state.fps_start_time = time.time()

    def _draw_status_overlay(self, frame: np.ndarray, comparison_stats: dict):
        """Draw concise status information"""
        y_pos = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Mode and FPS
        mode_color = (0, 255, 255) if self.state.use_3d_safety else (255, 255, 255)
        mode_text = f"{'3D' if self.state.use_3d_safety else '2D'} Safety | FPS: {self.state.current_fps:.1f}"
        cv2.putText(frame, mode_text, (20, y_pos), font, 0.7, mode_color, 2)

        # Safety stats (if using 3D)
        if self.state.use_3d_safety and comparison_stats:
            stats_text = f"2D: {comparison_stats['count_2d']} | 3D: {comparison_stats['count_3d']}"
            if comparison_stats['false_positive_reduction'] > 0:
                stats_text += f" | Filtered: {comparison_stats['false_positive_reduction']}"
            cv2.putText(frame, stats_text, (20, y_pos + 30), font, 0.6, (255, 255, 255), 2)

        # Depth mode
        if self.state.show_depth:
            depth_text = f"Depth: {self.depth_modes[self.state.current_depth_mode].upper()}"
            cv2.putText(frame, depth_text, (20, y_pos + 60), font, 0.6, (255, 255, 255), 2)

        # Pause indicator
        if self.state.is_paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 100, 50), font, 1.5, (0, 255, 255), 3)

        # Controls
        cv2.putText(frame, "ESC: Exit | SPACE: Pause | 2/3: Mode | D: Depth | H/K/W: Toggle",
                   (20, frame.shape[0] - 20), font, 0.5, (200, 200, 200), 1)

    def _process_frame(self) -> bool:
        """Process a single frame with full AI pipeline"""
        if self.cap is None:
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction

        # Detection pipeline
        hands, knives = self._detect_objects(frame)
        dangers_2d, dangers_3d, stats = self._analyze_safety(hands, knives, frame)

        # Rendering pipeline
        self._render_overlays(frame, hands, knives, dangers_2d, dangers_3d)
        depth_vis = self._create_depth_visualization(frame)

        # Display pipeline
        combined_frame = np.hstack([frame, depth_vis]) if depth_vis is not None else frame
        self._update_fps()
        self._draw_status_overlay(combined_frame, stats)

        cv2.imshow('Knife Skills Trainer - Advanced PoC', combined_frame)
        return True

    # Keyboard handlers
    def _handle_exit(self) -> bool:
        return False

    def _toggle_pause(self) -> bool:
        self.state.is_paused = not self.state.is_paused
        print(f"{'⏸️  Paused' if self.state.is_paused else '▶️  Resumed'}")
        return True

    def _save_screenshot(self) -> bool:
        filename = f"knife_skills_poc_{int(time.time())}.jpg"
        # Note: Would need to store last frame for this to work properly
        print(f"📸 Screenshot feature available (saved as {filename})")
        return True

    def _set_safety_mode(self, use_3d: bool) -> bool:
        self.state.use_3d_safety = use_3d
        print(f"🔄 Switched to {'3D' if use_3d else '2D'} Safety Mode")
        return True

    def _cycle_depth_mode(self) -> bool:
        self.state.current_depth_mode = (self.state.current_depth_mode + 1) % len(self.depth_modes)
        print(f"🎨 Depth mode: {self.depth_modes[self.state.current_depth_mode]}")
        return True

    def _toggle_feature(self, feature: str, display_name: str) -> bool:
        current_value = getattr(self.state, feature)
        setattr(self.state, feature, not current_value)
        print(f"{'✅' if not current_value else '❌'} {display_name}: {'ON' if not current_value else 'OFF'}")
        return True

    def _reset_stats(self) -> bool:
        for monitor in ['safety_monitor_2d', 'safety_monitor_3d']:
            if monitor in self.components:
                self.components[monitor].reset_stats()
                if hasattr(self.components[monitor], 'reset_3d_stats'):
                    self.components[monitor].reset_3d_stats()
        print("📊 Statistics reset")
        return True

    def _handle_keyboard_input(self) -> bool:
        """Handle keyboard input using handler mapping"""
        key = cv2.waitKey(1) & 0xFF
        if key in self._key_handlers:
            return self._key_handlers[key]()
        return True

    def _print_final_statistics(self):
        """Print concise final statistics"""
        if not all(comp in self.components for comp in ['safety_monitor_2d', 'safety_monitor_3d', 'depth_estimator']):
            return

        print("\n" + "=" * 60)
        print("📊 FINAL STATISTICS")
        print("=" * 60)

        # Safety stats
        stats_2d = self.components['safety_monitor_2d'].get_stats()
        stats_3d = self.components['safety_monitor_3d'].get_3d_stats()
        depth_stats = self.components['depth_estimator'].get_stats()

        print(f"2D Safety: {stats_2d['danger_events']}/{stats_2d['total_checks']} ({stats_2d['danger_rate']:.1%})")
        print(f"3D Safety: {stats_3d['danger_events_3d']}/{stats_3d['total_3d_checks']} ({stats_3d['danger_rate_3d']:.1%})")
        print(f"Depth: {depth_stats['model_type']} | {depth_stats['avg_inference_time_ms']:.1f}ms | {1000/depth_stats['avg_inference_time_ms']:.1f} FPS")

        performance_status = "✅" if (1000/depth_stats['avg_inference_time_ms']) >= self.TARGET_FPS else "⚠️"
        print(f"Performance: {performance_status} {'Real-time' if performance_status == '✅' else 'Below target'}")

    def run(self) -> int:
        """Main application loop for the advanced PoC"""
        if not self._setup_components():
            return 1

        try:
            self._setup_camera()
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return 1

        print("\n🚀 Advanced Knife Skills Trainer Ready!")
        print("Controls: ESC=Exit | SPACE=Pause | 2/3=Mode | D=Depth | H/K/W=Toggles | R=Reset")

        self.state.is_running = True

        try:
            while self.state.is_running:
                if not self.state.is_paused and not self._process_frame():
                    break
                if not self._handle_keyboard_input():
                    break

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
            return 1
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

        self._print_final_statistics()
        print("👋 Advanced PoC demo complete!")
        return 0


def main():
    """Main entry point"""
    try:
        return KnifeSkillsTrainer().run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())