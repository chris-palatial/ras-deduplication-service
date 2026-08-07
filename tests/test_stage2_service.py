import base64
import tempfile
import unittest
from pathlib import Path

import stage2_service as stage2


class Stage2ServiceTest(unittest.TestCase):
    def test_rejects_unknown_mode_before_model_bootstrap(self):
        result = stage2.run_stage2(
            {"mode": "typo", "video_b64": "eA==", "categories": ["chair"], "max_frames": 8}
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("dry_run, geometry, or full", result["error"])

    def test_rejects_category_and_frame_boundaries(self):
        result = stage2.run_stage2(
            {"mode": "dry_run", "video_b64": "eA==", "categories": ["x" * 65], "max_frames": 8}
        )
        self.assertEqual(result["status"], "error")
        result = stage2.run_stage2(
            {"mode": "dry_run", "video_b64": "eA==", "categories": ["chair"], "max_frames": 161}
        )
        self.assertEqual(result["status"], "error")

    def test_inline_video_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = stage2._materialize_video(
                {"video_b64": base64.b64encode(b"video").decode(), "media_type": "video/webm"},
                Path(tmp),
            )
            self.assertEqual(path.name, "input.webm")
            self.assertEqual(path.read_bytes(), b"video")
            with self.assertRaisesRegex(RuntimeError, "valid base64"):
                stage2._materialize_video({"video_b64": "***"}, Path(tmp))


if __name__ == "__main__":
    unittest.main()
