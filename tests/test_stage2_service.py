import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_ephemeral_geometry_skips_unusable_artifact_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            with patch.dict(
                stage2.os.environ,
                {"STAGE2_ARTIFACT_DIR": "", "STAGE2_KEEP_WORK": "0"},
            ):
                self.assertFalse(stage2._artifact_exports_enabled())
                manifest = stage2._artifact_manifest(out_dir, work)
        self.assertEqual(manifest["files"], [])
        self.assertIn("export was skipped", manifest["note"])


if __name__ == "__main__":
    unittest.main()
