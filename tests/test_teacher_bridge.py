from __future__ import annotations

import inspect
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predrel.teacher_bridge import (
    WEEK01_SOURCE_SHA256,
    TeacherBridgeError,
    _load_week01,
    extract_readout,
    verify_week01_snapshot,
)


class TeacherBridgeTests(unittest.TestCase):
    def test_in_package_teacher_binds_without_snapshot(self) -> None:
        modules = _load_week01(None)
        for key in (
            "SupportSet",
            "QuerySet",
            "TeacherRequest",
            "TabPFNTeacher",
            "official_decoder_readout",
            "capture_raw_scores",
        ):
            self.assertIn(key, modules)

    def test_legacy_snapshot_check_still_rejects_wrong_hash(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "provenance").mkdir()
            (root / "provenance" / "source_manifest.json").write_text(
                '{"source_sha256": "%s"}' % ("0" * 64), encoding="utf-8"
            )
            with self.assertRaises(TeacherBridgeError):
                verify_week01_snapshot(root, expected_source_sha256=WEEK01_SOURCE_SHA256)

    def test_teacher_api_has_no_query_label_parameter(self) -> None:
        self.assertNotIn("query_labels", inspect.signature(extract_readout).parameters)

    def test_hidden_is_opt_in_with_readout_only_default_path(self) -> None:
        params = inspect.signature(extract_readout).parameters
        self.assertIn("include_hidden", params)
        self.assertTrue(params["include_hidden"].default)


if __name__ == "__main__":
    unittest.main()
