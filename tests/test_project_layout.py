import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omnimem.cli import build_parser
from omnimem.config import require_memgallery_dir


class ProjectLayoutTest(unittest.TestCase):
    def test_cli_selects_memgallery_method(self):
        args, rest = build_parser().parse_known_args(
            ["benchmark", "memgallery", "gme", "--max-questions", "2"]
        )
        self.assertEqual(args.method, "gme")
        self.assertEqual(rest, ["--max-questions", "2"])

    def test_memgallery_path_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "Mem-Gallery"
            (dataset / "data" / "dialog").mkdir(parents=True)
            with patch.dict(
                os.environ,
                {"OMNIMEM_MEMGALLERY_DIR": str(dataset)},
                clear=False,
            ):
                self.assertEqual(require_memgallery_dir(), dataset.resolve())

    def test_memgallery_path_error_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            with self.assertRaisesRegex(
                FileNotFoundError,
                "OMNIMEM_MEMGALLERY_DIR",
            ):
                require_memgallery_dir(missing)

    def test_project_has_packaging_metadata(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "pyproject.toml").is_file())
        self.assertTrue((root / "README.md").is_file())
