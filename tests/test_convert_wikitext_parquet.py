import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "convert_wikitext_parquet.py"
)
SPEC = importlib.util.spec_from_file_location("convert_wikitext_parquet", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeColumn:
    def __init__(self, values):
        self.values = values

    def to_pylist(self):
        return self.values


class FakeBatch:
    def __init__(self, values):
        self.values = values

    def column(self, index):
        self._index = index
        return FakeColumn(self.values)


class FakeParquetFile:
    rows = {}

    def __init__(self, path):
        self.path = Path(path)

    def iter_batches(self, batch_size, columns):
        if batch_size <= 0 or columns != ["text"]:
            raise AssertionError("Unexpected parquet reader arguments.")
        yield FakeBatch(self.rows[self.path.name])


class FakeParquetModule:
    ParquetFile = FakeParquetFile


class ConvertWikiTextParquetTest(unittest.TestCase):
    def test_streams_shards_in_order_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.parquet"
            second = root / "second.parquet"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            FakeParquetFile.rows = {
                first.name: ["alpha", "beta\n"],
                second.name: ["gamma"],
            }
            output = root / "wiki.train.raw"

            manifest = MODULE.convert_shards(
                [first, second],
                output,
                batch_size=2,
                expected_rows=3,
                parquet_module=FakeParquetModule,
            )

            self.assertEqual(output.read_text(encoding="utf-8"), "alpha\nbeta\ngamma\n")
            self.assertEqual(manifest["row_count"], 3)
            self.assertEqual([item["rows"] for item in manifest["sources"]], [2, 1])
            self.assertEqual(
                [item["path"] for item in manifest["sources"]],
                [str(first.resolve()), str(second.resolve())],
            )

    def test_expected_row_mismatch_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            source.write_bytes(b"source")
            FakeParquetFile.rows = {source.name: ["only one"]}
            output = root / "wiki.raw"

            with self.assertRaisesRegex(ValueError, "expected 2"):
                MODULE.convert_shards(
                    [source],
                    output,
                    batch_size=1,
                    expected_rows=2,
                    parquet_module=FakeParquetModule,
                )

            self.assertFalse(output.exists())
            self.assertFalse((root / "wiki.raw.partial").exists())

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            output = root / "wiki.raw"
            source.write_bytes(b"source")
            output.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                MODULE.convert_shards(
                    [source],
                    output,
                    batch_size=1,
                    parquet_module=FakeParquetModule,
                )


if __name__ == "__main__":
    unittest.main()
