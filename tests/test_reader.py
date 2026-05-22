"""Unit tests for :mod:`spiraldb_nemo_curator.io.reader`.

These tests stub out the Spiral client so they exercise the dataclass plumbing,
identifier resolution, and the population of Video metadata without needing a
live SpiralDB cluster.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest


def _key_table(rows: list[dict[str, object]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def test_resolve_source_id_explicit_column():
    from spiraldb_nemo_curator.io.reader import _resolve_source_id

    key_row = {"clip_id": "abc", "shard": 3}
    assert _resolve_source_id(key_row, ["clip_id", "shard"], source_id_column="clip_id") == "abc"


def test_resolve_source_id_compound_key():
    from spiraldb_nemo_curator.io.reader import _resolve_source_id

    key_row = {"shard": 3, "clip_id": "abc"}
    # Order matches the key_columns argument, not the dict.
    assert _resolve_source_id(key_row, ["shard", "clip_id"], source_id_column=None) == "3/abc"


def test_partition_stage_emits_one_task_per_row(fake_spiral):
    from nemo_curator.tasks import _EmptyTask

    from spiraldb_nemo_curator.io.reader import SpiralPartitionStage, SpiralRowTask

    schema = pa.schema({"clip_id": pa.string()})
    table = _key_table([{"clip_id": "row-1"}, {"clip_id": "row-2"}, {"clip_id": "row-3"}], schema)
    project = fake_spiral.project("proj")
    # Register a fake source table with key schema ["clip_id"].
    project.create_table("clips", key_schema=pa.schema({"clip_id": pa.string()}), exist_ok=True)
    fake_spiral.scan_keys_responses.append(table)

    stage = SpiralPartitionStage(
        project_id="proj",
        table_name="clips",
        source_id_column="clip_id",
    )
    stage.setup()
    tasks = stage.process(_EmptyTask(task_id="t", dataset_name="d", data=None))

    assert [t.source_id for t in tasks] == ["row-1", "row-2", "row-3"]
    assert all(isinstance(t, SpiralRowTask) for t in tasks)
    assert all(t.dataset_name == "proj.clips" for t in tasks)


def test_partition_stage_rejects_non_key_source_column(fake_spiral):
    from spiraldb_nemo_curator.io.reader import SpiralPartitionStage

    project = fake_spiral.project("proj")
    project.create_table("clips", key_schema=pa.schema({"clip_id": pa.string()}), exist_ok=True)

    stage = SpiralPartitionStage(
        project_id="proj",
        table_name="clips",
        source_id_column="not_a_key",
    )
    with pytest.raises(ValueError, match="source_id_column"):
        stage.setup()


def test_partition_stage_compound_key_default_join(fake_spiral):
    from nemo_curator.tasks import _EmptyTask

    from spiraldb_nemo_curator.io.reader import SpiralPartitionStage

    schema = pa.schema({"shard": pa.int32(), "clip_id": pa.string()})
    table = _key_table(
        [
            {"shard": 0, "clip_id": "a"},
            {"shard": 1, "clip_id": "b"},
        ],
        schema,
    )
    project = fake_spiral.project("proj")
    project.create_table(
        "clips",
        key_schema=pa.schema({"shard": pa.int32(), "clip_id": pa.string()}),
        exist_ok=True,
    )
    fake_spiral.scan_keys_responses.append(table)

    stage = SpiralPartitionStage(project_id="proj", table_name="clips")
    stage.setup()
    tasks = stage.process(_EmptyTask(task_id="t", dataset_name="d", data=None))
    assert [t.source_id for t in tasks] == ["0/a", "1/b"]


def _stub_where(stage):
    """Replace the real expression-builder with a sentinel that doesn't touch native code."""
    stage._build_where = lambda key_row: ("WHERE", tuple(sorted(key_row.items())))


def test_video_reader_stage_populates_video(fake_spiral, monkeypatch):
    from nemo_curator.tasks.video import VideoMetadata

    from spiraldb_nemo_curator.io.reader import (
        SpiralRowTask,
        SpiralVideoReaderStage,
    )

    project = fake_spiral.project("proj")
    project.create_table("clips", key_schema=pa.schema({"clip_id": pa.string()}), exist_ok=True)
    payload = b"<mp4-bytes>"
    fake_spiral.scan_responses.append(
        pa.table({"video_bytes": pa.array([payload], type=pa.binary())})
    )

    captured = {}

    def fake_populate(self):
        captured["called"] = True
        self.metadata = VideoMetadata(
            size=len(self.source_bytes),
            height=720,
            width=1280,
            framerate=30.0,
            num_frames=300,
            duration=10.0,
            video_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            bit_rate_k=2500,
        )

    monkeypatch.setattr("nemo_curator.tasks.video.Video.populate_metadata", fake_populate)

    stage = SpiralVideoReaderStage(project_id="proj", table_name="clips")
    stage.setup()
    _stub_where(stage)
    task = SpiralRowTask(
        task_id="proj.clips#0",
        dataset_name="proj.clips",
        data={"clip_id": "row-1"},
        source_id="row-1",
    )
    out = stage.process(task)

    assert out.task_id == "row-1_processed"
    assert out.data.input_video == Path("row-1")
    assert out.data.source_bytes == payload
    assert captured.get("called") is True
    assert out.data.metadata.video_codec == "h264"


def test_video_reader_stage_records_metadata_error(fake_spiral, monkeypatch):
    from spiraldb_nemo_curator.io.reader import (
        SpiralRowTask,
        SpiralVideoReaderStage,
    )

    project = fake_spiral.project("proj")
    project.create_table("clips", key_schema=pa.schema({"clip_id": pa.string()}), exist_ok=True)
    fake_spiral.scan_responses.append(
        pa.table({"video_bytes": pa.array([b"\x00\x00not-a-real-mp4"], type=pa.binary())})
    )

    def boom(self):
        raise RuntimeError("ffprobe blew up")

    monkeypatch.setattr("nemo_curator.tasks.video.Video.populate_metadata", boom)

    stage = SpiralVideoReaderStage(project_id="proj", table_name="clips")
    stage.setup()
    _stub_where(stage)
    task = SpiralRowTask(
        task_id="proj.clips#0",
        dataset_name="proj.clips",
        data={"clip_id": "row-1"},
        source_id="row-1",
    )
    out = stage.process(task)

    # The task is still returned; the error is captured on the Video.
    assert out.data.errors == {"metadata": "ffprobe blew up"}


def test_video_reader_stage_raises_on_missing_row(fake_spiral):
    from spiraldb_nemo_curator.io.reader import (
        SpiralRowTask,
        SpiralVideoReaderStage,
    )

    project = fake_spiral.project("proj")
    project.create_table("clips", key_schema=pa.schema({"clip_id": pa.string()}), exist_ok=True)
    fake_spiral.scan_responses.append(pa.table({"video_bytes": pa.array([], type=pa.binary())}))

    stage = SpiralVideoReaderStage(project_id="proj", table_name="clips")
    stage.setup()
    _stub_where(stage)
    task = SpiralRowTask(
        task_id="proj.clips#0",
        dataset_name="proj.clips",
        data={"clip_id": "missing"},
        source_id="missing",
    )
    with pytest.raises(RuntimeError, match="no row for source_id"):
        stage.process(task)


def test_composite_reader_decompose_shape():
    from spiraldb_nemo_curator.io.reader import (
        SpiralPartitionStage,
        SpiralVideoReader,
        SpiralVideoReaderStage,
    )

    reader = SpiralVideoReader(
        project_id="proj",
        table_name="clips",
        video_column="video",
        source_id_column="clip_id",
        limit=10,
        verbose=True,
    )
    stages = reader.decompose()
    assert [type(s) for s in stages] == [SpiralPartitionStage, SpiralVideoReaderStage]
    partition, reader_stage = stages
    assert partition.project_id == "proj"
    assert partition.limit == 10
    assert reader_stage.video_column == "video"
    assert reader_stage.source_id_column == "clip_id"
