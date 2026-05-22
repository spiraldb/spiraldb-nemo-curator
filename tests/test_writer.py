"""Unit tests for :mod:`spiraldb_nemo_curator.io.writer`."""

from __future__ import annotations

import pyarrow as pa
import pytest


def test_flatten_clips_orders_passing_before_filtered(fake_video):
    from spiraldb_nemo_curator.io.writer import _flatten_clips

    pairs = _flatten_clips(fake_video)
    flags = [filtered for _, filtered in pairs]
    assert flags == [False, False, True]
    spans = [c.span for c, _ in pairs]
    assert spans == [(0.0, 4.0), (4.0, 7.5), (7.5, 9.0)]


def test_embedding_dim_picks_first_non_null(fake_video):
    from spiraldb_nemo_curator.io.writer import _embedding_dim, _flatten_clips

    assert _embedding_dim(_flatten_clips(fake_video)) == 4


def test_embedding_dim_returns_none_when_all_missing(fake_video_clip_factory):
    from nemo_curator.tasks.video import Video, VideoMetadata

    from spiraldb_nemo_curator.io.writer import _embedding_dim, _flatten_clips

    v = Video(input_video=__import__("pathlib").Path("row-x"))
    v.metadata = VideoMetadata()
    v.clips = [fake_video_clip_factory(embedding=None)]
    assert _embedding_dim(_flatten_clips(v)) is None


def test_window_to_dict_only_includes_configured_caption_models():
    from nemo_curator.tasks.video import _Window

    from spiraldb_nemo_curator.io.writer import _window_to_dict

    w = _Window(start_frame=10, end_frame=60)
    w.caption.update({"qwen": "a cat", "internvl": "an animal"})
    w.enhanced_caption.update({"qwen": "tabby cat"})
    out = _window_to_dict(w, caption_models=["qwen"], enhanced_caption_models=["qwen"])
    assert out == {
        "start_frame": 10,
        "end_frame": 60,
        "captions": {"qwen": "a cat"},
        "enhanced_captions": {"qwen": "tabby cat"},
    }


def test_window_to_dict_omits_caption_fields_when_no_models():
    from nemo_curator.tasks.video import _Window

    from spiraldb_nemo_curator.io.writer import _window_to_dict

    w = _Window(start_frame=10, end_frame=60)
    out = _window_to_dict(w, caption_models=[], enhanced_caption_models=[])
    assert out == {"start_frame": 10, "end_frame": 60}


def test_writer_build_batch_has_expected_shape(fake_spiral, fake_blob, fake_video):
    from spiraldb_nemo_curator.io.writer import SpiralClipWriter

    writer = SpiralClipWriter(
        project_id="proj",
        table_name="clips",
        caption_models=["qwen"],
        enhanced_caption_models=["qwen"],
    )
    writer.setup()
    batch = writer._build_batch(fake_video)
    assert batch is not None

    # 2 passing clips + 1 filtered clip
    assert len(batch["source_video"]) == 3
    assert batch["source_video"] == ["row-1", "row-1", "row-1"]
    assert batch["filtered"].to_pylist() == [False, False, True]
    assert batch["valid"].to_pylist() == [True, True, False]
    # Each passing clip carries the fake embedding; the filtered clip is null.
    emb = batch["cosmos_embed1_embedding"]
    assert isinstance(emb.type, pa.FixedSizeListType)
    assert emb.type.list_size == 4
    assert emb.to_pylist()[0] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert emb.to_pylist()[2] is None

    # Errors round-trip as list<struct{stage, message}>
    errs = batch["errors"]
    assert errs.to_pylist() == [[], [], [{"stage": "motion", "message": "too low"}]]

    # Per-video metadata is denormalized.
    assert batch["source_width"].to_pylist() == [1280, 1280, 1280]
    assert batch["source_video_codec"].to_pylist() == ["h264"] * 3

    # Captions struct is keyed by model name and present on every window.
    win = batch["windows"].to_pylist()
    assert win[0][0]["captions"] == {"qwen": "a cat"}
    assert win[1][0]["captions"] == {"qwen": "a dog"}

    # Clip blob and preview blob are present (one BlobValue per row).
    assert len(batch["clip"]) == 3
    assert len(batch["preview"]) == 3


def test_writer_build_batch_skips_embedding_when_all_null(
    fake_spiral, fake_blob, fake_video_clip_factory
):
    from pathlib import Path

    from nemo_curator.tasks.video import Video, VideoMetadata

    from spiraldb_nemo_curator.io.writer import SpiralClipWriter

    v = Video(input_video=Path("row-9"))
    v.metadata = VideoMetadata(width=640, height=480, framerate=24.0)
    v.clips = [
        fake_video_clip_factory(source_video="row-9", span=(0.0, 1.0), embedding=None),
    ]

    writer = SpiralClipWriter(project_id="proj", table_name="clips")
    writer.setup()
    batch = writer._build_batch(v)
    assert batch is not None
    assert "cosmos_embed1_embedding" not in batch


def test_writer_process_writes_one_batch(fake_spiral, fake_blob, fake_video_task):
    from spiraldb_nemo_curator.io.writer import SpiralClipWriter

    writer = SpiralClipWriter(
        project_id="proj",
        table_name="clips",
        caption_models=["qwen"],
        enhanced_caption_models=["qwen"],
    )
    writer.setup()
    out = writer.process(fake_video_task)

    table = fake_spiral.project("proj").tables["clips"]
    assert len(table.writes) == 1
    batch = table.writes[0]
    assert batch["source_video"] == ["row-1", "row-1", "row-1"]
    # Curator-style cleanup: buffers and embeddings have been released.
    assert out.data.clips[0].buffer is None
    assert out.data.clips[0].cosmos_embed1_embedding is None


def test_writer_process_dry_run_does_not_write(fake_spiral, fake_blob, fake_video_task):
    from spiraldb_nemo_curator.io.writer import SpiralClipWriter

    writer = SpiralClipWriter(project_id="proj", table_name="clips", dry_run=True)
    writer.setup()
    writer.process(fake_video_task)

    table = fake_spiral.project("proj").tables["clips"]
    assert table.writes == []


def test_writer_process_empty_video_is_noop(fake_spiral, fake_blob):
    from pathlib import Path

    from nemo_curator.tasks.video import Video, VideoTask

    from spiraldb_nemo_curator.io.writer import SpiralClipWriter

    video = Video(input_video=Path("row-empty"))
    task = VideoTask(task_id="row-empty_processed", dataset_name="proj.clips", data=video)
    writer = SpiralClipWriter(project_id="proj", table_name="clips")
    writer.setup()
    writer.process(task)

    table = fake_spiral.project("proj").tables["clips"]
    assert table.writes == []


def test_writer_creates_table_with_expected_key_schema(fake_spiral, fake_blob):
    from spiraldb_nemo_curator.io.writer import OUTPUT_KEY_SCHEMA, SpiralClipWriter

    writer = SpiralClipWriter(project_id="proj", table_name="clips")
    writer.setup()
    table = fake_spiral.project("proj").tables["clips"]
    assert table.key_schema.names == OUTPUT_KEY_SCHEMA.names == ["source_video", "clip_uuid"]
