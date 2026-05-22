"""Shared pytest fixtures.

These tests must run on a machine with both ``nemo_curator`` and a working
``pyspiral`` install. They never actually touch a Spiral cluster — the
``Spiral`` client is patched out for every test.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def fake_video_clip_factory():
    """Build a Curator ``Clip`` populated like a downstream stage would leave it."""
    from nemo_curator.tasks.video import Clip, _Window

    def _make(
        *,
        source_video: str = "row-1",
        span: tuple[float, float] = (0.0, 4.0),
        buffer: bytes | None = b"\x00mp4-bytes",
        embedding: list[float] | None = None,
        windows: list[tuple[int, int]] | None = None,
        captions: dict[str, str] | None = None,
        enhanced_captions: dict[str, str] | None = None,
        webp: bytes | None = b"\xffwebp-bytes",
        aesthetic: float | None = 0.7,
        motion_global: float | None = 0.2,
        motion_per_patch: float | None = 0.1,
        errors: dict[str, str] | None = None,
    ) -> Clip:
        clip = Clip(
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{source_video}_{span[0]}_{span[1]}"),
            source_video=source_video,
            span=span,
            buffer=buffer,
        )
        clip.aesthetic_score = aesthetic
        clip.motion_score_global_mean = motion_global
        clip.motion_score_per_patch_min_256 = motion_per_patch
        if embedding is not None:
            import numpy as np

            clip.cosmos_embed1_embedding = np.array(embedding, dtype="float32")
        for i, (s, e) in enumerate(windows or [(0, 60)]):
            w = _Window(start_frame=s, end_frame=e)
            if captions:
                w.caption.update(captions)
            if enhanced_captions:
                w.enhanced_caption.update(enhanced_captions)
            if i == 0 and webp is not None:
                w.webp_bytes = webp
            clip.windows.append(w)
        if errors:
            clip.errors.update(errors)
        return clip

    return _make


@pytest.fixture
def fake_video(fake_video_clip_factory):
    """Build a Curator ``Video`` with two passing clips and one filtered clip."""
    from nemo_curator.tasks.video import Video, VideoMetadata

    video = Video(input_video=Path("row-1"))
    video.metadata = VideoMetadata(
        size=10_000,
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
    video.clips = [
        fake_video_clip_factory(
            source_video="row-1",
            span=(0.0, 4.0),
            embedding=[0.1, 0.2, 0.3, 0.4],
            captions={"qwen": "a cat"},
            enhanced_captions={"qwen": "a tabby cat sitting"},
        ),
        fake_video_clip_factory(
            source_video="row-1",
            span=(4.0, 7.5),
            embedding=[0.5, 0.6, 0.7, 0.8],
            windows=[(120, 180)],
            captions={"qwen": "a dog"},
            enhanced_captions={"qwen": "a golden retriever running"},
        ),
    ]
    video.filtered_clips = [
        fake_video_clip_factory(
            source_video="row-1",
            span=(7.5, 9.0),
            buffer=None,
            embedding=None,
            windows=[(225, 270)],
            webp=None,
            errors={"motion": "too low"},
        )
    ]
    return video


@pytest.fixture
def fake_video_task(fake_video):
    from nemo_curator.tasks.video import VideoTask

    return VideoTask(task_id="row-1_processed", dataset_name="project.clips", data=fake_video)


class _FakeKeySchema:
    def __init__(self, names: list[str]) -> None:
        self.names = list(names)


class _FakeTable:
    def __init__(self, name: str, key_names: list[str]) -> None:
        self.name = name
        self.key_schema = _FakeKeySchema(key_names)
        self.writes: list[Any] = []

    def __getitem__(self, item: str) -> str:  # tracked for filter construction
        return f"col:{item}"

    def write(self, batch: Any, **kwargs: Any) -> None:
        self.writes.append(batch)


class _FakeScan:
    def __init__(self, table: Any) -> None:
        self._table = table

    def to_table(self) -> Any:
        return self._table


class _FakeProject:
    def __init__(self) -> None:
        self.tables: dict[str, _FakeTable] = {}

    def table(self, identifier: str) -> _FakeTable:
        return self.tables[identifier]

    def create_table(
        self, identifier: str, *, key_schema: Any, exist_ok: bool = False
    ) -> _FakeTable:
        names = list(key_schema.names) if hasattr(key_schema, "names") else list(key_schema.keys())
        return self.tables.setdefault(identifier, _FakeTable(identifier, names))


class FakeSpiral:
    """In-memory stand-in for ``spiral.Spiral`` used in tests."""

    def __init__(self) -> None:
        self.projects: dict[str, _FakeProject] = {}
        self.scan_responses: list[Any] = []
        self.scan_keys_responses: list[Any] = []

    def project(self, project_id: str) -> _FakeProject:
        return self.projects.setdefault(project_id, _FakeProject())

    def scan(self, *args: Any, **kwargs: Any) -> _FakeScan:
        if not self.scan_responses:
            raise AssertionError("FakeSpiral.scan called with no queued response")
        return _FakeScan(self.scan_responses.pop(0))

    def scan_keys(self, *args: Any, **kwargs: Any) -> _FakeScan:
        if not self.scan_keys_responses:
            raise AssertionError("FakeSpiral.scan_keys called with no queued response")
        return _FakeScan(self.scan_keys_responses.pop(0))


@pytest.fixture
def fake_spiral(monkeypatch):
    fake = FakeSpiral()
    monkeypatch.setattr("spiraldb_nemo_curator.io.reader.Spiral", lambda: fake, raising=False)
    monkeypatch.setattr("spiraldb_nemo_curator.io.writer.Spiral", lambda: fake, raising=False)
    # Spiral is imported lazily inside setup(); patch the module attribute that the
    # `from spiral import Spiral` statement will resolve.
    import spiral

    monkeypatch.setattr(spiral, "Spiral", lambda: fake)
    yield fake


@pytest.fixture
def fake_blob(monkeypatch):
    """Replace ``se.Blob`` with a record-only stand-in so tests don't need native blob handling."""
    from spiral import expressions as se

    sentinel = MagicMock()

    def _blob(value=None, **kwargs):
        rec = MagicMock(name="BlobValue")
        rec.value = value
        rec.kwargs = kwargs
        rec.sentinel = sentinel
        return rec

    monkeypatch.setattr(se, "Blob", _blob)
    return sentinel
