"""SpiralDB-backed replacement for NeMo Curator's :class:`ClipWriterStage`.

Writes one Arrow batch per :class:`VideoTask`, covering both ``video.clips``
and ``video.filtered_clips`` in a single ``tbl.write()`` (implicit transaction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask

if TYPE_CHECKING:
    from collections.abc import Sequence

OUTPUT_KEY_SCHEMA = pa.schema(
    [
        pa.field("source_video", pa.string()),
        pa.field("clip_uuid", pa.string()),
    ]
)


def _windows_struct_type(
    caption_models: Sequence[str],
    enhanced_caption_models: Sequence[str],
) -> pa.StructType:
    """Build the per-window struct type, gated on configured caption models."""
    fields: list[pa.Field] = [
        pa.field("start_frame", pa.int32()),
        pa.field("end_frame", pa.int32()),
    ]
    if caption_models:
        fields.append(
            pa.field(
                "captions",
                pa.struct([pa.field(m, pa.string()) for m in caption_models]),
            )
        )
    if enhanced_caption_models:
        fields.append(
            pa.field(
                "enhanced_captions",
                pa.struct([pa.field(m, pa.string()) for m in enhanced_caption_models]),
            )
        )
    return pa.struct(fields)


def _errors_struct_type() -> pa.StructType:
    return pa.struct(
        [
            pa.field("stage", pa.string()),
            pa.field("message", pa.string()),
        ]
    )


def _window_to_dict(
    window: Any,
    caption_models: Sequence[str],
    enhanced_caption_models: Sequence[str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "start_frame": int(window.start_frame),
        "end_frame": int(window.end_frame),
    }
    if caption_models:
        entry["captions"] = {m: window.caption.get(m) for m in caption_models}
    if enhanced_caption_models:
        entry["enhanced_captions"] = {
            m: window.enhanced_caption.get(m) for m in enhanced_caption_models
        }
    return entry


def _errors_to_list(errors: dict[str, str]) -> list[dict[str, str]]:
    return [{"stage": stage, "message": message} for stage, message in errors.items()]


def _flatten_clips(video: Video) -> list[tuple[Clip, bool]]:
    """Pair every clip with its ``filtered`` flag in a single sequence."""
    items: list[tuple[Clip, bool]] = [(c, False) for c in video.clips]
    items.extend((c, True) for c in video.filtered_clips)
    return items


def _embedding_dim(clips: Sequence[tuple[Clip, bool]]) -> int | None:
    """Find the first non-null embedding dimension, or ``None`` if all null."""
    for clip, _ in clips:
        emb = clip.cosmos_embed1_embedding
        if emb is None:
            continue
        return int(emb.reshape(-1).shape[0])
    return None


@dataclass
class SpiralClipWriter(ProcessingStage[VideoTask, VideoTask]):
    """Write clips and filtered clips to a SpiralDB table in one batched write.

    Replaces Curator's :class:`ClipWriterStage`. The output schema is a single
    wide ``clips`` table keyed by ``(source_video, clip_uuid)``. Per-video
    metadata is denormalized onto every clip row; the per-clip MP4 and the
    first window's WebP preview live in their own :class:`se.Blob` columns
    (each automatically gets its own column group).

    Args:
        project_id: Spiral project hosting the output table.
        table_name: Output table identifier, in the form ``dataset.table`` or
            ``table`` (using the ``default`` dataset).
        caption_models: Caption model names to surface as fields inside each
            window's ``captions`` struct. Each per-window record will contain
            one ``string`` field per model; missing captions are stored as
            null.
        enhanced_caption_models: Same as ``caption_models``, but for enhanced
            captions.
        dry_run: When ``True``, skip the actual write and just return the task.
        verbose: When ``True``, emit informational logs.
    """

    project_id: str
    table_name: str
    caption_models: Sequence[str] = field(default_factory=list)
    enhanced_caption_models: Sequence[str] = field(default_factory=list)
    dry_run: bool = False
    verbose: bool = False
    name: str = "spiral_clip_writer"

    def __post_init__(self) -> None:
        self.resources = Resources(cpus=0.5)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002
        from spiral import Spiral

        self._spiral = Spiral()
        self._project = self._spiral.project(self.project_id)
        self._table = self._project.create_table(
            self.table_name,
            key_schema=OUTPUT_KEY_SCHEMA,
            exist_ok=True,
        )

    def _build_batch(self, video: Video) -> dict[str, Any] | None:
        from spiral import expressions as se

        pairs = _flatten_clips(video)
        if not pairs:
            return None

        meta: VideoMetadata = video.metadata
        windows_type = _windows_struct_type(self.caption_models, self.enhanced_caption_models)
        errors_type = _errors_struct_type()

        source_video: list[str] = []
        clip_uuid: list[str] = []
        span_start: list[float] = []
        span_end: list[float] = []
        duration: list[float] = []
        filtered_col: list[bool] = []
        valid_col: list[bool] = []
        aesthetic: list[float | None] = []
        motion_global: list[float | None] = []
        motion_per_patch: list[float | None] = []
        errors_col: list[list[dict[str, str]]] = []
        windows_col: list[list[dict[str, Any]]] = []

        clip_buffers: list[bytes | None] = []
        preview_buffers: list[bytes | None] = []

        embedding_dim = _embedding_dim(pairs)
        embedding_values: list[list[float] | None] = []

        for clip, is_filtered in pairs:
            source_video.append(str(clip.source_video))
            clip_uuid.append(str(clip.uuid))
            span_start.append(float(clip.span[0]))
            span_end.append(float(clip.span[1]))
            duration.append(float(clip.span[1]) - float(clip.span[0]))
            filtered_col.append(is_filtered)
            valid_col.append(bool(clip.buffer and clip.windows))
            aesthetic.append(
                float(clip.aesthetic_score) if clip.aesthetic_score is not None else None
            )
            motion_global.append(
                float(clip.motion_score_global_mean)
                if clip.motion_score_global_mean is not None
                else None
            )
            motion_per_patch.append(
                float(clip.motion_score_per_patch_min_256)
                if clip.motion_score_per_patch_min_256 is not None
                else None
            )
            errors_col.append(_errors_to_list(clip.errors))
            windows_col.append(
                [
                    _window_to_dict(w, self.caption_models, self.enhanced_caption_models)
                    for w in clip.windows
                ]
            )

            clip_buffers.append(bytes(clip.buffer) if clip.buffer else None)
            first_webp = (
                clip.windows[0].webp_bytes if clip.windows and clip.windows[0].webp_bytes else None
            )
            preview_buffers.append(bytes(first_webp) if first_webp else None)

            if embedding_dim is not None:
                emb = clip.cosmos_embed1_embedding
                embedding_values.append(
                    emb.reshape(-1).astype("float32").tolist() if emb is not None else None
                )

        batch: dict[str, Any] = {
            "source_video": source_video,
            "clip_uuid": clip_uuid,
            "span_start": pa.array(span_start, type=pa.float64()),
            "span_end": pa.array(span_end, type=pa.float64()),
            "duration": pa.array(duration, type=pa.float64()),
            "filtered": pa.array(filtered_col, type=pa.bool_()),
            "valid": pa.array(valid_col, type=pa.bool_()),
            "aesthetic_score": pa.array(aesthetic, type=pa.float32()),
            "motion_score_global_mean": pa.array(motion_global, type=pa.float32()),
            "motion_score_per_patch_min_256": pa.array(motion_per_patch, type=pa.float32()),
            "errors": pa.array(errors_col, type=pa.list_(errors_type)),
            "windows": pa.array(windows_col, type=pa.list_(windows_type)),
            "source_width": pa.array([meta.width] * len(pairs), type=pa.int32()),
            "source_height": pa.array([meta.height] * len(pairs), type=pa.int32()),
            "source_framerate": pa.array([meta.framerate] * len(pairs), type=pa.float32()),
            "source_num_frames": pa.array([meta.num_frames] * len(pairs), type=pa.int64()),
            "source_duration": pa.array([meta.duration] * len(pairs), type=pa.float64()),
            "source_video_codec": pa.array([meta.video_codec] * len(pairs), type=pa.string()),
            "source_pixel_format": pa.array([meta.pixel_format] * len(pairs), type=pa.string()),
            "source_audio_codec": pa.array([meta.audio_codec] * len(pairs), type=pa.string()),
            "source_bit_rate_k": pa.array([meta.bit_rate_k] * len(pairs), type=pa.int32()),
            "clip": [
                se.Blob(b, mime_type="video/mp4") if b is not None else se.Blob(b"")
                for b in clip_buffers
            ],
            "preview": [
                se.Blob(b, mime_type="image/webp") if b is not None else se.Blob(b"")
                for b in preview_buffers
            ],
        }

        if embedding_dim is not None:
            batch["cosmos_embed1_embedding"] = pa.array(
                embedding_values,
                type=pa.list_(pa.float32(), embedding_dim),
            )

        return batch

    def _release_clip_memory(self, video: Video) -> None:
        """Mirror Curator's post-write cleanup: drop intermediate payloads."""
        for clip in list(video.clips) + list(video.filtered_clips):
            clip.buffer = None
            clip.cosmos_embed1_embedding = None
            for window in clip.windows:
                window.mp4_bytes = None
                window.llm_inputs.clear()
                window.caption.clear()
                window.enhanced_caption.clear()
                window.webp_bytes = None

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data
        batch = self._build_batch(video)
        if batch is None:
            if self.verbose:
                logger.info(f"SpiralClipWriter: no clips for {video.input_path}, skipping write")
            self._release_clip_memory(video)
            return task

        n_rows = len(batch["source_video"])
        if self.dry_run:
            if self.verbose:
                logger.info(
                    f"SpiralClipWriter: dry_run, would have written {n_rows} clip(s) "
                    f"for {video.input_path}"
                )
            self._release_clip_memory(video)
            return task

        self._table.write(batch)

        if self.verbose:
            logger.info(
                f"SpiralClipWriter: wrote {n_rows} clip(s) for {video.input_path} "
                f"to {self.project_id}.{self.table_name}"
            )

        self._release_clip_memory(video)
        return task
