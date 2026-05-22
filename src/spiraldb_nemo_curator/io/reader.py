"""SpiralDB-backed replacement for NeMo Curator's :class:`VideoReader`.

The composite stage decomposes into two execution stages:

* :class:`SpiralPartitionStage` — keys-only scan that emits one
  :class:`SpiralRowTask` per source row.
* :class:`SpiralVideoReaderStage` — keyed point-scan that materializes the
  blob payload into a :class:`Video` and populates metadata.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task, _EmptyTask
from nemo_curator.tasks.video import Video, VideoTask

if TYPE_CHECKING:
    from spiral import Project, Spiral, Table
    from spiral.expressions import ExprLike


@dataclass
class SpiralRowTask(Task[dict[str, Any]]):
    """Task identifying a single source row by its primary key.

    The :class:`Task` ``data`` payload is a ``{key_column: value}`` dict that
    forms a complete primary-key tuple for the source table. ``source_id`` is
    the resolved string used downstream as ``Video.input_video`` — it must be
    byte-identical across reruns for clip-uuid idempotency.
    """

    data: dict[str, Any] = field(default_factory=dict)
    source_id: str = ""

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        return bool(self.source_id)


def _resolve_source_id(
    key_row: dict[str, Any],
    key_columns: list[str],
    source_id_column: str | None,
) -> str:
    """Resolve the source identifier for a row.

    If ``source_id_column`` is set, returns ``str(key_row[source_id_column])``.
    Otherwise, joins every key column's stringified value with ``"/"``.
    """
    if source_id_column is not None:
        return str(key_row[source_id_column])
    return "/".join(str(key_row[c]) for c in key_columns)


@dataclass
class SpiralPartitionStage(ProcessingStage[_EmptyTask, SpiralRowTask]):
    """Keys-only scan of the source table.

    Emits one :class:`SpiralRowTask` per source row. The scan is built lazily
    in :meth:`setup` so the stage pickles cleanly across distributed workers.
    """

    project_id: str
    table_name: str
    source_id_column: str | None = None
    where: "ExprLike | None" = None
    limit: int | None = None
    verbose: bool = False
    name: str = "spiral_partition"

    def __post_init__(self) -> None:
        self.resources = Resources(cpus=0.5)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002
        from spiral import Spiral

        self._spiral: Spiral = Spiral()
        self._project = self._spiral.project(self.project_id)
        self._table = self._project.table(self.table_name)
        self._key_columns: list[str] = list(self._table.key_schema.names)

        if self.source_id_column is not None and self.source_id_column not in self._key_columns:
            msg = (
                f"source_id_column={self.source_id_column!r} is not part of the "
                f"key schema for {self.project_id}.{self.table_name} "
                f"(keys: {self._key_columns}). For row identity to be stable, "
                "the source id must be a key column."
            )
            raise ValueError(msg)

    def process(self, _: _EmptyTask) -> list[SpiralRowTask]:
        scan = self._spiral.scan_keys(
            self._table,
            where=self.where,
            limit=self.limit,
            hide_progress_bar=not self.verbose,
        )
        keys_table: pa.Table = scan.to_table()

        rows = keys_table.to_pylist()
        if self.verbose:
            logger.info(
                f"SpiralPartitionStage: {len(rows)} row(s) from "
                f"{self.project_id}.{self.table_name}"
                + (f" (limit={self.limit})" if self.limit else "")
            )

        dataset_name = f"{self.project_id}.{self.table_name}"
        tasks: list[SpiralRowTask] = []
        for i, key_row in enumerate(rows):
            source_id = _resolve_source_id(key_row, self._key_columns, self.source_id_column)
            tasks.append(
                SpiralRowTask(
                    task_id=f"{dataset_name}#{i}",
                    dataset_name=dataset_name,
                    data=key_row,
                    source_id=source_id,
                )
            )
        return tasks


@dataclass
class SpiralVideoReaderStage(ProcessingStage[SpiralRowTask, VideoTask]):
    """Keyed point-scan that resolves one source row into a :class:`VideoTask`.

    The Spiral handles are constructed lazily in :meth:`setup` so the stage
    pickles cleanly across workers; the value scan filters on the row's key
    columns and projects ``se.blob.bytes(table[video_column])``.
    """

    project_id: str
    table_name: str
    video_column: str = "video"
    source_id_column: str | None = None
    verbose: bool = False
    name: str = "spiral_video_reader"

    def __post_init__(self) -> None:
        self.resources = Resources(cpus=1.0)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["source_bytes", "metadata"]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002
        from spiral import Spiral

        self._spiral: Spiral = Spiral()
        self._project = self._spiral.project(self.project_id)
        self._table = self._project.table(self.table_name)
        self._key_columns: list[str] = list(self._table.key_schema.names)

    def _build_where(self, key_row: dict[str, Any]) -> Any:
        from spiral import expressions as se

        clauses = [self._table[col] == key_row[col] for col in self._key_columns]
        return se.and_(*clauses)

    def process(self, task: SpiralRowTask) -> VideoTask:
        from spiral import expressions as se

        where = self._build_where(task.data)
        scan = self._spiral.scan(
            {"video_bytes": se.blob.bytes(self._table[self.video_column])},
            where=where,
            limit=1,
            hide_progress_bar=not self.verbose,
        )
        result: pa.Table = scan.to_table()

        if result.num_rows == 0:
            msg = (
                f"SpiralVideoReaderStage: no row for source_id={task.source_id!r} "
                f"in {self.project_id}.{self.table_name}"
            )
            raise RuntimeError(msg)

        video_bytes = result.column("video_bytes")[0].as_py()
        if not isinstance(video_bytes, (bytes, bytearray, memoryview)):
            msg = (
                f"SpiralVideoReaderStage: expected bytes for "
                f"{self.video_column!r}, got {type(video_bytes).__name__}"
            )
            raise TypeError(msg)

        video = Video(input_video=Path(task.source_id))
        video.source_bytes = bytes(video_bytes)

        try:
            video.populate_metadata()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"SpiralVideoReaderStage: failed to extract metadata for "
                f"{task.source_id}: {e}"
            )
            video.errors["metadata"] = str(e)

        if self.verbose and video.metadata.size is not None:
            logger.info(
                f"SpiralVideoReaderStage: read source_id={task.source_id} "
                f"size={video.metadata.size}B "
                f"res={video.metadata.width}x{video.metadata.height} "
                f"fps={video.metadata.framerate} "
                f"duration={video.metadata.duration}"
            )

        return VideoTask(
            task_id=f"{task.source_id}_processed",
            dataset_name=task.dataset_name,
            data=video,
            _metadata=deepcopy(task._metadata),
            _stage_perf=deepcopy(task._stage_perf),
        )


@dataclass
class SpiralVideoReader(CompositeStage[_EmptyTask, VideoTask]):
    """Composite reader that streams videos out of a SpiralDB table.

    Decomposes into :class:`SpiralPartitionStage` (keys-only scan) followed by
    :class:`SpiralVideoReaderStage` (one keyed point-scan per source row).

    Args:
        project_id: Spiral project containing the source table.
        table_name: Source table identifier, in the form ``dataset.table`` or
            ``table`` (using the ``default`` dataset).
        video_column: Name of the :class:`se.Blob` column holding the MP4
            payload. Defaults to ``"video"``.
        source_id_column: Optional key column whose value is used as the
            ``source_id`` propagated into :class:`Video.input_video`. When
            ``None``, every key column is stringified and joined by ``"/"``.
        where: Optional Spiral filter expression applied to the keys scan.
        limit: Optional maximum number of source rows.
        verbose: When ``True``, emits per-stage progress logs.
    """

    project_id: str
    table_name: str
    video_column: str = "video"
    source_id_column: str | None = None
    where: "ExprLike | None" = None
    limit: int | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        super().__init__()
        self.name = "spiral_video_reader"

    def decompose(self) -> list[ProcessingStage]:
        partition_stage = SpiralPartitionStage(
            project_id=self.project_id,
            table_name=self.table_name,
            source_id_column=self.source_id_column,
            where=self.where,
            limit=self.limit,
            verbose=self.verbose,
        )
        reader_stage = SpiralVideoReaderStage(
            project_id=self.project_id,
            table_name=self.table_name,
            video_column=self.video_column,
            source_id_column=self.source_id_column,
            verbose=self.verbose,
        )
        return [partition_stage, reader_stage]

    def get_description(self) -> str:
        return (
            f"Reads videos from SpiralDB table {self.project_id}.{self.table_name} "
            f"(video column: {self.video_column!r}, "
            f"limit: {self.limit if self.limit is not None else 'unlimited'})"
        )
