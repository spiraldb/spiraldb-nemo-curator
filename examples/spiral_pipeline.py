"""End-to-end NeMo Curator video pipeline backed by SpiralDB I/O.

The middle stages (TransNetV2 clip extraction, motion filtering, aesthetic
filtering, Cosmos-Embed1, captioning) are reused unchanged from
``nemo_curator``; only the read and write endpoints are swapped for the
SpiralDB-backed equivalents.

This example assumes:

* A Spiral project ``$SPIRAL_PROJECT`` exists.
* A source table ``$SPIRAL_PROJECT.input_videos`` holds your raw MP4s in a
  ``se.Blob`` column named ``video`` and uses ``clip_id: string`` as its key.
* The destination table ``$SPIRAL_PROJECT.clips`` will be auto-created on
  first run (key schema: ``source_video: string, clip_uuid: string``).

Run on a GPU host with both ``nemo_curator`` and ``pyspiral`` installed:

    SPIRAL_PROJECT=proj_abc python examples/spiral_pipeline.py
"""

from __future__ import annotations

import os

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from nemo_curator.stages.video.clipping.clip_extraction_stages import FixedStrideExtractorStage
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.stages.video.clipping.frame_extraction import FrameExtractionStage
from nemo_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage
from nemo_curator.stages.video.clipping.video_frame_extraction import (
    VideoFrameExtractionStage,
)  # noqa: F401 -- imported for parity with Curator's example
from nemo_curator.stages.video.clipping.clip_transcoding_stages import (
    ClipTranscodingStage,
)
from nemo_curator.stages.video.embedding.cosmos_embed1_stages import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from nemo_curator.stages.video.filtering.motion_filter import MotionFilterStage
from nemo_curator.stages.video.preview.preview import PreviewStage

from spiraldb_nemo_curator.io.video import SpiralClipWriter, SpiralVideoReader


def build_pipeline() -> Pipeline:
    project_id = os.environ["SPIRAL_PROJECT"]
    input_table = os.environ.get("SPIRAL_INPUT_TABLE", "input_videos")
    output_table = os.environ.get("SPIRAL_OUTPUT_TABLE", "clips")

    reader = SpiralVideoReader(
        project_id=project_id,
        table_name=input_table,
        video_column="video",
        source_id_column="clip_id",
        verbose=True,
    )

    writer = SpiralClipWriter(
        project_id=project_id,
        table_name=output_table,
        caption_models=["qwen"],
        enhanced_caption_models=["qwen"],
        verbose=True,
    )

    pipeline = Pipeline(name="spiral_video_curation")
    pipeline.add_stage(reader)

    # Curator's standard video curation chain — these stages are unchanged.
    pipeline.add_stage(FrameExtractionStage())
    pipeline.add_stage(TransNetV2ClipExtractionStage())
    # Or, for fixed-stride clips: FixedStrideExtractorStage()
    pipeline.add_stage(ClipTranscodingStage(encoder="libopenh264"))
    pipeline.add_stage(MotionFilterStage())
    pipeline.add_stage(ClipFrameExtractionStage())
    pipeline.add_stage(CosmosEmbed1FrameCreationStage())
    pipeline.add_stage(CosmosEmbed1EmbeddingStage())
    pipeline.add_stage(PreviewStage())
    pipeline.add_stage(CaptionGenerationStage(model_variant="qwen"))

    pipeline.add_stage(writer)
    return pipeline


def main() -> None:
    pipeline = build_pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
