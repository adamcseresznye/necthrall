from typing import List, Dict, Optional
from loguru import logger


def log_pipeline_stage(
    stage_name: str,
    input_count: int,
    output_count: int,
    success_rate: Optional[float] = None,
    drop_reasons: Optional[List[str]] = None,
) -> None:
    """Standard logging format for pipeline stages.

    Args:
        stage_name: Friendly stage name (e.g., 'SearchAgent')
        input_count: Number of papers entering the stage
        output_count: Number of papers leaving the stage
        success_rate: Optional success rate (0-1)
        drop_reasons: Optional list of reasons why papers were dropped
    """
    try:
        logger.info(f"📊 PIPELINE STAGE: {stage_name}")
        logger.info(f"├─ Input: {input_count} papers")
        logger.info(f"├─ Output: {output_count} papers")

        if success_rate is not None:
            logger.info(f"├─ Success Rate: {success_rate:.1%}")

        retention_rate = (output_count / input_count * 100) if input_count > 0 else 0
        logger.info(f"├─ Retention: {retention_rate:.1f}%")

        if drop_reasons:
            logger.info(f"└─ Drop Reasons: {', '.join(drop_reasons)}")
        else:
            logger.info(f"└─ Status: Completed")

    except Exception as e:
        logger.warning(f"Failed to log pipeline stage {stage_name}: {e}")


def log_pipeline_summary(stages_data: Dict[str, int]) -> None:
    """Generate complete pipeline attrition summary.

    stages_data should contain keys like:
      'openalex_available', 'search_agent', 'filtering_agent', 'processing_agent', 'synthesis_ready'
    """
    try:
        initial_count = stages_data.get("openalex_available", 0)
        final_count = stages_data.get(
            "synthesis_ready", stages_data.get("processing_agent", 0)
        )

        logger.info(f"📈 COMPLETE PIPELINE SUMMARY:")
        logger.info(f"├─ OpenAlex Available: {initial_count} papers")

        for stage, count in stages_data.items():
            if stage == "openalex_available":
                continue
            percentage = (count / initial_count * 100) if initial_count > 0 else 0
            logger.info(
                f"├─ {stage.replace('_', ' ').title()}: {count} papers ({percentage:.1f}%)"
            )

        total_retention = (
            (final_count / initial_count * 100) if initial_count > 0 else 0
        )
        logger.info(f"└─ Overall Retention: {total_retention:.1f}%")

    except Exception as e:
        logger.warning(f"Failed to log pipeline summary: {e}")
