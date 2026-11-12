"""Structured JSON logging for CloudWatch Logs / Elasticsearch.

One JSON object per line (timestamp, level, logger, event and keyword context),
queryable in CloudWatch Logs Insights or Kibana without a custom parser.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(service_name: str) -> structlog.stdlib.BoundLogger:
    """Configure structlog to write one JSON object per line to stdout.

    `LOG_LEVEL` (default "INFO") sets the minimum level. Called once per process.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Third-party libraries (uvicorn, botocore, mlflow...) use stdlib logging;
    # this formatter renders them as the same JSON instead of mixing plain text
    # into the stream.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    return structlog.get_logger(service_name)
