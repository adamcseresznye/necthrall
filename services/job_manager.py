"""In-memory job manager for async background job processing."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from loguru import logger


JobStatus = Literal["pending", "processing", "completed", "failed"]


class JobManager:
    """Simple in-memory job manager for background task tracking.

    Usage:
        manager = JobManager()
        job_id = manager.create_job()
        manager.update_job(job_id, "processing")
        manager.update_job(job_id, "completed", result=response)
        job = manager.get_job(job_id)
    """

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self) -> str:
        """Create a new job with pending status.

        Returns:
            The UUID string for the new job.
        """
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "result": None,
            "error": None,
        }
        logger.info(f"📋 Created job {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a job by ID.

        Args:
            job_id: The UUID string of the job.

        Returns:
            The job dict if found, None otherwise.
        """
        job = self.jobs.get(job_id)
        if job is None:
            logger.warning(f"⚠️ Job {job_id} not found")
        return job

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update a job's status and optionally its result or error.

        Args:
            job_id: The UUID string of the job.
            status: New status ("pending", "processing", "completed", "failed").
            result: The QueryResponse object (when completed).
            error: Error message (if failed).

        Returns:
            True if job was updated, False if job not found.
        """
        job = self.jobs.get(job_id)
        if job is None:
            logger.warning(f"⚠️ Cannot update job {job_id}: not found")
            return False

        job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error

        logger.info(f"🔄 Updated job {job_id} to status: {status}")
        return True


# Global singleton instance
job_manager = JobManager()


__all__ = ["JobManager", "job_manager", "JobStatus"]
