import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.analysis import AnalysisJob
from app.models.video import Video

logger = logging.getLogger("tempo.services.storage")


class StorageManagerService:
    @staticmethod
    def get_storage_stats() -> Dict[str, Any]:
        """
        Calculates disk usage across raw videos, processed artifacts, and dropzone directories.
        """
        def get_dir_size(path: str) -> int:
            total = 0
            if not os.path.exists(path):
                return 0
            for root, _, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
            return total

        raw_size = get_dir_size(settings.UPLOAD_RAW_DIR)
        out_size = get_dir_size(settings.OUTPUT_VIDEO_DIR)
        drop_size = get_dir_size(settings.AUTO_INGEST_DROPZONE_DIR)

        return {
            "raw_videos_bytes": raw_size,
            "output_videos_bytes": out_size,
            "dropzone_bytes": drop_size,
            "total_storage_bytes": raw_size + out_size + drop_size,
            "retention_days": settings.STORAGE_RETENTION_DAYS,
            "purge_raw_videos_enabled": settings.PURGE_RAW_VIDEO_AFTER_ANALYSIS
        }

    @staticmethod
    def enforce_retention_policy(db: Session) -> Dict[str, int]:
        """
        Purges raw video binaries older than retention period or already analyzed if configured.
        Keeps database analytics and reports intact.
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=settings.STORAGE_RETENTION_DAYS)
        purged_count = 0
        freed_bytes = 0

        # Query analyzed videos older than retention policy
        videos = db.query(Video).filter(
            Video.created_at < cutoff_date,
            Video.status == "ANALYZED"
        ).all()

        for vid in videos:
            if vid.file_path and os.path.exists(vid.file_path):
                try:
                    fsize = os.path.getsize(vid.file_path)
                    os.remove(vid.file_path)
                    purged_count += 1
                    freed_bytes += fsize
                    vid.file_path = f"[PURGED_BY_POLICY_{datetime.now().strftime('%Y%m%d')}]"
                except Exception as e:
                    logger.error(f"Error purging video file {vid.file_path}: {str(e)}")

        db.commit()
        logger.info(f"Retention policy applied: Purged {purged_count} files, freed {freed_bytes / (1024*1024):.2f} MB")
        return {
            "purged_count": purged_count,
            "freed_bytes": freed_bytes
        }
