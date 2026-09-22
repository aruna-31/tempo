import os
import re
import uuid
from typing import Generator, Optional, Tuple
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from app.core.config import settings


class StorageService:
    @staticmethod
    def ensure_directories_exist():
        """Ensure storage folders for raw and output videos exist."""
        os.makedirs(settings.UPLOAD_RAW_DIR, exist_ok=True)
        os.makedirs(settings.OUTPUT_VIDEO_DIR, exist_ok=True)

    @staticmethod
    def validate_video_file(file: UploadFile) -> Tuple[str, str]:
        """
        Validate video file extension and MIME type.
        Returns (sanitized_original_filename, file_extension).
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename"
            )

        # Sanitize filename (remove path traversal, invalid characters)
        raw_name = os.path.basename(file.filename)
        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', raw_name)
        file_ext = os.path.splitext(clean_name)[1].lower()

        if file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported video format '{file_ext}'. "
                    f"Allowed formats: {', '.join(sorted(settings.ALLOWED_VIDEO_EXTENSIONS))}"
                )
            )

        # Check content-type if supplied
        if file.content_type:
            content_type = file.content_type.lower()
            if content_type not in settings.ALLOWED_VIDEO_MIME_TYPES:
                # Also accept generic video/ if valid extension
                if not content_type.startswith("video/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid MIME type '{file.content_type}' for video upload"
                    )

        return clean_name, file_ext

    @staticmethod
    def save_upload_file_sync(file: UploadFile, clean_name: str, file_ext: str) -> Tuple[str, int]:
        """
        Stream upload file to storage disk with strict size quota enforcement.
        Returns (absolute_file_path, file_size_bytes).
        """
        StorageService.ensure_directories_exist()
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        target_path = os.path.abspath(os.path.join(settings.UPLOAD_RAW_DIR, unique_filename))

        file_size = 0
        try:
            with open(target_path, "wb") as buffer:
                while chunk := file.file.read(1024 * 1024):  # 1MB buffer
                    file_size += len(chunk)
                    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
                        buffer.close()
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=(
                                f"Uploaded file exceeds maximum limit of "
                                f"{settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
                            )
                        )
                    buffer.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save video file to storage: {str(e)}"
            )

        return target_path, file_size

    @staticmethod
    def delete_file(file_path: Optional[str]) -> bool:
        """Safely remove a file from storage."""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def get_mime_type(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
        }
        return mime_map.get(ext, "video/mp4")

    @staticmethod
    def range_streamer(
        file_path: str, start: int, end: int, chunk_size: int = 1024 * 512
    ) -> Generator[bytes, None, None]:
        """Generator that yields file chunks within a specific byte range."""
        with open(file_path, "rb") as f:
            f.seek(start)
            bytes_left = end - start + 1
            while bytes_left > 0:
                read_size = min(chunk_size, bytes_left)
                data = f.read(read_size)
                if not data:
                    break
                bytes_left -= len(data)
                yield data

    @staticmethod
    def create_range_response(file_path: str, range_header: Optional[str]) -> StreamingResponse:
        """
        Creates an HTTP 206 Partial Content StreamingResponse or HTTP 200 StreamingResponse
        supporting byte ranges for video streaming and seeking.
        """
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found on storage"
            )

        file_size = os.path.getsize(file_path)
        content_type = StorageService.get_mime_type(file_path)

        if range_header:
            # Format: bytes=start-end (e.g. bytes=0-1048575 or bytes=1000-)
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header.strip())
            if range_match:
                start_str, end_str = range_match.groups()
                start = int(start_str)
                end = int(end_str) if end_str else file_size - 1

                if start >= file_size or end >= file_size or start > end:
                    raise HTTPException(
                        status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                        detail="Requested range not satisfiable",
                        headers={"Content-Range": f"bytes */{file_size}"}
                    )

                content_length = end - start + 1
                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                    "Content-Type": content_type,
                }
                return StreamingResponse(
                    StorageService.range_streamer(file_path, start, end),
                    status_code=status.HTTP_206_PARTIAL_CONTENT,
                    headers=headers
                )

        # Full file streaming
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }
        return StreamingResponse(
            StorageService.range_streamer(file_path, 0, file_size - 1),
            status_code=status.HTTP_200_OK,
            headers=headers
        )
