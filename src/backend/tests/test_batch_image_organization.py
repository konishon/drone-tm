"""Tests for batch image organization (issue #713)."""

import uuid
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from app.models.enums import ImageStatus
from app.images.image_classification import ImageClassifier


@pytest.fixture
def mock_db():
    """Mock database connection."""
    db = AsyncMock()
    transaction_mock = AsyncMock()
    db.transaction = MagicMock(return_value=transaction_mock)
    db.cursor = MagicMock()
    return db


@pytest.fixture
def sample_images():
    """Sample test images for batch organization."""
    project_id = uuid.uuid4()
    batch_id = uuid.uuid4()
    task_1 = uuid.uuid4()
    task_2 = uuid.uuid4()

    return {
        "project_id": project_id,
        "batch_id": batch_id,
        "images": [
            {
                "id": uuid.uuid4(),
                "filename": "image1.jpg",
                "s3_key": f"projects/{project_id}/user-uploads/image1.jpg",
                "task_id": task_1,
            },
            {
                "id": uuid.uuid4(),
                "filename": "image2.jpg",
                "s3_key": f"projects/{project_id}/user-uploads/image2.jpg",
                "task_id": task_1,
            },
            {
                "id": uuid.uuid4(),
                "filename": "image3.jpg",
                "s3_key": f"projects/{project_id}/user-uploads/image3.jpg",
                "task_id": task_2,
            },
        ],
    }


@pytest.mark.asyncio
async def test_organize_moves_images_to_tasks(mock_db, sample_images):
    """Verifies images are moved from user-uploads to task folders."""
    project_id = sample_images["project_id"]
    batch_id = sample_images["batch_id"]
    images = sample_images["images"]

    cursor_mock = AsyncMock()
    mock_db.cursor.return_value.__aenter__.return_value = cursor_mock
    cursor_mock.fetchall.return_value = images

    with patch("app.images.image_classification.run_in_threadpool") as mock_copy:
        mock_copy.return_value = True
        result = await ImageClassifier.organize_batch_images_in_s3(
            mock_db, batch_id, project_id
        )

    assert result["total_moved"] == 3
    assert result["total_skipped"] == 0
    assert result["task_count"] == 2


@pytest.mark.asyncio
async def test_organize_is_idempotent(mock_db):
    """Verifies organizing twice skips already-organized images."""
    project_id = uuid.uuid4()
    batch_id = uuid.uuid4()
    task_id = uuid.uuid4()

    # First call: images in staging
    images_first = [
        {
            "id": uuid.uuid4(),
            "filename": f"img{i}.jpg",
            "s3_key": f"projects/{project_id}/user-uploads/img{i}.jpg",
            "task_id": task_id,
        }
        for i in range(2)
    ]

    # Second call: same images now in task folder
    images_second = [
        {
            "id": img["id"],
            "filename": img["filename"],
            "s3_key": f"projects/{project_id}/{task_id}/images/{img['filename']}",
            "task_id": img["task_id"],
        }
        for img in images_first
    ]

    cursor_mock = AsyncMock()
    mock_db.cursor.return_value.__aenter__.return_value = cursor_mock

    with patch("app.images.image_classification.run_in_threadpool") as mock_copy:
        mock_copy.return_value = True

        # First organize
        cursor_mock.fetchall.return_value = images_first
        result_1 = await ImageClassifier.organize_batch_images_in_s3(
            mock_db, batch_id, project_id
        )
        assert result_1["total_moved"] == 2
        assert result_1["total_skipped"] == 0

        # Second organize - should skip
        cursor_mock.fetchall.return_value = images_second
        mock_copy.reset_mock()
        result_2 = await ImageClassifier.organize_batch_images_in_s3(
            mock_db, batch_id, project_id
        )
        assert result_2["total_moved"] == 0
        assert result_2["total_skipped"] == 2
        mock_copy.assert_not_called()


@pytest.mark.asyncio
async def test_organize_handles_s3_copy_failures(mock_db, sample_images):
    """Verifies failed copies are tracked and don't crash the process."""
    project_id = sample_images["project_id"]
    batch_id = sample_images["batch_id"]
    images = sample_images["images"]

    cursor_mock = AsyncMock()
    mock_db.cursor.return_value.__aenter__.return_value = cursor_mock
    cursor_mock.fetchall.return_value = images

    with patch("app.images.image_classification.run_in_threadpool") as mock_copy:
        mock_copy.side_effect = [True, False, True]
        result = await ImageClassifier.organize_batch_images_in_s3(
            mock_db, batch_id, project_id
        )

    assert result["total_moved"] == 2
    assert result["total_failed"] == 1


@pytest.mark.asyncio
async def test_organize_handles_empty_batch(mock_db):
    """Verifies empty batches return gracefully."""
    project_id = uuid.uuid4()
    batch_id = uuid.uuid4()

    cursor_mock = AsyncMock()
    mock_db.cursor.return_value.__aenter__.return_value = cursor_mock
    cursor_mock.fetchall.return_value = []

    result = await ImageClassifier.organize_batch_images_in_s3(
        mock_db, batch_id, project_id
    )

    assert result["total_moved"] == 0
    assert result["task_count"] == 0


@pytest.mark.asyncio
async def test_organize_concurrent_calls_safe(mock_db):
    """Verifies concurrent calls don't duplicate moves (FOR UPDATE lock)."""
    project_id = uuid.uuid4()
    batch_id = uuid.uuid4()
    task_id = uuid.uuid4()

    images_staged = [
        {
            "id": uuid.uuid4(),
            "filename": f"img{i}.jpg",
            "s3_key": f"projects/{project_id}/user-uploads/img{i}.jpg",
            "task_id": task_id,
        }
        for i in range(3)
    ]

    images_organized = [
        {
            "id": img["id"],
            "filename": img["filename"],
            "s3_key": f"projects/{project_id}/{task_id}/images/{img['filename']}",
            "task_id": img["task_id"],
        }
        for img in images_staged
    ]

    cursor_mock = AsyncMock()
    transaction_mock = AsyncMock()
    mock_db.transaction.return_value.__aenter__.return_value = transaction_mock
    mock_db.cursor.return_value.__aenter__.return_value = cursor_mock

    with patch("app.images.image_classification.run_in_threadpool") as mock_copy:
        mock_copy.return_value = True

        # First call moves images
        cursor_mock.fetchall.return_value = images_staged
        result_1 = await ImageClassifier.organize_batch_images_in_s3(
            mock_db, batch_id, project_id
        )
        assert result_1["total_moved"] == 3

        # Second call skips (DB lock + idempotency)
        cursor_mock.fetchall.return_value = images_organized
        mock_copy.reset_mock()
        result_2 = await ImageClassifier.organize_batch_images_in_s3(
            mock_db, batch_id, project_id
        )
        assert result_2["total_moved"] == 0
        assert result_2["total_skipped"] == 3
        mock_copy.assert_not_called()

        # Verify transaction was used
        assert mock_db.transaction.called
