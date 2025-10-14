import csv
import os
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional


router = APIRouter(prefix="/metadata", tags=["metadata"])


class ParticipantMetadata(BaseModel):
    name: str = Field(min_length=1)
    weight: float
    age: int
    sex: str
    media_path: str  # directory or file path relative to data root


@router.post("")
def save_metadata(meta: ParticipantMetadata):
    base_dir = "data_collection_1"
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "participants.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "weight", "age", "sex", "media_path"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(meta.model_dump())
    return {"status": "saved", "csv": csv_path}


