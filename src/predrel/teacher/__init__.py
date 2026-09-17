"""Public Week 1 TabPFN teacher API."""

from .config import TeacherConfig
from .tabpfn_teacher import (
    TabPFNTeacher,
    TeacherEnvironmentError,
    TeacherExtractionError,
)

__all__ = [
    "TabPFNTeacher",
    "TeacherConfig",
    "TeacherEnvironmentError",
    "TeacherExtractionError",
]
