"""光谱数据模型."""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Spectrum:
    """拉曼光谱数据结构.

    Attributes:
        raman_shift: 拉曼位移 (cm-1)，横坐标
        intensity: 光谱强度 (counts)，纵坐标
        filename: 来源文件名
        metadata: 附加元数据
    """

    raman_shift: np.ndarray
    intensity: np.ndarray
    filename: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.raman_shift)

    @property
    def shape(self) -> tuple:
        return self.raman_shift.shape

    def __post_init__(self):
        if isinstance(self.raman_shift, list):
            self.raman_shift = np.array(self.raman_shift, dtype=np.float64)
        if isinstance(self.intensity, list):
            self.intensity = np.array(self.intensity, dtype=np.float64)
        if self.raman_shift.shape != self.intensity.shape:
            raise ValueError(
                f"raman_shift shape {self.raman_shift.shape} "
                f"does not match intensity shape {self.intensity.shape}"
            )

    def crop(self, start: float, end: float) -> "Spectrum":
        """截取指定拉曼位移范围的光谱.

        Args:
            start: 起始拉曼位移 (cm-1)
            end: 终止拉曼位移 (cm-1)

        Returns:
            截取后的新 Spectrum 对象
        """
        mask = (self.raman_shift >= start) & (self.raman_shift <= end)
        return Spectrum(
            raman_shift=self.raman_shift[mask].copy(),
            intensity=self.intensity[mask].copy(),
            filename=self.filename,
            metadata=self.metadata.copy(),
        )

    def normalize(self) -> "Spectrum":
        """对光谱进行归一化 (除以最大值)."""
        max_val = np.max(self.intensity)
        if max_val == 0:
            return Spectrum(
                raman_shift=self.raman_shift.copy(),
                intensity=self.intensity.copy(),
                filename=self.filename,
                metadata=self.metadata.copy(),
            )
        return Spectrum(
            raman_shift=self.raman_shift.copy(),
            intensity=self.intensity / max_val,
            filename=self.filename,
            metadata=self.metadata.copy(),
        )
