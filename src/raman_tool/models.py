"""光谱数据模型."""

from dataclasses import dataclass, field
import numpy as np

from raman_tool.validation import region_mask, validate_spectrum_arrays


RAMAN_SHIFT_UNIT = "cm-1"
PIXEL_UNIT = "px"


def normalize_x_unit(value: str | None) -> str:
    unit = str(RAMAN_SHIFT_UNIT if value is None else value).strip().lower()
    if unit in {"px", "pixel", "pixels"}:
        return PIXEL_UNIT
    if unit in {"", "cm-1", "cm^-1", "cm⁻¹", "1/cm"}:
        return RAMAN_SHIFT_UNIT
    raise ValueError(f"Unsupported x-axis unit: {value!r}; supported units are cm-1 and px")


def x_unit_label(unit: str | None) -> str:
    return "px" if normalize_x_unit(unit) == PIXEL_UNIT else "cm⁻¹"


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

    @property
    def x_unit(self) -> str:
        return normalize_x_unit(self.metadata.get("x_unit"))

    @property
    def x_unit_label(self) -> str:
        return x_unit_label(self.x_unit)

    @property
    def x_label(self) -> str:
        return "像素位置 (px)" if self.x_unit == PIXEL_UNIT else "拉曼位移 (cm⁻¹)"

    @property
    def x_plot_label(self) -> str:
        """Matplotlib-safe x-axis label."""
        return "像素位置 (px)" if self.x_unit == PIXEL_UNIT else r"拉曼位移 (cm$^{-1}$)"

    @property
    def is_raman_shift(self) -> bool:
        return self.x_unit == RAMAN_SHIFT_UNIT

    def __post_init__(self):
        self.raman_shift, self.intensity = validate_spectrum_arrays(self.raman_shift, self.intensity)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        self.metadata = self.metadata.copy()
        self.metadata["x_unit"] = normalize_x_unit(self.metadata.get("x_unit"))

    def crop(self, start: float, end: float) -> "Spectrum":
        """截取指定横坐标范围的光谱.

        Args:
            start: 起始横坐标
            end: 终止横坐标

        Returns:
            截取后的新 Spectrum 对象
        """
        mask = region_mask(self.raman_shift, start, end, name="crop region")
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
