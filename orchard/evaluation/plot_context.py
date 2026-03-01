"""
PlotContext: lightweight DTO that decouples visualization from the full Config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core import Config


@dataclass(frozen=True)
class PlotContext:
    """
    Immutable snapshot of all configuration fields needed by visualization functions.
    """

    arch_name: str
    resolution: int
    fig_dpi: int
    plot_style: str
    cmap_confusion: str
    grid_cols: int
    n_samples: int
    fig_size_predictions: tuple[int, int]
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    use_tta: bool = False
    is_anatomical: bool = True
    is_texture_based: bool = True

    @classmethod
    def from_config(cls, cfg: Config) -> PlotContext:
        """
        Build a PlotContext from the full Config object.
        """
        meta = cfg.dataset.metadata
        return cls(
            arch_name=cfg.architecture.name,
            resolution=cfg.dataset.resolution,
            fig_dpi=cfg.evaluation.fig_dpi,
            plot_style=cfg.evaluation.plot_style,
            cmap_confusion=cfg.evaluation.cmap_confusion,
            grid_cols=cfg.evaluation.grid_cols,
            n_samples=cfg.evaluation.n_samples,
            fig_size_predictions=cfg.evaluation.fig_size_predictions,
            mean=cfg.dataset.mean,
            std=cfg.dataset.std,
            use_tta=cfg.training.use_tta,
            is_anatomical=meta.is_anatomical if meta else True,
            is_texture_based=meta.is_texture_based if meta else True,
        )
