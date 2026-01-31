from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclasses.dataclass
class GeometryConfig:
    length_x_mm: float = 141.0
    length_y_mm: float = 66.0
    length_z_mm: float = 218.0
    cells_x: int = 1  # number of cells tiled along x
    cells_y: int = 1  # number of cells tiled along y
    cold_plate_face: str = "z_min"  # which face is the cold-plate interface


@dataclasses.dataclass
class PhysicsConfig:
    rho: float = 1800.0  # kg/m^3
    cp: float = 900.0  # J/(kg*K)
    k: float = 1.5  # W/(m*K)
    k_x: Optional[float] = None  # optional anisotropic conductivity along x
    k_y: Optional[float] = None  # optional anisotropic conductivity along y
    k_z: Optional[float] = None  # optional anisotropic conductivity along z
    t_max: float = 120.0  # seconds
    init_temp: float = 298.15  # Kelvin
    # 监督数据与残差的权重，用于快速调节收敛行为
    data_weight: float = 1.0
    bc_weight: float = 1.0
    ic_weight: float = 1.0
    residual_weight: float = 1.0
    q_scale: float = 1.0  
    bc_warmup_epochs: int = 0  
    bc_freeze_epochs: int = 0  
    cold_bc_weight: Optional[float] = None  # optional weight for cold-plate-side convection
    cold_bc_warmup_epochs: int = 0
    cold_bc_freeze_epochs: int = 0
    bc_temp_scale: float = 10.0  # temperature scale (K) for boundary residual normalization
    cold_bc_temp_scale: Optional[float] = None  # optional scale for cold-plate boundary
    interface_temp_scale: float = 5.0  # temperature scale (K) for interface continuity
    interface_flux_scale: Optional[float] = None  # optional heat-flux scale (W/m^2) for interface
    h_env: float = 10.0  # W/(m^2*K) equivalent natural convection for non-cold-plate faces
    interface_weight: float = 1.0  # weight for solid-fluid interface continuity
    interface_freeze_epochs: int = 0  # epochs to keep interface coupling off
    interface_warmup_epochs: int = 0  # ramp interface weight after freeze
    fluid_residual_weight: float = 0.0  # weight for coolant energy equation (set >0 to enable)
    fluid_inlet_weight: float = 0.0  # weight for coolant inlet Dirichlet
    fluid_residual_warmup_epochs: int = 0  # optional warmup for fluid residual
    fluid_inlet_warmup_epochs: int = 0  # optional warmup for inlet condition
    fluid_residual_freeze_epochs: int = 0  # epochs to keep fluid residual off
    fluid_inlet_freeze_epochs: int = 0  # epochs to keep fluid inlet off
    smooth_weight: float = 0.0  # optional spatial gradient penalty to damp high-frequency noise


@dataclasses.dataclass
class FluidConfig:
    thickness_mm: float = 3.0  # coolant layer thickness attached to cold-plate face
    rho: float = 1200.0  # kg/m^3
    cp: float = 1500.0  # J/(kg*K)
    k: float = 0.08  # W/(m*K)
    vel_m_s: float = 1.0  # mean flow speed magnitude for advection
    flow_dir: str = "x"  # advection direction (x/y/z)
    inlet_temp: float = 285.0  # K
    channel_count: int = 1  # number of parallel channels on cold plate
    channel_pitch_mm: float = 0.0  # center-to-center spacing between channels
    channel_width_mm: float = 0.0  # channel width along the distribution axis
    channel_axis: str = "x"  # axis along which channels are distributed (tangential to cold plate)


@dataclasses.dataclass
class PlateConfig:
    thickness_mm: float = 0.0  # cold plate thickness attached to cold-plate face
    rho: float = 2700.0  # kg/m^3
    cp: float = 900.0  # J/(kg*K)
    k: float = 200.0  # W/(m*K)


@dataclasses.dataclass
class PhaseChangeConfig:
    """Phase change material configuration using enthalpy method."""
    enabled: bool = False  # whether to enable phase change modeling
    latent_heat: float = 200000.0  # J/kg, latent heat of phase change
    melting_temp: float = 320.0  # K, melting/phase change temperature
    transition_width: float = 2.0  # K, width of phase transition zone
    pcm_region: str = "plate"  # where PCM is located: "battery" or "plate"


@dataclasses.dataclass
class ConditionConfig:
    name: str
    q_poly: List[float]
    t_env: float
    h_coeff: float
    inlet_temp: float
    # 可选的训练/测试数据路径
    train_interior: Optional[str] = None
    train_boundary: Optional[str] = None
    test: Optional[str] = None
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelConfig:
    hidden_layers: List[int] = dataclasses.field(default_factory=lambda: [128] * 6)
    activation: str = "sine"
    fourier_features: int = 6
    fourier_sigma: float = 3.0
    max_q_terms: int = 8


@dataclasses.dataclass
class VisualizationConfig:
    """Visualization options for evaluation heatmaps."""
    x_slices_mm: List[float] = dataclasses.field(default_factory=lambda: [33.0, 132.0, 264.0, 396.0])
    heatmap_time_points: List[float] = dataclasses.field(default_factory=lambda: [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0])


@dataclasses.dataclass
class OptimConfig:
    lr: float = 1e-3
    epochs: int = 2000
    batch_data: int = 256
    batch_residual: int = 4096
    batch_boundary: int = 2048
    batch_initial: int = 1024
    print_every: int = 100
    plot_every: int = 500
    grad_clip: float = 0.0  
    lr_patience: int = 0  # epochs without improvement before lr decay (0 disables scheduler)
    lr_decay: float = 0.5  # decay factor for scheduler
    lr_min: float = 0.0  # minimum lr for scheduler
    # 每个 epoch 的最大 step 数，用于限制计算量；None 表示每 epoch 仅 1 step（原始行为）
    max_steps_per_epoch: Optional[int] = None
    # 自适应 loss 权重：根据各 loss 分量的 EMA 归一化，避免单一分量主导
    adaptive_loss_weights: bool = False


@dataclasses.dataclass
class Config:
    geometry: GeometryConfig
    physics: PhysicsConfig
    fluid: FluidConfig
    plate: PlateConfig
    phase_change: PhaseChangeConfig
    model: ModelConfig
    optim: OptimConfig
    conditions: List[ConditionConfig]
    visualization: VisualizationConfig = dataclasses.field(default_factory=VisualizationConfig)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_dataclass(cls, data: Dict[str, Any]):
    return cls(**data)


def config_to_dict(cfg: Config) -> dict:
    """Convert Config to a JSON-serializable dictionary."""
    return dataclasses.asdict(cfg)


def load_config(path: str) -> Config:
    raw = _load_yaml(Path(path))
    geometry = _as_dataclass(GeometryConfig, raw.get("geometry", {}))
    physics = _as_dataclass(PhysicsConfig, raw.get("physics", {}))
    fluid = _as_dataclass(FluidConfig, raw.get("fluid", {}))
    plate = _as_dataclass(PlateConfig, raw.get("plate", {}))
    phase_change = _as_dataclass(PhaseChangeConfig, raw.get("phase_change", {}))
    model = _as_dataclass(ModelConfig, raw.get("model", {}))
    optim = _as_dataclass(OptimConfig, raw.get("optim", {}))
    conditions = [
        _as_dataclass(ConditionConfig, c)
        for c in raw.get("conditions", [])
    ]
    if not conditions:
        raise ValueError("At least one condition must be provided in the config.")
    vis_raw = raw.get("visualization", {})
    visualization = _as_dataclass(VisualizationConfig, vis_raw) if vis_raw else VisualizationConfig()
    return Config(
        geometry=geometry,
        physics=physics,
        fluid=fluid,
        plate=plate,
        phase_change=phase_change,
        model=model,
        optim=optim,
        conditions=conditions,
        visualization=visualization,
    )
