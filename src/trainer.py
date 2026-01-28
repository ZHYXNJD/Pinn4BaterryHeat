from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .config import Config, ConditionConfig
from .geometry import BoxGeometry
from .pinn import PINNModel
from .visualizer import Visualizer


class BatteryDataset(Dataset):
    def __init__(self, data_path):
        # 读取数据
        df = pd.read_csv(data_path)
        df[["x", "y", "z"]] /= 1000

        # 输入特征 (x,y,z,t)
        self.X = df[["x", "y", "z", "t"]].values.astype("float32")

        # 输出标签 (temperature)
        self.y = df["temperature"].values.astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


@dataclasses.dataclass
class ConditionRuntime:
    name: str
    q_coeffs: torch.Tensor  # shape (max_q_terms,)
    t_env: float
    h_coeff: float
    inlet_temp: float
    train_interior: str
    train_boundary: str
    test:str

    def vector(self) -> torch.Tensor:
        return torch.cat(
            [
                self.q_coeffs,
                torch.tensor(
                    [self.t_env, self.h_coeff, self.inlet_temp],
                    dtype=torch.float32,
                ),
            ],
            dim=-1,
        )


class PINNTrainer:
    def __init__(self, cfg: Config, device: str = "cpu", output_dir: str = "plots"):
        self.cfg = cfg
        self.device = torch.device(device)
        gx = cfg.geometry.length_x_mm * max(cfg.geometry.cells_x, 1) / 1000.0
        gy = cfg.geometry.length_y_mm * max(cfg.geometry.cells_y, 1) / 1000.0
        gz = cfg.geometry.length_z_mm / 1000.0
        self.geometry = BoxGeometry(
            length_x=gx,
            length_y=gy,
            length_z=gz,
        )
        self.cold_plate_face_idx = self._parse_face(cfg.geometry.cold_plate_face)
        self.model = PINNModel(
            cond_dim=cfg.model.max_q_terms + 3,
            hidden_layers=cfg.model.hidden_layers,
            activation=cfg.model.activation,
            fourier_features=cfg.model.fourier_features,
            fourier_sigma=cfg.model.fourier_sigma,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.optim.lr)
        self.mse = nn.MSELoss()
        self.conditions = self._prepare_conditions(cfg.conditions, cfg.model.max_q_terms)
        self.q_scale = cfg.physics.q_scale if cfg.physics.q_scale > 0 else 1.0
        self.kx = cfg.physics.k_x if cfg.physics.k_x is not None else cfg.physics.k
        self.ky = cfg.physics.k_y if cfg.physics.k_y is not None else cfg.physics.k
        self.kz = cfg.physics.k_z if cfg.physics.k_z is not None else cfg.physics.k
        self.cold_bc_weight = (
            cfg.physics.cold_bc_weight if cfg.physics.cold_bc_weight is not None else cfg.physics.bc_weight
        )
        self.bc_temp_scale = max(self._cfg_float(cfg.physics.bc_temp_scale, "physics.bc_temp_scale"), 1e-6)
        self.cold_bc_temp_scale = (
            max(self._cfg_float(cfg.physics.cold_bc_temp_scale, "physics.cold_bc_temp_scale"), 1e-6)
            if cfg.physics.cold_bc_temp_scale is not None
            else self.bc_temp_scale
        )
        self.interface_temp_scale = max(
            self._cfg_float(cfg.physics.interface_temp_scale, "physics.interface_temp_scale"), 1e-6)
        self.visualizer = Visualizer(save_dir=output_dir)
        self.plate_thickness = cfg.plate.thickness_mm / 1000.0
        self.plate_rho = cfg.plate.rho
        self.plate_cp = cfg.plate.cp
        self.plate_k = cfg.plate.k
        self.fluid_thickness = cfg.fluid.thickness_mm / 1000.0
        self.use_fluid = self.fluid_thickness > 0 and (
                cfg.physics.fluid_residual_weight > 0 or cfg.physics.fluid_inlet_weight > 0
        )
        self.fluid_offset = self.plate_thickness if self.plate_thickness > 0 else 0.0
        channel_axis = cfg.fluid.channel_axis.lower()
        if channel_axis.startswith("x"):
            self.channel_axis = 0
        elif channel_axis.startswith("y"):
            self.channel_axis = 1
        else:
            self.channel_axis = 2
        self.channel_width = cfg.fluid.channel_width_mm / 1000.0 if cfg.fluid.channel_width_mm > 0 else 0.0
        self.channel_pitch = cfg.fluid.channel_pitch_mm / 1000.0 if cfg.fluid.channel_pitch_mm > 0 else 0.0
        self.channel_count = max(cfg.fluid.channel_count, 0)
        self.channel_centers = self.geometry.channel_centers(
            self.channel_axis, self.channel_count, self.channel_pitch, device=self.device
        )
        flow_dir = cfg.fluid.flow_dir.lower()
        if flow_dir.startswith("x"):
            self.flow_dir = 0
            self.flow_vec = torch.tensor([cfg.fluid.vel_m_s, 0.0, 0.0], device=self.device)
        elif flow_dir.startswith("y"):
            self.flow_dir = 1
            self.flow_vec = torch.tensor([0.0, cfg.fluid.vel_m_s, 0.0], device=self.device)
        else:
            self.flow_dir = 2
            self.flow_vec = torch.tensor([0.0, 0.0, cfg.fluid.vel_m_s], device=self.device)
        # AMP mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')
        self.scheduler = None
        if self.cfg.optim.lr_patience > 0:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg.optim.lr_decay,
                patience=self.cfg.optim.lr_patience,
                min_lr=self.cfg.optim.lr_min,
            )
        if cfg.physics.interface_flux_scale is not None:
            self.interface_flux_scale = max(
                self._cfg_float(cfg.physics.interface_flux_scale, "physics.interface_flux_scale"), 1e-6
            )
        else:
            self.interface_flux_scale = self._default_interface_flux_scale()

    def _parse_face(self, face: str) -> int:
        mapping = {
            "x_min": 0, "xmin": 0, "x-": 0,
            "x_max": 1, "xmax": 1, "x+": 1,
            "y_min": 2, "ymin": 2, "y-": 2,
            "y_max": 3, "ymax": 3, "y+": 3,
            "z_min": 4, "zmin": 4, "z-": 4,
            "z_max": 5, "zmax": 5, "z+": 5,
        }
        key = face.lower().replace("-", "_").replace("+", "+").strip()
        if key not in mapping:
            raise ValueError(f"Unsupported cold_plate_face '{face}', expected one of {list(mapping.keys())}")
        return mapping[key]

    def _k_for_face(self, face_idx: int) -> float:
        if face_idx in (0, 1):
            return self.kx
        if face_idx in (2, 3):
            return self.ky
        return self.kz

    def _k_for_faces(self, faces: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        k_map = torch.tensor(
            [self.kx, self.kx, self.ky, self.ky, self.kz, self.kz],
            device=faces.device,
            dtype=dtype,
        )
        return k_map[faces].unsqueeze(-1)

    def _get_face_and_normal(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据坐标判断点在哪个面上，并返回面索引和法向量。
        
        Args:
            xyz: 坐标点 (n, 3)
            
        Returns:
            faces: 面索引 (0-5)，0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max
            normals: 法向量 (n, 3)
        """
        half = torch.tensor(
            [self.geometry.length_x / 2, self.geometry.length_y / 2, self.geometry.length_z / 2],
            device=xyz.device,
            dtype=xyz.dtype,
        )
        n = xyz.shape[0]
        faces = torch.zeros(n, dtype=torch.long, device=xyz.device)
        normals = torch.zeros((n, 3), device=xyz.device, dtype=xyz.dtype)

        # 计算每个点到各面的距离（绝对值）
        dist_x_min = torch.abs(xyz[:, 0] + half[0])
        dist_x_max = torch.abs(xyz[:, 0] - half[0])
        dist_y_min = torch.abs(xyz[:, 1] + half[1])
        dist_y_max = torch.abs(xyz[:, 1] - half[1])
        dist_z_min = torch.abs(xyz[:, 2] + half[2])
        dist_z_max = torch.abs(xyz[:, 2] - half[2])

        # 将所有距离堆叠，找到最近的面
        dists = torch.stack([
            dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max
        ], dim=1)  # (n, 6)

        # 找到最小距离对应的面索引
        faces = torch.argmin(dists, dim=1)

        # 根据面索引设置法向量
        mask_0 = faces == 0  # x_min
        normals[mask_0, 0] = -1.0

        mask_1 = faces == 1  # x_max
        normals[mask_1, 0] = 1.0

        mask_2 = faces == 2  # y_min
        normals[mask_2, 1] = -1.0

        mask_3 = faces == 3  # y_max
        normals[mask_3, 1] = 1.0

        mask_4 = faces == 4  # z_min
        normals[mask_4, 2] = -1.0

        mask_5 = faces == 5  # z_max
        normals[mask_5, 2] = 1.0

        return faces, normals

    def _cfg_float(self, value, name: str) -> float:
        try:
            return float(value)
        except Exception as exc:
            raise ValueError(f"Config field '{name}' must be numeric, got {value!r}") from exc

    def _default_interface_flux_scale(self) -> float:
        k_scale = max(self.plate_k, self.kx, self.ky, self.kz)
        if self.plate_thickness > 0:
            length_scale = self.plate_thickness
        else:
            length_scale = 0.1 * min(
                self.geometry.length_x, self.geometry.length_y, self.geometry.length_z
            )
        length_scale = max(length_scale, 1e-4)
        return max(k_scale * self.interface_temp_scale / length_scale, 1e-6)

    def _liquid_fraction(self, T: torch.Tensor) -> torch.Tensor:
        """Compute liquid fraction f(T) using sigmoid function for phase change.
        
        f(T) = 1 / (1 + exp(-(T - T_m) / delta))
        
        Returns 0 for T << T_m (solid), 1 for T >> T_m (liquid).
        """
        T_m = self.cfg.phase_change.melting_temp
        delta = max(self.cfg.phase_change.transition_width, 1e-6)
        return torch.sigmoid((T - T_m) / delta)

    def _effective_cp(self, T: torch.Tensor, base_cp: float) -> torch.Tensor:
        """Compute effective specific heat capacity including latent heat.
        
        c_eff = c_p + L * df/dT
        where df/dT = f * (1 - f) / delta
        
        This implements the enthalpy method for phase change.
        """
        if not self.cfg.phase_change.enabled:
            return torch.full_like(T, base_cp)

        L = self.cfg.phase_change.latent_heat
        delta = max(self.cfg.phase_change.transition_width, 1e-6)
        f = self._liquid_fraction(T)
        # df/dT for sigmoid: f * (1 - f) / delta
        df_dT = f * (1.0 - f) / delta
        return base_cp + L * df_dT

    def _eval_T_and_dTdn(
            self,
            coords: torch.Tensor,
            normals: torch.Tensor,
            cond_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = coords[:, :3]
        t = coords[:, 3:4]
        norm_coords = self._normalize(xyz, t)
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp
        grads = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )[0]
        dTdn = (
                grads[:, 0:1] * normals[:, 0:1]
                + grads[:, 1:2] * normals[:, 1:2]
                + grads[:, 2:3] * normals[:, 2:3]
        )
        return T, dTdn

    def _prepare_conditions(
            self, conds: List[ConditionConfig], max_terms: int
    ) -> List[ConditionRuntime]:
        def _float_list(name: str, values: List) -> List[float]:
            out = []
            for i, v in enumerate(values):
                try:
                    out.append(float(v))
                except Exception as exc:
                    raise ValueError(
                        f"Condition '{name}' q_poly[{i}] must be numeric, got {v!r}"
                    ) from exc
            return out

        def _to_float(name: str, value) -> float:
            try:
                return float(value)
            except Exception as exc:
                raise ValueError(f"Condition '{name}' field must be numeric, got {value!r}") from exc

        result = []
        for c in conds:
            q_values = _float_list(c.name, c.q_poly)
            coeffs = torch.zeros(max_terms, dtype=torch.float32)
            n = min(len(q_values), max_terms)
            coeffs[:n] = torch.tensor(q_values[:n], dtype=torch.float32)
            result.append(
                ConditionRuntime(
                    name=c.name,
                    q_coeffs=coeffs,
                    t_env=_to_float(c.name, c.t_env),
                    h_coeff=_to_float(c.name, c.h_coeff),
                    inlet_temp=_to_float(c.name, c.inlet_temp),
                    train_interior = c.train_interior,
                    train_boundary = c.train_boundary,
                    test = c.test
                )
            )
        return result

    def _normalize(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        half = torch.tensor(
            [
                self.geometry.length_x / 2,
                self.geometry.length_y / 2,
                self.geometry.length_z / 2,
            ],
            device=xyz.device,
            dtype=xyz.dtype,
        )
        # extend normalization span along the cold-plate normal to include plate/fluid layers
        extra = 0.0
        if self.plate_thickness > 0:
            extra += self.plate_thickness
        if self.use_fluid and self.fluid_thickness > 0:
            extra += self.fluid_thickness
        if extra > 0:
            if self.cold_plate_face_idx in (0, 1):  # x faces
                half[0] = half[0] + extra
            elif self.cold_plate_face_idx in (2, 3):  # y faces
                half[1] = half[1] + extra
            else:  # z faces
                half[2] = half[2] + extra
        xyz_n = xyz / half
        t_n = (t / self.cfg.physics.t_max) * 2.0 - 1.0
        return torch.cat([xyz_n, t_n], dim=1)

    def _eval_q(self, t: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        t_flat = t.squeeze(-1)
        terms = []
        current = torch.ones_like(t_flat)
        for _ in range(coeffs.numel()):
            terms.append(current)
            current = current * t_flat
        powers = torch.stack(terms, dim=1).to(t.device)
        return (powers * coeffs.to(t.device)).sum(dim=1)

    def _data_loss(self, cond: ConditionRuntime, x_batch, y_batch) -> torch.Tensor:
        xyz = x_batch[:, :3].to(self.device)
        t = x_batch[:, 3:4].to(self.device)
        norm_coords = self._normalize(xyz, t)
        temperature_target = y_batch.to(self.device).view(-1, 1)
        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(temperature_target.shape[0], 1)
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + cond.inlet_temp
        return self.mse(T, temperature_target)

    def _residual_loss(self, cond: ConditionRuntime) -> torch.Tensor:
        n = self.cfg.optim.batch_residual
        coords = torch.cat(
            [
                self.geometry.sample_interior(n, self.device),
                torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max,
            ],
            dim=1,
        )
        coords = coords.clone().detach().requires_grad_(True)
        xyz = coords[:, :3].clone()
        t = coords[:, 3:4].clone()

        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        norm_coords = self._normalize(xyz, t)
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp

        grads = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )[0]
        dTdx, dTdy, dTdz, dTdt = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3], grads[:, 3:4]
        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.physics.smooth_weight > 0:
            smooth_loss = (dTdx ** 2 + dTdy ** 2 + dTdz ** 2).mean()

        ones_dTdx = torch.ones_like(dTdx)
        ones_dTdy = torch.ones_like(dTdy)
        ones_dTdz = torch.ones_like(dTdz)
        d2Tdx2 = torch.autograd.grad(
            outputs=dTdx,
            inputs=coords,
            grad_outputs=ones_dTdx,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        d2Tdy2 = torch.autograd.grad(
            outputs=dTdy,
            inputs=coords,
            grad_outputs=ones_dTdy,
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        d2Tdz2 = torch.autograd.grad(
            outputs=dTdz,
            inputs=coords,
            grad_outputs=ones_dTdz,
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]
        laplace = self.kx * d2Tdx2 + self.ky * d2Tdy2 + self.kz * d2Tdz2

        q_val = self._eval_q(t, cond.q_coeffs).unsqueeze(-1)
        # Phase change support: use effective cp if enabled for battery region
        if self.cfg.phase_change.enabled and self.cfg.phase_change.pcm_region == "battery":
            cp_eff = self._effective_cp(T, self.cfg.physics.cp)
            rho_cp = self.cfg.physics.rho * cp_eff
        else:
            rho_cp = self.cfg.physics.rho * self.cfg.physics.cp
        res = dTdt - (laplace / rho_cp) - (self.q_scale * q_val / rho_cp)
        base = (res ** 2).mean()
        return base + self.cfg.physics.smooth_weight * smooth_loss

    def _plate_residual(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.plate_thickness <= 0:
            return torch.tensor(0.0, device=self.device)
        n = self.cfg.optim.batch_residual
        xyz = self.geometry.sample_layer(
            n,
            self.device,
            self.cold_plate_face_idx,
            self.plate_thickness,
            offset=0.0,
        )
        t = torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max
        coords = torch.cat([xyz, t], dim=1).clone().detach().requires_grad_(True)

        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        norm_coords = self._normalize(coords[:, :3], coords[:, 3:4])
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp

        grads = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )[0]
        dTdx, dTdy, dTdz, dTdt = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3], grads[:, 3:4]
        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.physics.smooth_weight > 0:
            smooth_loss = (dTdx ** 2 + dTdy ** 2 + dTdz ** 2).mean()

        ones_dTdx = torch.ones_like(dTdx)
        ones_dTdy = torch.ones_like(dTdy)
        ones_dTdz = torch.ones_like(dTdz)
        d2Tdx2 = torch.autograd.grad(
            outputs=dTdx,
            inputs=coords,
            grad_outputs=ones_dTdx,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        d2Tdy2 = torch.autograd.grad(
            outputs=dTdy,
            inputs=coords,
            grad_outputs=ones_dTdy,
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]
        d2Tdz2 = torch.autograd.grad(
            outputs=dTdz,
            inputs=coords,
            grad_outputs=ones_dTdz,
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]
        laplace = d2Tdx2 + d2Tdy2 + d2Tdz2

        # Phase change support: use effective cp if enabled for plate region
        if self.cfg.phase_change.enabled and self.cfg.phase_change.pcm_region == "plate":
            cp_eff = self._effective_cp(T, self.plate_cp)
            rho_cp = self.plate_rho * cp_eff
        else:
            rho_cp = self.plate_rho * self.plate_cp
        res = dTdt - (self.plate_k / rho_cp) * laplace
        base = (res ** 2).mean()
        return base + self.cfg.physics.smooth_weight * smooth_loss

    # 由于采样点中包含边界点 这里需要修改采样逻辑
    # 不再随机采样 而是直接加载boundary data
    # 观测点的值作为label 计算loss
    def _boundary_loss(self, cond: ConditionRuntime, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """使用边界数据计算边界损失。
        
        Args:
            cond: 条件参数
            x_batch: 边界点坐标 (batch_size, 4) - (x, y, z, t)
            y_batch: 边界点观测温度 (batch_size,)
        """
        xyz_raw = x_batch[:, :3].to(self.device)
        t_raw = x_batch[:, 3:4].to(self.device)
        temperature_target = y_batch.to(self.device)

        # 判断每个点在哪个面上，并获取法向量
        faces, normals = self._get_face_and_normal(xyz_raw)

        # 构建坐标并计算梯度
        coords = torch.cat([xyz_raw, t_raw], dim=1).clone().detach().requires_grad_(True)
        # 3. 【关键修改】从 coords 中切分出 xyz 和 t
        # 这样 T_net 才会依赖于 coords，autograd 才能计算 d(T)/d(coords)
        xyz_grad = coords[:, :3]
        t_grad = coords[:, 3:4]
        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(xyz_grad.shape[0], 1)
        # 使用带有梯度的变量 xyz_grad, t_grad 进行归一化和前向传播
        norm_coords = self._normalize(xyz_grad, t_grad)
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp

        grads = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=torch.ones_like(T),
            # 只需要一阶导数供后续 loss 使用，不再对其求导
            create_graph=True,
            retain_graph=True,
        )[0]
        dTdx, dTdy, dTdz = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
        dTdn = (dTdx * normals[:, 0:1] + dTdy * normals[:, 1:2] + dTdz * normals[:, 2:3])

        # 使用观测温度作为参考温度，而不是固定的cond.t_env
        # Equivalent convection: k_n * dT/dn + h * (T - T_obs) = 0
        h_env = torch.full_like(dTdn, self.cfg.physics.h_env)
        # 修复形状：确保 target 是 (N, 1) 而不是 (N,)
        T_ref = temperature_target.view(-1, 1)
        cold_mask = (faces == self.cold_plate_face_idx).unsqueeze(-1)

        # Only apply convection on non-cold faces; cold face is handled by interface coupling loss.
        k_n = self._k_for_faces(faces, dTdn.dtype)
        bc_res = k_n * dTdn + h_env * (T - T_ref)
        bc_scale = torch.clamp(h_env, min=1.0)
        bc_res = bc_res / bc_scale / self.bc_temp_scale
        bc_res = bc_res[~cold_mask.squeeze(-1)]
        if bc_res.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        return (bc_res ** 2).mean()

    def _plate_boundary_loss(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.use_fluid:
            return torch.tensor(0.0, device=self.device)
        n = self.cfg.optim.batch_boundary
        xyz, normals = self.geometry.sample_face(n, self.device, self.cold_plate_face_idx)
        offset = self.plate_thickness
        xyz = xyz + normals * offset
        t = torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max
        coords = torch.cat([xyz, t], dim=1).clone().detach().requires_grad_(True)

        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        norm_coords = self._normalize(coords[:, :3], coords[:, 3:4])
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp

        grads = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )[0]
        dTdx, dTdy, dTdz = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
        dTdn = (dTdx * normals[:, 0:1] + dTdy * normals[:, 1:2] + dTdz * normals[:, 2:3])

        h_coeff = torch.full_like(dTdn, max(cond.h_coeff, 0.0))
        T_ref = torch.full_like(T, cond.inlet_temp)
        k_value = self.plate_k if self.plate_thickness > 0 else self._k_for_face(self.cold_plate_face_idx)
        k_n = torch.full_like(dTdn, k_value)
        bc_res = k_n * dTdn + h_coeff * (T - T_ref)
        bc_scale = torch.clamp(h_coeff, min=1.0)
        bc_res = bc_res / bc_scale / self.cold_bc_temp_scale
        return (bc_res ** 2).mean()

    def _fluid_residual(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.cfg.physics.fluid_residual_weight <= 0 or self.fluid_thickness <= 0:
            return torch.tensor(0.0, device=self.device)
        n = self.cfg.optim.batch_residual
        xyz_raw = self.geometry.sample_channel_layer(
            n,
            self.device,
            self.cold_plate_face_idx,
            self.fluid_thickness,
            self.channel_axis,
            self.channel_centers,
            self.channel_width,
            offset=self.fluid_offset,
        )
        t_raw = torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max
        coords = torch.cat([xyz_raw, t_raw], dim=1).clone().detach().requires_grad_(True)

        xyz = coords[:, :3]
        t = coords[:, 3:4]

        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        norm_coords = self._normalize(xyz, t)
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp

        grads = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )[0]
        dTdx, dTdy, dTdz, dTdt = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3], grads[:, 3:4]

        # diffusion term
        ones_dTdx = torch.ones_like(dTdx)
        ones_dTdy = torch.ones_like(dTdy)
        ones_dTdz = torch.ones_like(dTdz)
        d2Tdx2 = torch.autograd.grad(
            dTdx, coords, grad_outputs=ones_dTdx, create_graph=True, retain_graph=True
        )[0][:, 0:1]

        d2Tdy2 = torch.autograd.grad(
            dTdy, coords, grad_outputs=ones_dTdy, create_graph=True, retain_graph=True
        )[0][:, 1:2]

        d2Tdz2 = torch.autograd.grad(
            dTdz, coords, grad_outputs=ones_dTdz, create_graph=True, retain_graph=True
        )[0][:, 2:3]
        laplace = d2Tdx2 + d2Tdy2 + d2Tdz2

        v = self.flow_vec.to(coords)
        adv = v[0] * dTdx + v[1] * dTdy + v[2] * dTdz
        if self.cfg.phase_change.enabled and self.cfg.phase_change.pcm_region == "battery":
            cp_eff = self._effective_cp(T, self.cfg.physics.cp)
        else:
            cp_eff = self.cfg.fluid.cp
        rho_cp = self.cfg.fluid.rho * cp_eff
        res = dTdt + adv - (self.cfg.fluid.k / rho_cp) * laplace
        return (res ** 2).mean()

    def _interface_loss(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.cfg.physics.interface_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        if self.plate_thickness > 0:
            loss = self._battery_plate_interface(cond)
            if self.use_fluid and self.fluid_thickness > 0:
                loss = loss + self._plate_fluid_interface(cond)
            return loss
        if not self.use_fluid:
            return torch.tensor(0.0, device=self.device)
        return self._battery_fluid_interface(cond)

    def _battery_plate_interface(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.plate_thickness <= 0:
            return torch.tensor(0.0, device=self.device)
        n = self.cfg.optim.batch_boundary
        xyz, normals = self.geometry.sample_face(n, self.device, self.cold_plate_face_idx)
        t = torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max
        eps = max(min(self.plate_thickness * 0.1, 1e-4), 1e-6)

        coords_batt = torch.cat([xyz, t], dim=1).clone().detach().requires_grad_(True)
        coords_plate = torch.cat([xyz + normals * eps, t], dim=1).clone().detach().requires_grad_(True)

        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        T_batt, dTdn_batt = self._eval_T_and_dTdn(coords_batt, normals, cond_vec)
        T_plate, dTdn_plate = self._eval_T_and_dTdn(coords_plate, normals, cond_vec)

        k_batt = self._k_for_face(self.cold_plate_face_idx)
        cont_res = (T_batt - T_plate) / self.interface_temp_scale
        flux_res = (k_batt * dTdn_batt - self.plate_k * dTdn_plate) / self.interface_flux_scale
        return (cont_res ** 2).mean() + (flux_res ** 2).mean()

    def _plate_fluid_interface(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.plate_thickness <= 0 or self.fluid_thickness <= 0:
            return torch.tensor(0.0, device=self.device)
        n = self.cfg.optim.batch_boundary
        xyz, normals = self.geometry.sample_face(n, self.device, self.cold_plate_face_idx)
        t = torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max
        eps = max(min(min(self.plate_thickness, self.fluid_thickness) * 0.1, 1e-4), 1e-6)
        base = xyz + normals * self.plate_thickness

        coords_plate = torch.cat([base - normals * eps, t], dim=1).clone().detach().requires_grad_(True)
        coords_fluid = torch.cat([base + normals * eps, t], dim=1).clone().detach().requires_grad_(True)

        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        T_plate, dTdn_plate = self._eval_T_and_dTdn(coords_plate, normals, cond_vec)
        T_fluid, dTdn_fluid = self._eval_T_and_dTdn(coords_fluid, normals, cond_vec)

        cont_res = (T_plate - T_fluid) / self.interface_temp_scale
        flux_res = (self.plate_k * dTdn_plate - self.cfg.fluid.k * dTdn_fluid) / self.interface_flux_scale
        return (cont_res ** 2).mean() + (flux_res ** 2).mean()

    def _battery_fluid_interface(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.fluid_thickness <= 0:
            return torch.tensor(0.0, device=self.device)
        n = self.cfg.optim.batch_boundary
        xyz, normals = self.geometry.sample_face(n, self.device, self.cold_plate_face_idx)
        t = torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max
        eps = max(min(self.fluid_thickness * 0.1, 1e-4), 1e-6)

        coords_batt = torch.cat([xyz, t], dim=1).clone().detach().requires_grad_(True)
        coords_fluid = torch.cat([xyz + normals * eps, t], dim=1).clone().detach().requires_grad_(True)

        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        T_batt, dTdn_batt = self._eval_T_and_dTdn(coords_batt, normals, cond_vec)
        T_fluid, dTdn_fluid = self._eval_T_and_dTdn(coords_fluid, normals, cond_vec)

        k_batt = self._k_for_face(self.cold_plate_face_idx)
        cont_res = (T_batt - T_fluid) / self.interface_temp_scale
        flux_res = (k_batt * dTdn_batt - self.cfg.fluid.k * dTdn_fluid) / self.interface_flux_scale
        return (cont_res ** 2).mean() + (flux_res ** 2).mean()

    def _fluid_inlet_loss(self, cond: ConditionRuntime) -> torch.Tensor:
        if self.cfg.physics.fluid_inlet_weight <= 0 or self.fluid_thickness <= 0:
            return torch.tensor(0.0, device=self.device)
        n = self.cfg.optim.batch_boundary
        xyz = self.geometry.sample_channel_inlet(
            n,
            self.device,
            self.cold_plate_face_idx,
            self.fluid_thickness,
            self.flow_dir,
            self.channel_axis,
            self.channel_centers,
            self.channel_width,
            offset=self.fluid_offset,
        )
        t = torch.rand((n, 1), device=self.device) * self.cfg.physics.t_max
        coords = torch.cat([xyz, t], dim=1)
        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        norm_coords = self._normalize(xyz, t)
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp
        target = torch.full_like(T, cond.inlet_temp)
        return self.mse(T, target)

    def _initial_loss(self, cond: ConditionRuntime) -> torch.Tensor:
        n = self.cfg.optim.batch_initial
        coords = torch.cat(
            [self.geometry.sample_interior(n, self.device), torch.zeros((n, 1), device=self.device)],
            dim=1,
        )
        coords = coords.clone().detach().requires_grad_(True)
        xyz = coords[:, :3].clone()
        t = coords[:, 3:4].clone()
        cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(n, 1)
        norm_coords = self._normalize(xyz, t)
        T_net = self.model(norm_coords, cond_vec)
        T = T_net + self.cfg.physics.init_temp
        target = torch.full_like(T, self.cfg.physics.init_temp)
        return self.mse(T, target)

    def train(self):
        """Training loop based on the original per-epoch/per-condition structure.

        每个 epoch、每个工况只计算一次 PDE 残差和边界损失；
        如有配置的监督数据，则在该工况上最多取一个 batch 做 data loss，
        避免 steps_per_epoch 的额外嵌套导致显存爆炸。
        """

        def scheduled_weight(base: float, epoch: int, warmup: int = 0, freeze: int = 0) -> float:
            if base <= 0:
                return 0.0
            if freeze > 0 and epoch <= freeze:
                return 0.0
            if warmup > 0:
                start = freeze if freeze > 0 else 0
                progress = (epoch - start) / float(warmup)
                return base * min(1.0, max(0.0, progress))
            return base

        num_conditions = len(self.conditions)
        if num_conditions == 0:
            raise ValueError("No training conditions provided.")

        # 为每个工况构建一次性的 Dataset/DataLoader（如果配置了路径）
        batch_size = self.cfg.optim.batch_data
        interior_loaders: list[DataLoader | None] = []
        boundary_loaders: list[DataLoader | None] = []

        print("Initializing Datasets from config (per-condition, no step loop)...")
        for i, cond in enumerate(self.conditions):
            has_interior = getattr(cond, "train_interior", None)
            has_boundary = getattr(cond, "train_boundary", None)

            ld_int = None
            ld_bnd = None
            if has_interior:
                print(f"  - Condition {i} ({cond.name}): train_interior = {has_interior}")
                ds_int = BatteryDataset(has_interior)
                ld_int = DataLoader(ds_int, batch_size=batch_size, shuffle=True)
            if has_boundary:
                print(f"  - Condition {i} ({cond.name}): train_boundary = {has_boundary}")
                ds_bnd = BatteryDataset(has_boundary)
                ld_bnd = DataLoader(ds_bnd, batch_size=batch_size, shuffle=True)

            interior_loaders.append(ld_int)
            boundary_loaders.append(ld_bnd)

        for epoch in range(1, self.cfg.optim.epochs + 1):
            epoch_loss = 0.0
            total_data = 0.0
            total_res = 0.0
            total_fluid_res = 0.0
            total_bc = 0.0
            total_cold_bc = 0.0
            total_ic = 0.0
            total_fluid_inlet = 0.0
            total_interface = 0.0

            self.optimizer.zero_grad()

            bc_w = scheduled_weight(
                self.cfg.physics.bc_weight,
                epoch,
                warmup=self.cfg.physics.bc_warmup_epochs,
                freeze=0,
            )
            cold_bc_w = scheduled_weight(
                self.cold_bc_weight,
                epoch,
                warmup=self.cfg.physics.cold_bc_warmup_epochs,
                freeze=self.cfg.physics.cold_bc_freeze_epochs,
            )
            fluid_res_w = scheduled_weight(
                self.cfg.physics.fluid_residual_weight,
                epoch,
                warmup=self.cfg.physics.fluid_residual_warmup_epochs,
                freeze=self.cfg.physics.fluid_residual_freeze_epochs,
            )
            fluid_inlet_w = scheduled_weight(
                self.cfg.physics.fluid_inlet_weight,
                epoch,
                warmup=self.cfg.physics.fluid_inlet_warmup_epochs,
                freeze=self.cfg.physics.fluid_inlet_freeze_epochs,
            )
            interface_w = scheduled_weight(
                self.cfg.physics.interface_weight,
                epoch,
                warmup=self.cfg.physics.interface_warmup_epochs,
                freeze=self.cfg.physics.interface_freeze_epochs,
            )

            for idx, cond in enumerate(self.conditions):
                # 1) 可选的有监督 data loss（单个 batch，避免显存暴涨）
                loss_data = torch.tensor(0.0, device=self.device)
                loss_bc_data = torch.tensor(0.0, device=self.device)

                ld_int = interior_loaders[idx]
                if ld_int is not None:
                    try:
                        x_int, y_int = next(iter(ld_int))
                        loss_data = self._data_loss(cond, x_int, y_int)
                    except StopIteration:
                        pass

                ld_bnd = boundary_loaders[idx]
                if ld_bnd is not None and epoch > self.cfg.physics.bc_freeze_epochs:
                    try:
                        x_bnd, y_bnd = next(iter(ld_bnd))
                        loss_bc_data = self._boundary_loss(cond, x_bnd, y_bnd)
                    except StopIteration:
                        pass

                # 2) PDE 残差与物理损失（原始结构）
                res_loss = self._residual_loss(cond)
                plate_res_loss = self._plate_residual(cond)
                fluid_res_loss = self._fluid_residual(cond)
                bc_loss = torch.tensor(0.0, device=self.device)
                plate_bc_loss = torch.tensor(0.0, device=self.device)
                if ld_bnd is None and epoch > self.cfg.physics.bc_freeze_epochs:
                    # 仅当没有监督边界数据时，才使用物理边界损失
                    bc_loss = torch.tensor(0.0, device=self.device)  # 保留接口以防后续需要恢复
                if epoch > self.cfg.physics.cold_bc_freeze_epochs:
                    plate_bc_loss = self._plate_boundary_loss(cond)
                ic_loss = self._initial_loss(cond)
                fluid_inlet_loss = self._fluid_inlet_loss(cond)
                interface_loss = self._interface_loss(cond)

                data_weight = getattr(self.cfg.physics, "data_weight", 1.0)

                loss = (
                    data_weight * loss_data
                    + bc_w * loss_bc_data
                    + self.cfg.physics.residual_weight * (res_loss + plate_res_loss)
                    + fluid_res_w * fluid_res_loss
                    + cold_bc_w * plate_bc_loss
                    + self.cfg.physics.ic_weight * ic_loss
                    + fluid_inlet_w * fluid_inlet_loss
                    + interface_w * interface_loss
                )

                loss.backward()
                epoch_loss += loss.item()
                total_data += loss_data.item()
                total_res += res_loss.item() + plate_res_loss.item()
                total_fluid_res += fluid_res_loss.item()
                total_bc += loss_bc_data.item() + bc_loss.item()
                total_cold_bc += plate_bc_loss.item()
                total_ic += ic_loss.item()
                total_fluid_inlet += fluid_inlet_loss.item()
                total_interface += interface_loss.item()

            if self.cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.optimizer.step()

            avg_total = epoch_loss / num_conditions
            avg_data = total_data / num_conditions
            avg_res = total_res / num_conditions
            avg_fluid_res = total_fluid_res / num_conditions
            avg_bc = total_bc / num_conditions
            avg_cold_bc = total_cold_bc / num_conditions
            avg_ic = total_ic / num_conditions
            avg_fluid_inlet = total_fluid_inlet / num_conditions
            avg_interface = total_interface / num_conditions

            if epoch % self.cfg.optim.print_every == 0:
                print(
                    f"[{epoch:06d}] total_loss={avg_total:.4e} "
                    f"(data={avg_data:.3e}, "
                    f"res={avg_res:.3e}, fluid_res={avg_fluid_res:.3e} "
                    f"bc={avg_bc:.3e}, cold_bc={avg_cold_bc:.3e}"
                    f"ic={avg_ic:.3e}"
                    f"fluid_inlet={avg_fluid_inlet:.3e}, "
                    f"interface={avg_interface:.3e}"
                )

            self.visualizer.log_loss(
                avg_total,
                avg_data,
                avg_res,
                avg_bc + avg_cold_bc,
                avg_ic,
                fluid_res=avg_fluid_res,
                fluid_inlet=avg_fluid_inlet,
                interface=avg_interface,
            )

            if self.scheduler is not None:
                self.scheduler.step(avg_total)

            if epoch % self.cfg.optim.plot_every == 0:
                self.visualizer.plot_loss(epoch)
                self.visualizer.plot_temperature(
                    self.model,
                    self.geometry,
                    self.device,
                    epoch,
                    self.conditions[0],
                    self.cfg.physics.t_max,
                    self.cfg.physics.init_temp,
                    normalize_fn=self._normalize,
                )

    def save(self, path: str):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": dataclasses.asdict(self.cfg),
            },
            path,
        )

    def evaluate(self):
        print("Starting final evaluation...")
        self.visualizer.plot_loss(self.cfg.optim.epochs)

        for i, cond in enumerate(self.conditions):
            print(f"Plotting final temperature for condition: {cond.name}")
            self.visualizer.plot_temperature(
                self.model,
                self.geometry,
                self.device,
                self.cfg.optim.epochs,
                cond,
                self.cfg.physics.t_max,
                self.cfg.physics.init_temp,
                name_suffix="final",
                normalize_fn=self._normalize,
            )

    def evaluate_on_test(self):
        """Evaluate model on independent test CSVs configured per condition."""
        print("Evaluating on test datasets...")
        results = []
        save_dir = getattr(self.visualizer, "save_dir", "plots")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, cond in enumerate(self.conditions):
            test_path = getattr(cond, "test", None)
            if not test_path:
                print(f"  - Condition {cond.name}: no test dataset configured, skipping.")
                continue

            try:
                ds_test = BatteryDataset(test_path)
            except FileNotFoundError:
                print(f"  - Condition {cond.name}: test file '{test_path}' not found, skipping.")
                continue

            loader = DataLoader(ds_test, batch_size=self.cfg.optim.batch_data, shuffle=False)
            all_errors = []
            with torch.no_grad():
                for x_batch, y_batch in loader:
                    xyz = x_batch[:, :3].to(self.device)
                    t = x_batch[:, 3:4].to(self.device)
                    norm_coords = self._normalize(xyz, t)
                    cond_vec = cond.vector().to(self.device).unsqueeze(0).repeat(y_batch.shape[0], 1)
                    T_net = self.model(norm_coords, cond_vec)
                    T = T_net + cond.t_env
                    target = y_batch.to(self.device).view(-1, 1)
                    rel_err = (T - target).abs() / (target.abs() + 1e-3)
                    all_errors.append(rel_err.detach().cpu().view(-1))

            if not all_errors:
                print(f"  - Condition {cond.name}: empty test loader, skipping.")
                continue

            errors = torch.cat(all_errors)
            mean_err = errors.mean().item()
            p95_err = torch.quantile(errors, 0.95).item()
            max_err = errors.max().item()

            print(
                f"  - Condition {cond.name}: "
                f"mean rel err={mean_err:.3%}, p95={p95_err:.3%}, max={max_err:.3%}"
            )
            results.append(
                {
                    "condition": cond.name,
                    "mean_rel_error": mean_err,
                    "p95_rel_error": p95_err,
                    "max_rel_error": max_err,
                }
            )

        if results:
            df = pd.DataFrame(results)
            metrics_path = save_dir / "test_metrics.csv"
            df.to_csv(metrics_path, index=False)
            print(f"Saved test metrics to {metrics_path}")
