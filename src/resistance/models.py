"""
Shared data models for resistance and propulsion calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class ShipResistanceSpecs:
    Lwl: float
    Lpp: float
    Lbp: float
    Beam: float
    Draft: float
    Depth: float
    VolumeOfDisplacement: float
    Cb: float
    Cm: float
    S: float
    LCB_Percent: float
    PropellerDiameter: float
    AEAO: float
    NumberOfBlades: int
    N_MCR: float
    DesignSpeed: float
    eta_drive: float = 0.98
    ContainerHeight: float = 2.6
    ContainerTiersOnDeck: int = 3
    SternShapeCoefficient: float = 10.0

    @property
    def eta_shaft(self) -> float:
        """Backward-compatible alias for shaft efficiency."""
        return self.eta_drive

    @eta_shaft.setter
    def eta_shaft(self, value: float) -> None:
        self.eta_drive = float(value)

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "ShipResistanceSpecs":
        return cls(
            Lwl=float(data.get("Lwl", data.get("Lpp", data.get("Lbp")))),
            Lpp=float(data.get("Lpp", data.get("Lbp", data.get("Lwl")))),
            Lbp=float(data.get("Lbp", data.get("Lpp", data.get("Lwl")))),
            Beam=float(data.get("Beam", data.get("B"))),
            Draft=float(data.get("Draft", data.get("T"))),
            Depth=float(data.get("Depth", data.get("D", 19.0))),
            VolumeOfDisplacement=float(
                data.get("VolumeOfDisplacement", data.get("displacement_vol"))
            ),
            Cb=float(data["Cb"]),
            Cm=float(data.get("Cm", data.get("CM", 0.985))),
            S=float(data.get("S", data.get("Surface_area"))),
            LCB_Percent=float(data.get("LCB_Percent", data.get("LCB", 0.0))),
            PropellerDiameter=float(data.get("PropellerDiameter", 7.9)),
            AEAO=float(data.get("AEAO", 0.65)),
            NumberOfBlades=int(data.get("NumberOfBlades", 5)),
            N_MCR=float(data.get("N_MCR", 104.0)),
            DesignSpeed=float(data.get("DesignSpeed", 24.0)),
            eta_drive=float(data.get("eta_shaft", data.get("eta_drive", 0.98))),
            ContainerHeight=float(data.get("ContainerHeight", 2.6)),
            ContainerTiersOnDeck=int(data.get("ContainerTiersOnDeck", 4)),
            SternShapeCoefficient=float(data.get("SternShapeCoefficient", 10.0)),
        )


@dataclass
class EnvironmentData:
    wind_speed_ms: float = 0.0
    wind_dir_deg: float | None = None
    current_speed_ms: float = 0.0
    current_dir_deg: float = 0.0
    wave_height_m: float = 0.0
    wave_period_s: float = 0.0
    wave_dir_deg: float = 0.0
    rel_wind_speed_ms: float | None = None
    rel_wind_dir_deg: float | None = None


@dataclass
class CalmWaterResult:
    total_resistance_kn: float
    calm_resistance_n: float
    wake_fraction: float
    thrust_deduction: float
    relative_rotation_efficiency: float


@dataclass
class ResistanceResult:
    total_resistance_n: float
    calm_resistance_n: float
    wind_resistance_n: float
    wave_resistance_n: float
    speed_over_ground_knots: float
    wake_fraction: float
    thrust_deduction: float
    relative_rotation_efficiency: float
    relative_wind_speed_ms: float = 0.0
    relative_wind_dir_deg: float = 0.0


@dataclass
class PropulsionResult:
    power_kw: float
    rpm: float
    torque_knm: float
    j_op: float
    v_ideal: float
    eta_d: float
    eta_o: float
