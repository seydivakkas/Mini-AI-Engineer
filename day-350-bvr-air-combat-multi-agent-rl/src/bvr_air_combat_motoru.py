"""
Day 350: Beyond Visual Range (BVR) Air Combat Multi-Agent Reinforcement Learning (MARL)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Görüş Ötesi (BVR) Hava Muharebesi Kinematiğini, Aktif Radar Güdümlü Füze (ARH)
ve Çoklu Ajan Takviyeli Öğrenme (MARL) Taktik Karar Politikalarını (Crank, Pump, F-Pole) içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class ActiveRadarMissile:
    """
    Aktif Radar Güdümlü (ARH - Active Radar Homing) Görüş Ötesi Hava-Hava Füzesi (Örn: GÖKDOĞAN / AMRAAM).
    Oransal Seyrüsefer (Proportional Navigation - PNG) Güdüm Yasası.
    """
    def __init__(self, missile_id: str, shooter_id: str, target_id: str, pos_km: np.ndarray, heading_rad: float):
        self.missile_id = missile_id
        self.shooter_id = shooter_id
        self.target_id = target_id
        self.pos = pos_km.copy()
        self.speed_kms = 1.1 # Mach 3.5 (~1100 m/s = 1.1 km/s)
        self.heading = heading_rad
        self.is_active = True
        self.is_pitbull = False # Terminal otonom aktif radar arayıcı fazı
        self.fuel_sec = 25.0

    def step(self, dt: float, target_pos: np.ndarray, shooter_has_radar_lock: bool) -> bool:
        """
        Füze güdüm adımı. Hedefe çarptıysa True döner.
        """
        if not self.is_active or self.fuel_sec <= 0:
            self.is_active = False
            return False

        self.fuel_sec -= dt
        to_target = target_pos - self.pos
        dist_km = float(np.linalg.norm(to_target))

        # 15 km altında otonom Pitbull fazına geçer, veri bağı ihtiyacı biter
        if dist_km < 15.0:
            self.is_pitbull = True

        # Pitbull öncesinde atıcı radar kilidini kaybederse füze kör uçar
        if not self.is_pitbull and not shooter_has_radar_lock:
            # Düz uçar (Balistik sürüklenme)
            pass
        else:
            # PNG Güdüm: Hedef görüş hattı açısına (LOS) yönelme
            los_angle = float(np.arctan2(to_target[1], to_target[0]))
            d_angle = (los_angle - self.heading + np.pi) % (2 * np.pi) - np.pi
            turn_rate = np.clip(3.0 * d_angle, -0.35, 0.35) # Max 20 deg/s
            self.heading += turn_rate * dt

        # Pozisyon güncelle
        vel = np.array([self.speed_kms * np.cos(self.heading), self.speed_kms * np.sin(self.heading)])
        self.pos += vel * dt

        # Çarpışma / Patlama yarıçapı (< 0.15 km = 150m Proximity Fuze)
        if dist_km < 0.15:
            self.is_active = False
            return True

        return False


class BVRFighterAgent:
    """
    Görüş Ötesi (BVR) Muharip Savaş Uçağı Ajanı (Blue vs Red).
    """
    def __init__(self, fighter_id: str, team: str, pos_km: np.ndarray, heading_rad: float):
        self.fighter_id = fighter_id
        self.team = team # "BLUE" veya "RED"
        self.pos = pos_km.copy()
        self.speed_kms = 0.35 # Mach 1.1 (~350 m/s = 0.35 km/s)
        self.heading = heading_rad
        self.is_alive = True
        self.missile_ammo = 2
        self.radar_gimbal_deg = 60.0 # Radar tarama açısı (+-60 deg)
        self.tactical_state = "INTERCEPT" # "INTERCEPT", "CRANK", "DRAG_PUMP"

    def has_radar_lock(self, target_pos: np.ndarray) -> bool:
        """Hedef radar tarama konisi (Gimbal Limit) içinde mi kontrol eder."""
        if not self.is_alive:
            return False
        to_target = target_pos - self.pos
        los_angle = np.arctan2(to_target[1], to_target[0])
        angle_off_nose = np.abs((los_angle - self.heading + np.pi) % (2 * np.pi) - np.pi)
        return bool(np.rad2deg(angle_off_nose) <= self.radar_gimbal_deg)


class MARLTacticalPolicy:
    """
    Çoklu Ajan Takviyeli Öğrenme (MARL) Tabanlı BVR Taktik Karar Politikası.
    Atış sonrası Crank açısına dönme, F-Pole optimizasyonu ve füze tehdidinde Pump manevrası yapar.
    """
    @staticmethod
    def decide_action(
        fighter: BVRFighterAgent,
        target: BVRFighterAgent,
        incoming_missiles: List[ActiveRadarMissile],
        my_missiles: List[ActiveRadarMissile],
        dt: float
    ) -> Optional[ActiveRadarMissile]:
        """
        Ajanın yönelme, hızlanma ve füze ateşleme kararlarını üretir.
        """
        if not fighter.is_alive or not target.is_alive:
            return None

        fired_missile = None
        to_target = target.pos - fighter.pos
        dist_to_target = float(np.linalg.norm(to_target))
        los_angle = float(np.arctan2(to_target[1], to_target[0]))

        # 1. Gelen Tehdit Füzesi Var mı? -> DRAG / PUMP Manevrası (180 derece kaçış)
        threat_missiles = [m for m in incoming_missiles if m.is_active and m.target_id == fighter.fighter_id]
        has_active_missile = any(m.is_active and not m.is_pitbull and m.shooter_id == fighter.fighter_id for m in my_missiles)
        
        if len(threat_missiles) > 0 and (threat_missiles[0].is_pitbull or dist_to_target < 20.0):
            fighter.tactical_state = "DRAG_PUMP"
            threat_m = threat_missiles[0]
            escape_heading = (threat_m.heading + np.pi) % (2 * np.pi)
            d_head = (escape_heading - fighter.heading + np.pi) % (2 * np.pi) - np.pi
            fighter.heading += np.clip(d_head, -0.25, 0.25) * dt * 5.0

        # 2. Aktif Ateşlediğim Füze Pitbull Fazına Geçmediyse -> CRANK Manevrası (Radar limitinde 52 derece dön)
        elif has_active_missile:
            fighter.tactical_state = "CRANK"
            crank_heading = los_angle + np.deg2rad(52.0)
            d_head = (crank_heading - fighter.heading + np.pi) % (2 * np.pi) - np.pi
            fighter.heading += np.clip(d_head, -0.25, 0.25) * dt * 5.0

        # 3. Yaklaşma ve Ateşleme Fazı -> INTERCEPT
        else:
            fighter.tactical_state = "INTERCEPT"
            d_head = (los_angle - fighter.heading + np.pi) % (2 * np.pi) - np.pi
            fighter.heading += np.clip(d_head, -0.25, 0.25) * dt * 5.0

            # 45 km altına inildiğinde ve mühimmat varsa füze ateşle (MAR - Max Abort Range)
            if dist_to_target <= 45.0 and fighter.missile_ammo > 0:
                fighter.missile_ammo -= 1
                fired_missile = ActiveRadarMissile(
                    missile_id=f"MSL_{fighter.fighter_id}_{target.fighter_id}",
                    shooter_id=fighter.fighter_id,
                    target_id=target.fighter_id,
                    pos_km=fighter.pos,
                    heading_rad=fighter.heading
                )
                fighter.tactical_state = "CRANK"

        # Uçak Konumunu İlerlet
        vel = np.array([fighter.speed_kms * np.cos(fighter.heading), fighter.speed_kms * np.sin(fighter.heading)])
        fighter.pos += vel * dt

        return fired_missile


class BVRAirCombatArena:
    """
    2v2 Mavi vs Kırmızı Görüş Ötesi Hava Muharebesi Simülasyon Arenası.
    """
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        # Blue Team (AI MARL Ajanları)
        self.blue_1 = BVRFighterAgent("BLUE_LEAD", "BLUE", np.array([-40.0, 10.0]), np.deg2rad(0.0))
        self.blue_2 = BVRFighterAgent("BLUE_WING", "BLUE", np.array([-40.0, -10.0]), np.deg2rad(0.0))

        # Red Team (Hedef Düşman Savaş Uçakları)
        self.red_1 = BVRFighterAgent("RED_LEAD", "RED", np.array([40.0, 10.0]), np.deg2rad(180.0))
        self.red_2 = BVRFighterAgent("RED_WING", "RED", np.array([40.0, -10.0]), np.deg2rad(180.0))

        self.missiles: List[ActiveRadarMissile] = []
        self.policy = MARLTacticalPolicy()

    def step_arena(self) -> Dict[str, Any]:
        """Bir taktik zaman adımını simüle eder."""
        # 1. Blue Kararları
        f1 = self.policy.decide_action(self.blue_1, self.red_1, self.missiles, self.missiles, self.dt)
        if f1: self.missiles.append(f1)

        f2 = self.policy.decide_action(self.blue_2, self.red_2, self.missiles, self.missiles, self.dt)
        if f2: self.missiles.append(f2)

        # 2. Red Kararları
        r1 = self.policy.decide_action(self.red_1, self.blue_1, self.missiles, self.missiles, self.dt)
        if r1: self.missiles.append(r1)

        r2 = self.policy.decide_action(self.red_2, self.blue_2, self.missiles, self.missiles, self.dt)
        if r2: self.missiles.append(r2)

        # 3. Füzeleri İlerlet
        fighter_map = {
            "BLUE_LEAD": self.blue_1, "BLUE_WING": self.blue_2,
            "RED_LEAD": self.red_1, "RED_WING": self.red_2
        }

        for msl in self.missiles:
            if msl.is_active:
                target = fighter_map[msl.target_id]
                shooter = fighter_map[msl.shooter_id]
                has_lock = shooter.has_radar_lock(target.pos)
                hit = msl.step(self.dt, target.pos, has_lock)
                if hit:
                    target.is_alive = False

        return {
            "blue_alive": sum([self.blue_1.is_alive, self.blue_2.is_alive]),
            "red_alive": sum([self.red_1.is_alive, self.red_2.is_alive]),
            "positions": {k: v.pos.copy() for k, v in fighter_map.items()},
            "missiles": [(m.pos.copy(), m.is_active) for m in self.missiles]
        }
