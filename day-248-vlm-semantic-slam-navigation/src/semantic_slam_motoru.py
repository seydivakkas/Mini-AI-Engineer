"""
VLM Destekli Semantik SLAM ve Otonom Navigasyon Motoru (Day 248).
Doluluk Izgarası (Occupancy Grid), VLM Açık-Kelime Ankrajı ve A* Semantik Rotalama.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import heapq


class OccupancyGridMap:
    """2D Doluluk Izgarası (Occupancy Grid) ve Maliyet Haritası (Costmap) Yöneticisi."""

    def __init__(self, genislik: int = 50, yukseklik: int = 50, cozunurluk_m: float = 0.1):
        self.W = genislik
        self.H = yukseklik
        self.cozunurluk = cozunurluk_m
        self.izgara = np.zeros((yukseklik, genislik), dtype=np.float32)  # 0.0: Boş, 1.0: Engel

    def add_obstacle(self, x: int, y: int, yaricap: int = 1):
        """Izgaraya statik engel veya duvar ekler."""
        for dy in range(-yaricap, yaricap + 1):
            for dx in range(-yaricap, yaricap + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    self.izgara[ny, nx] = 1.0

    def compute_inflation_costmap(self, guvenlik_yaricapi: int = 2) -> np.ndarray:
        """Robotun boyutuna göre engellerin etrafına güvenlik şişirmesi (inflation layer) ekler."""
        costmap = self.izgara.copy()
        engeller = np.argwhere(self.izgara > 0.5)

        for ey, ex in engeller:
            for dy in range(-guvenlik_yaricapi, guvenlik_yaricapi + 1):
                for dx in range(-guvenlik_yaricapi, guvenlik_yaricapi + 1):
                    mesafe = np.sqrt(dx**2 + dy**2)
                    if mesafe <= guvenlik_yaricapi:
                        nx, ny = ex + dx, ey + dy
                        if 0 <= nx < self.W and 0 <= ny < self.H:
                            maliyet = max(0.0, 1.0 - (mesafe / (guvenlik_yaricapi + 1)))
                            costmap[ny, nx] = max(costmap[ny, nx], maliyet)

        return costmap


class VLMSemanticAnchor:
    """Açık Kelime Dağarcıklı VLM Semantik Yer İmleri ve Eşleme Motoru."""

    def __init__(self):
        self.semantik_nesneler: List[Dict[str, Any]] = [
            {"id": "cup_red", "etiket": "kırmızı kahve kupası", "pos": (12, 38), "kategori": "mutfak"},
            {"id": "water_bottle", "etiket": "mavi su şişesi", "pos": (35, 15), "kategori": "içecek"},
            {"id": "charging_dock", "etiket": "şarj istasyonu", "pos": (42, 42), "kategori": "elektronik"},
            {"id": "bookshelf", "etiket": "ahşap kitaplık", "pos": (20, 10), "kategori": "mobilya"},
        ]

    def ground_language_query(self, dogal_dil_sorgusu: str) -> Dict[str, Any]:
        """Doğal dil sorgusunu haritadaki en uygun semantik nesneye eşitler (VLM Grounding)."""
        sorgu_kucuk = dogal_dil_sorgusu.lower()
        en_iyi_nesne = None
        en_yuksek_benzerlik = -1.0

        for nesne in self.semantik_nesneler:
            etiket = nesne["etiket"].lower()
            # Kelime bazlı VLM benzerlik yaklaşımı
            ortak_kelimeler = sum(1 for w in sorgu_kucuk.split() if w in etiket)
            benzerlik = (ortak_kelimeler + 0.1) / max(len(sorgu_kucuk.split()), 1)

            # Özel anahtar kelime eşleşmesi
            if any(k in sorgu_kucuk for k in ["kupa", "kahve", "kırmızı"]) and "kupa" in etiket:
                benzerlik = 0.95
            elif any(k in sorgu_kucuk for k in ["şişe", "su", "mavi"]) and "şişe" in etiket:
                benzerlik = 0.92
            elif any(k in sorgu_kucuk for k in ["şarj", "istasyon", "dock"]) and "şarj" in etiket:
                benzerlik = 0.98
            elif any(k in sorgu_kucuk for k in ["kitap", "kitaplık", "ahşap"]) and "kitaplık" in etiket:
                benzerlik = 0.94

            if benzerlik > en_yuksek_benzerlik:
                en_yuksek_benzerlik = benzerlik
                en_iyi_nesne = nesne

        return {
            "sorgu": dogal_dil_sorgusu,
            "eslesen_nesne": en_iyi_nesne,
            "guven_skoru": round(float(en_yuksek_benzerlik), 3),
            "hedef_koordinat": en_iyi_nesne["pos"] if en_iyi_nesne else (25, 25),
        }


class AStarPathPlanner:
    """8 Bağlantılı Güvenli Izgara Üzerinde Optimum A* Yol Planlayıcısı."""

    @classmethod
    def plan_path(cls, costmap: np.ndarray, baslangic: Tuple[int, int], hedef: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* algoritması ile engellerden kaçınarak hedefe en kısa yolu üretir."""
        H, W = costmap.shape
        start_x, start_y = baslangic
        goal_x, goal_y = hedef

        # Öncelik Kuyruğu: (f_score, x, y)
        open_set = []
        heapq.heappush(open_set, (0.0, start_x, start_y))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {baslangic: 0.0}

        yonler = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (goal_x, goal_y):
                # Yolu Geriye Doğru Çıkar
                yol = [(cx, cy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    yol.append((cx, cy))
                yol.reverse()
                return yol

            for dx, dy in yonler:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < W and 0 <= ny < H:
                    # Engelli / Tehlikeli Alan Kontrolü (Cost >= 0.85)
                    if costmap[ny, nx] >= 0.85:
                        continue

                    hareket_maliyeti = np.sqrt(dx**2 + dy**2) + costmap[ny, nx] * 3.0
                    yeni_g = g_score.get((cx, cy), float("inf")) + hareket_maliyeti

                    if yeni_g < g_score.get((nx, ny), float("inf")):
                        came_from[(nx, ny)] = (cx, cy)
                        g_score[(nx, ny)] = yeni_g
                        # Öklid Sezgisel (Heuristic)
                        h_score = np.sqrt((nx - goal_x)**2 + (ny - goal_y)**2)
                        f_score = yeni_g + h_score
                        heapq.heappush(open_set, (f_score, nx, ny))

        return [baslangic]  # Hedefe ulaşılamazsa başlangıcı dön


class SemanticSLAMSystem:
    """Lidar Izgarası, VLM Ankrajı ve A* Navigasyonunu Birleştiren Otonom Sistem."""

    def __init__(self, W: int = 50, H: int = 50):
        self.harita = OccupancyGridMap(genislik=W, yukseklik=H)
        self.vlm = VLMSemanticAnchor()
        self.robot_pos = (5, 5)

        # Varsayılan Duvarlar ve Odalar
        self._harita_kur()

    def _harita_kur(self):
        """Oda duvarları ve engelleri tanımlar."""
        # Dış Duvarlar
        for x in range(50):
            self.harita.add_obstacle(x, 0)
            self.harita.add_obstacle(x, 49)
        for y in range(50):
            self.harita.add_obstacle(0, y)
            self.harita.add_obstacle(49, y)

        # İç Bölme Duvarı (Kapı Aralıklı)
        for y in range(10, 40):
            if y not in [24, 25, 26]:  # Kapı boşluğu
                self.harita.add_obstacle(25, y)

    def navigate_with_language(self, sorgu: str) -> Dict[str, Any]:
        """Doğal dil komutunu çözer, hedefi bulur ve A* yörüngesi oluşturur."""
        grounding = self.vlm.ground_language_query(sorgu)
        hedef_pos = grounding["hedef_koordinat"]

        costmap = self.harita.compute_inflation_costmap(guvenlik_yaricapi=2)
        yol = AStarPathPlanner.plan_path(costmap, self.robot_pos, hedef_pos)

        return {
            "sorgu": sorgu,
            "hedef_nesne": grounding["eslesen_nesne"]["etiket"],
            "hedef_pos": hedef_pos,
            "guven_skoru": grounding["guven_skoru"],
            "yol_nokta_sayisi": len(yol),
            "yol_koordinatlari": yol,
            "basarili": len(yol) > 1,
        }
