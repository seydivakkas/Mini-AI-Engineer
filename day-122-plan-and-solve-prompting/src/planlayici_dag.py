"""
Görev Ayrıştırma ve Yönlü Döngüsüz Çizge (DAG) Modülü (Day 122 - Faz 7).
Karmaşık görevlerin alt görevlere ayrıştırılması, bağımlılıkların yönetimi ve topolojik sıralama.
"""

from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, deque


class AltGorev:
    """Tek bir alt görevi temsil eden veri sınıfı."""

    def __init__(
        self,
        id: str,
        tanim: str,
        bagimliliklar: Optional[List[str]] = None,
        arac_adi: Optional[str] = None,
        girdi_sablonu: Optional[str] = None,
    ):
        self.id = id
        self.tanim = tanim
        self.bagimliliklar = bagimliliklar or []
        self.arac_adi = arac_adi
        self.girdi_sablonu = girdi_sablonu or ""
        self.durum = "bekliyor"  # "bekliyor", "calisiyor", "tamamlandi", "basarisiz"
        self.sonuc: Any = None

    def __repr__(self):
        return f"<AltGorev id='{self.id}' durum='{self.durum}' bagimliliklar={self.bagimliliklar}>"


class GorevDAG:
    """Alt görevler arasındaki bağımlılıkları yöneten Yönlü Döngüsüz Çizge (DAG)."""

    def __init__(self):
        self.gorevler: Dict[str, AltGorev] = {}
        self.komsuluk: Dict[str, List[str]] = defaultdict(list)
        self.giris_dereceleri: Dict[str, int] = defaultdict(int)

    def gorev_ekle(self, gorev: AltGorev):
        """Çizgeye yeni bir alt görev ekler."""
        if gorev.id in self.gorevler:
            raise ValueError(f"Görev ID zaten mevcut: {gorev.id}")

        self.gorevler[gorev.id] = gorev
        self.giris_dereceleri[gorev.id] = len(gorev.bagimliliklar)

        for bagimli_olunan_id in gorev.bagimliliklar:
            self.komsuluk[bagimli_olunan_id].append(gorev.id)

    def dongu_var_mi(self) -> bool:
        """Çizgede döngü (Cycle) olup olmadığını DFS ile kontrol eder."""
        ziyaret_edildi: Set[str] = set()
        yigit: Set[str] = set()

        def dfs(node: str) -> bool:
            ziyaret_edildi.add(node)
            yigit.add(node)
            for komsu in self.komsuluk[node]:
                if komsu not in ziyaret_edildi:
                    if dfs(komsu):
                        return True
                elif komsu in yigit:
                    return True
            yigit.remove(node)
            return False

        for node in self.gorevler:
            if node not in ziyaret_edildi:
                if dfs(node):
                    return True
        return False

    def topolojik_sirala(self) -> List[AltGorev]:
        """Kahn algoritması ile görevleri yürütülme sırasına göre topolojik sıralar."""
        if self.dongu_var_mi():
            raise ValueError("Çizgede döngü (Cycle) tespit edildi! Topolojik sıralama yapılamaz.")

        dereceler = {g_id: len(g.bagimliliklar) for g_id, g in self.gorevler.items()}
        kuyruk = deque([g_id for g_id, d in dereceler.items() if d == 0])
        sirali_id_listesi: List[str] = []

        while kuyruk:
            mevcut_id = kuyruk.popleft()
            sirali_id_listesi.append(mevcut_id)

            for komsu_id in self.komsuluk[mevcut_id]:
                dereceler[komsu_id] -= 1
                if dereceler[komsu_id] == 0:
                    kuyruk.append(komsu_id)

        if len(sirali_id_listesi) != len(self.gorevler):
            raise ValueError("Topolojik sıralama tamamlanamadı. Çözümlenemeyen bağımlılıklar var.")

        return [self.gorevler[g_id] for g_id in sirali_id_listesi]

    def hazir_gorevleri_getir(self) -> List[AltGorev]:
        """Tüm bağımlılıkları tamamlanmış ve çalıştırılmaya hazır görevleri döndürür."""
        hazirlar = []
        for g_id, g in self.gorevler.items():
            if g.durum == "bekliyor":
                hepsi_tamam = all(
                    self.gorevler[bag_id].durum == "tamamlandi"
                    for bag_id in g.bagimliliklar
                )
                if hepsi_tamam:
                    hazirlar.append(g)
        return hazirlar
