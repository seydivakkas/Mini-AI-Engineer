"""
Ajan Hafıza Sistemleri Motoru (Day 225 - FAZ 12).
Kısa Vadeli Çalışma Belleği ve Vektörel Uzun Vadeli Epizodik Bellek (MemGPT & Generative Agents).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class MemoryItem:
    """Hafıza Hücresi Veri Yapısı."""

    def __init__(
        self,
        hafiza_id: int,
        icerik: str,
        vektor: List[float],
        onem_puani: float = 0.5,
        zaman_damgasi: int = 1,
        tur: str = "KISA_VADELI",
    ):
        self.hafiza_id = hafiza_id
        self.icerik = icerik
        self.vektor = np.array(vektor, dtype=np.float32)
        self.onem_puani = float(onem_puani)
        self.zaman_damgasi = int(zaman_damgasi)
        self.tur = tur  # 'KISA_VADELI', 'EPIZODIK', 'SEMANTIK'


class ShortTermWorkingMemory:
    """Kısa Vadeli Çalışma Belleği (Aktif Bağlam Penceresi - FIFO & Özetleme)."""

    def __init__(self, kapasite: int = 4):
        self.kapasite = kapasite
        self.tampon: List[str] = []

    def ekle(self, mesaj: str) -> Optional[str]:
        """Mesajı aktif belleğe ekler; taşma olursa taşan mesajı döner."""
        tasim_verisi = None
        if len(self.tampon) >= self.kapasite:
            tasim_verisi = self.tampon.pop(0)  # FIFO boşaltma
        self.tampon.append(mesaj)
        return tasim_verisi

    def baglam_metni(self) -> str:
        """Mevcut kısa vadeli bağlamı prompt metnine dönüştürür."""
        return "\n".join([f"- {m}" for m in self.tampon])


class LongTermVectorMemory:
    """Uzun Vadeli Vektörel Epizodik Bellek (Anlamsal Arama & Yenilik-Önem Puanlaması)."""

    def __init__(self):
        self.hafiza_havuzu: List[MemoryItem] = []
        self._sayac = 1

    def ekle(self, icerik: str, vektor: List[float], onem: float = 0.5, zaman: int = 1, tur: str = "EPIZODIK") -> MemoryItem:
        """Yeni bir uzun vadeli epizodik hatıra kaydeder."""
        item = MemoryItem(
            hafiza_id=self._sayac,
            icerik=icerik,
            vektor=vektor,
            onem_puani=onem,
            zaman_damgasi=zaman,
            tur=tur,
        )
        self.hafiza_havuzu.append(item)
        self._sayac += 1
        return item

    def benzerlik_hesapla(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Kosinüs Benzerliği (Cosine Similarity)."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def sorgula(
        self,
        sorgu_vektoru: List[float],
        mevcut_zaman: int = 10,
        top_k: int = 2,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Üçlü Ağırlıklı Geri Çağırma (Generative Agents Formülü):
        Skor = 0.5 * Anlamsal_Benzerlik + 0.3 * Onem_Puani + 0.2 * Yenilik (Recency)
        """
        if not self.hafiza_havuzu:
            return []

        q_vec = np.array(sorgu_vektoru, dtype=np.float32)
        skorlu_liste = []

        for item in self.hafiza_havuzu:
            sim = self.benzerlik_hesapla(q_vec, item.vektor)
            # Yenilik faktörü (Zaman farkı ne kadar azsa o kadar taze)
            zaman_farki = max(1, mevcut_zaman - item.zaman_damgasi)
            yenilik = np.exp(-0.1 * zaman_farki)

            bilesik_skor = (0.5 * sim) + (0.3 * item.onem_puani) + (0.2 * yenilik)
            skorlu_liste.append((item, float(bilesik_skor)))

        skorlu_liste.sort(key=lambda x: x[1], reverse=True)
        return skorlu_liste[:top_k]


class AgenticMemorySystem:
    """Çift Kademeli Birleşik Ajan Hafıza Sistemi."""

    def __init__(self, kisa_vadeli_kapasite: int = 3):
        self.kisa_bellek = ShortTermWorkingMemory(kapasite=kisa_vadeli_kapasite)
        self.uzun_bellek = LongTermVectorMemory()
        self.zaman_sayaci = 1

    def etkilesim_kaydet(
        self,
        icerik: str,
        vektor: List[float],
        onem_puani: float = 0.5,
        uzun_vadeye_konsolide_et: bool = False,
    ) -> None:
        """Kısa vadeli hafızaya yazar; taşan veya önemli bilgileri uzun vadeye aktarır."""
        tasim = self.kisa_bellek.ekle(icerik)
        self.zaman_sayaci += 1

        if uzun_vadeye_konsolide_et or (tasim is not None and onem_puani >= 0.7):
            # Uzun vadeli epizodik belleğe kalıcı kaydet
            self.uzun_bellek.ekle(
                icerik=icerik if uzun_vadeye_konsolide_et else tasim,
                vektor=vektor,
                onem=onem_puani,
                zaman=self.zaman_sayaci,
                tur="EPIZODIK",
            )

    def dinamik_baglam_olustur(self, sorgu: str, sorgu_vektoru: List[float]) -> str:
        """Sorguyla eşleşen uzun vadeli anıları ve aktif kısa vadeli bağlamı birleştirir."""
        hatirlananlar = self.uzun_bellek.sorgula(sorgu_vektoru, mevcut_zaman=self.zaman_sayaci, top_k=2)

        baglam = "=== [AJAN HAFIZA RAPORU] ===\n"
        baglam += "-- Hatırlanan Uzun Vadeli Epizodik Anılar --\n"
        if hatirlananlar:
            for item, skor in hatirlananlar:
                baglam += f"* (Skor: {skor:.2f}) {item.icerik}\n"
        else:
            baglam += "* Eşleşen hatıra bulunamadı.\n"

        baglam += "\n-- Aktif Kısa Vadeli Çalışma Belleği --\n"
        baglam += self.kisa_bellek.baglam_metni()
        return baglam
