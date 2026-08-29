"""
Düşünce Yığını ve Kontrol Noktası Modülü (Day 148 - Faz 8).
Stack tabanlı düşünce çerçeveleri (Thought Frames) ve geri yükleme (Rollback) yönetimi.
"""

from typing import List, Dict, Any, Optional


class DusunceKaresi:
    """Yığında saklanan tek bir düşünce adım çerçevesi."""

    def __init__(
        self,
        adim_id: int,
        durum_verisi: Dict[str, Any],
        aciklama: str,
        kontrol_noktasi_mi: bool = False,
    ):
        self.adim_id = adim_id
        self.durum_verisi = dict(durum_verisi)
        self.aciklama = aciklama
        self.kontrol_noktasi_mi = kontrol_noktasi_mi
        self.denenmis_aksiyonlar: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adim_id": self.adim_id,
            "durum_verisi": self.durum_verisi,
            "aciklama": self.aciklama,
            "kontrol_noktasi_mi": self.kontrol_noktasi_mi,
            "denenmis_aksiyonlar": self.denenmis_aksiyonlar,
        }


class DusunceYigini:
    """Geri izleme ve hata kurtarma için LIFO düşünce yığını."""

    def __init__(self):
        self.yigin: List[DusunceKaresi] = []
        self.geri_izleme_kayitlari: List[Dict[str, Any]] = []

    def bos_mu(self) -> bool:
        return len(self.yigin) == 0

    def boyut(self) -> int:
        return len(self.yigin)

    def ekle(self, kare: DusunceKaresi):
        """Yeni bir düşünce çerçevesini yığına ekler (Push)."""
        self.yigin.append(kare)

    def son_kare(self) -> Optional[DusunceKaresi]:
        """Yığının en üstündeki aktif düşünce karesini döner."""
        return self.yigin[-1] if self.yigin else None

    def son_gecerli_kontrol_noktasina_geri_sar(self, hata_nedeni: str) -> Optional[DusunceKaresi]:
        """
        Hata tespit edildiğinde, son onaylanmış kontrol noktasına kadar olan hatalı adımları yığından atar (Rollback).
        """
        silinen_adimlar = []

        while len(self.yigin) > 1:
            mevcut = self.yigin[-1]
            if mevcut.kontrol_noktasi_mi:
                break
            atilan = self.yigin.pop()
            silinen_adimlar.append(atilan.aciklama)

        self.geri_izleme_kayitlari.append({
            "hata_nedeni": hata_nedeni,
            "silinen_adim_sayisi": len(silinen_adimlar),
            "silinen_adimlar": silinen_adimlar,
            "donulen_adim_id": self.son_kare().adim_id if self.son_kare() else 0,
        })

        return self.son_kare()

    def aktif_zincir(self) -> List[str]:
        """Yığındaki geçerli düşünce açıklamalarının sırasını döner."""
        return [k.aciklama for k in self.yigin]
