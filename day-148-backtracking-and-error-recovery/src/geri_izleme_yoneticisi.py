"""
Geri İzleme ve Hata Kurtarma Yöneticisi Modülü (Day 148 - Faz 8).
Reasoning LLM'lerin içsel monoloğu ve dinamik geri izleme orkestratörü.
"""

from typing import List, Dict, Any, Optional
from .dusunce_yigini import DusunceYigini, DusunceKaresi
from .cikmaz_sokak_tespitcisi import CikmazSokakTespitcisi


class GeriIzlemeYoneticisi:
    """Çıkmaz sokakları fark edip geri dönen ve doğru yolu bulan akıl yürütme motoru."""

    def __init__(self):
        self.yigin = DusunceYigini()
        self.ic_monolog_kayitlari: List[str] = []

    def akil_yurut_ve_kurtar(self, baslangic_durumu: Dict[str, Any], aday_adımlar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aday adımları sırayla dener. Çıkmazla karşılaşırsa otomatik geri sarar ve alternatif dener.
        """
        self.yigin = DusunceYigini()
        self.ic_monolog_kayitlari = []

        # 1. Kök Kontrol Noktasını Ekle
        kok_kare = DusunceKaresi(
            adim_id=0,
            durum_verisi=baslangic_durumu,
            aciklama="Başlangıç: Problem tanımlandı",
            kontrol_noktasi_mi=True,
        )
        self.yigin.ekle(kok_kare)

        for adim in aday_adımlar:
            adim_metni = adim.get("metin", "")
            yeni_durum = adim.get("yeni_durum", {})
            kontrol_noktasi_mi = adim.get("kontrol_noktasi_mi", False)

            # Çıkmaz denetimi
            cikmaz_mi, hata_nedeni, guven = CikmazSokakTespitcisi.denetle(adim_metni, yeni_durum)

            if cikmaz_mi:
                # İçsel Monolog / Düzeltme Tetiklenir
                monolog = f"<think> Bekle, bu çıkarım hatalı! ({hata_nedeni}) Geri dönüyorum... </think>"
                self.ic_monolog_kayitlari.append(monolog)

                # Geri sarma (Backtrack / Rollback)
                self.yigin.son_gecerli_kontrol_noktasina_geri_sar(hata_nedeni=hata_nedeni)
            else:
                # Geçerli adım yığına eklenir
                yeni_id = self.yigin.boyut()
                yeni_kare = DusunceKaresi(
                    adim_id=yeni_id,
                    durum_verisi=yeni_durum,
                    aciklama=adim_metni,
                    kontrol_noktasi_mi=kontrol_noktasi_mi,
                )
                self.yigin.ekle(yeni_kare)

        return {
            "nihai_gecerli_zincir": self.yigin.aktif_zincir(),
            "toplam_geri_izleme_sayisi": len(self.yigin.geri_izleme_kayitlari),
            "geri_izleme_kayitlari": self.yigin.geri_izleme_kayitlari,
            "ic_monologlar": self.ic_monolog_kayitlari,
            "nihai_durum": self.yigin.son_kare().durum_verisi if self.yigin.son_kare() else {},
            "basarili_mi": len(self.yigin.aktif_zincir()) >= 3,
        }
