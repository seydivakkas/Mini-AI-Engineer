"""
Halı Zekası Merkezi Orkestratörü (AI Carpet Intelligence Suite Orchestrator).
Renk, Arama, Kusur ve RAG Modüllerini Tek Bir Üretim Hattında Birleştirir.
"""

from typing import Dict, Any, Optional
from PIL import Image
from .moduller.renk_motoru import RenkZekasiMotoru
from .moduller.arama_motoru import GorselAramaMotoru
from .moduller.kusur_motoru import KusurTespitMotoru
from .moduller.rag_motoru import SektorelRAGMotoru


class HaliZekasiOrkestrator:
    """Tek bir halı görseli üzerinden 4 yapay zeka modülünü çalıştırarak fabrika teftiş raporu üretir."""

    def __init__(self):
        self.renk_motoru = RenkZekasiMotoru()
        self.arama_motoru = GorselAramaMotoru()
        self.kusur_motoru = KusurTespitMotoru()
        self.rag_motoru = SektorelRAGMotoru()

    def tam_denetim_yap(
        self,
        test_gorseli: Image.Image,
        referans_gorseli: Optional[Image.Image] = None,
        k_iplik: int = 5
    ) -> Dict[str, Any]:
        """Tüm zeka motorlarını sırayla çalıştırıp konsolide teftiş raporu oluşturur."""
        # 1. Renk Zekası
        renk_sonuc = self.renk_motoru.analiz_et(test_gorseli, k_iplik=k_iplik)

        # 2. Görsel Arama & Katalog Eşleşmesi
        arama_sonuc = self.arama_motoru.ara(test_gorseli, top_k=3)

        # 3. Dokuma Kusur Tespiti
        kusur_sonuc = self.kusur_motoru.tespit_et(test_gorseli, referans_gorseli=referans_gorseli)

        # 4. Kusurlar İçin Otomatik RAG Çözüm Danışmanı
        rag_onerileri = []
        for k in kusur_sonuc["kusurlar"]:
            oneri = self.rag_motoru.hata_icin_cozum_getir(k["kusur_turu"])
            rag_onerileri.append({
                "kusur_id": k["kusur_id"],
                "kusur_turu": k["kusur_turu"],
                "siddet": k["siddet"],
                "standart": oneri["standart_adi"],
                "oneri": oneri["oneri"]
            })

        # 5. Fabrika Genel Kalite Skoru ve Üretim Hattı Kararı
        kritik_sayisi = kusur_sonuc["kritik_kusur_sayisi"]
        toplam_kusur = kusur_sonuc["kusur_sayisi"]

        # Skor Hesaplama
        ceza = (kritik_sayisi * 35.0) + ((toplam_kusur - kritik_sayisi) * 12.0)
        if not renk_sonuc["parti_renk_uyumu"]:
            ceza += 15.0

        genel_kalite_skoru = float(max(0.0, min(100.0, 100.0 - ceza)))

        if kritik_sayisi > 0 or genel_kalite_skoru < 65.0:
            fabrika_karari = "PARTI_RED_URETIMI_DURDUR"
            onay = False
        elif genel_kalite_skoru < 85.0 or toplam_kusur > 0:
            fabrika_karari = "PARTI_KABUL_2_KALITE_SEVK"
            onay = True
        else:
            fabrika_karari = "PARTI_ONAYLANDI_1_KALITE_PREMIUM"
            onay = True

        return {
            "genel_kalite_skoru": genel_kalite_skoru,
            "fabrika_karari": fabrika_karari,
            "sevkiyat_onayi": onay,
            "renk_analizi": renk_sonuc,
            "gorsel_arama": arama_sonuc,
            "kusur_tespiti": kusur_sonuc,
            "rag_cozum_onerileri": rag_onerileri
        }
