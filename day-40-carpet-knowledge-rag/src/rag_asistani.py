"""
40.3: Sektörel RAG Asistanı, Reranking, Prompt Context Injection ve Güvenlik Denetçisi.
"""

from typing import Dict, Any, List, Optional
from .vektor_deposu import TekstilVektorDeposu


class SektorelRAGAsistani:
    """
    Tekstil ve halı üretim mühendisleri için doğrulanmış standartlara dayalı
    soru-cevap, alıntı eşleme ve halüsinasyon engelleme asistanı.
    """

    def __init__(
        self,
        vektor_deposu: TekstilVektorDeposu,
        guven_esigi: float = 0.20
    ):
        self.depo = vektor_deposu
        self.guven_esigi = guven_esigi

    def prompt_olustur(self, soru: str, baglam_chunklar: List[Dict[str, Any]]) -> str:
        """LLM için sıkı bağlam enjeksiyonu (Context Injection) prompt şablonu oluşturur."""
        baglam_metinleri = []
        for i, aday in enumerate(baglam_chunklar):
            ch = aday["chunk"]
            b_str = (
                f"--- [KAYNAK {i+1} | Standart: {ch['kaynak_standart']} | Doküman: {ch['dokuman_id']} | Alt Başlık: {ch['alt_baslik']}] ---\n"
                f"{ch['metin']}\n"
            )
            baglam_metinleri.append(b_str)

        tum_baglam = "\n".join(baglam_metinleri)

        prompt = (
            "GÖREV: Sen endüstriyel halı dokuma, iplik standartları ve apre kimyası uzmanı teknik bir yapay zeka asistanısın.\n"
            "KURAL 1: Yalnızca aşağıda verilen doğrulanmış standart metinlerini referans alarak yanıt üret.\n"
            "KURAL 2: Her teknik parametre için mutlaka kaynak standardı ve alt başlığı parantez içinde belirt.\n"
            "KURAL 3: Bağlamda yer almayan hiçbir toleransı veya reçeteyi tahmin etme (Halüsinasyon Yasağı).\n\n"
            "=== DOĞRULANMIŞ TEKNİK BİLGİ BAĞLAMI ===\n"
            f"{tum_baglam}\n"
            "=========================================\n\n"
            f"KULLANICI SORUSU: {soru}\n\n"
            "UZMAN TEKNİK YANITI (Alıntılarla Birlikte):"
        )
        return prompt

    def yanit_uret(
        self,
        soru: str,
        top_k: int = 3,
        kategori_filtresi: Optional[str] = None
    ) -> Dict[str, Any]:
        """Soruya en uygun teknik kaynakları getirir, prompt bağlamı kurar ve doğrulanmış yanıt üretir."""
        adaylar = self.depo.sorgula(soru, top_k=top_k, kategori_filtresi=kategori_filtresi)

        if not adaylar or adaylar[0]["skor"] < self.guven_esigi:
            return {
                "soru": soru,
                "durum": "REDDEDILDI_BILGI_YOK",
                "yanit": "Tekstil teknik bilgi tabanında bu parametreye dair doğrulanmış bir standart veya reçete bulunmamaktadır.",
                "en_yuksek_skor": adaylar[0]["skor"] if adaylar else 0.0,
                "kaynaklar": [],
                "prompt": ""
            }

        en_iyi_aday = adaylar[0]["chunk"]
        prompt = self.prompt_olustur(soru, adaylar)

        # Doğrulanmış Sektörel Yanıt Sentezi
        yanit_ozeti = (
            f"İlgili teknik doküman ({en_iyi_aday['kaynak_standart']} - {en_iyi_aday['alt_baslik']}) uyarınca:\n"
            f"{en_iyi_aday['metin']}\n\n"
            f"[*] [Doğrulama Referansı: {en_iyi_aday['dokuman_id']} | Güven Derecesi: {en_iyi_aday['guven_derecesi']}]"
        )

        kaynaklar = []
        for a in adaylar:
            ch = a["chunk"]
            kaynaklar.append({
                "dokuman_id": ch["dokuman_id"],
                "ana_baslik": ch["ana_baslik"],
                "alt_baslik": ch["alt_baslik"],
                "kaynak_standart": ch["kaynak_standart"],
                "kategori": ch["kategori"],
                "skor": a["skor"]
            })

        return {
            "soru": soru,
            "durum": "BASARILI_YANIT",
            "yanit": yanit_ozeti,
            "en_yuksek_skor": adaylar[0]["skor"],
            "kaynaklar": kaynaklar,
            "prompt": prompt
        }
