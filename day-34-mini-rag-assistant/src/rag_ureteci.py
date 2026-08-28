"""
RAG Bağlam Enjeksiyonu ve Doğrulanabilir Yanıt Üretim Motoru (Context Injection & Grounded Generator).
"""

from typing import List, Dict, Any


class RAGUreteci:
    """
    Erişilen bağlam parçalarını sistem prompt'una enjekte eden,
    kaynak atıflı (grounded with citation) ve halüsinasyon korumalı yanıt sentezleyici.
    """

    PROMPT_SABLONU = (
        "### GÖREV TALİMATI:\n"
        "Aşağıdaki [BAĞLAM] bölümünde verilen teknik doküman parçalarını kullanarak kullanıcının sorusunu yanıtla.\n"
        "Yalnızca verilen bağlamdaki gerçeklere dayan; bağlam dışı spekülasyon yapma.\n"
        "Kullandığın her bilgi için cümlenin sonuna [Kaynak: CHUNK_ID] atfını ekle.\n\n"
        "### [BAĞLAM]:\n"
        "{baglam_metni}\n\n"
        "### [KULLANICI SORUSU]:\n"
        "{soru}\n\n"
        "### [UZMAN YANITI]:"
    )

    def __init__(self, guven_esigi: float = 0.20):
        self.guven_esigi = guven_esigi

    def baglam_olustur(self, getirilen_parcalar: List[Dict[str, Any]]) -> str:
        """Getirilen parça listesini yapılandırılmış prompt metnine dönüştürür."""
        if not getirilen_parcalar:
            return "Hiçbir ilgili doküman parçası bulunamadı."

        satirlar = []
        for p in getirilen_parcalar:
            satir = f"• [Kaynak: {p['chunk_id']}] ({p['baslik']}): {p['metin']}"
            satirlar.append(satir)
        return "\n".join(satirlar)

    def prompt_hazirla(self, soru: str, getirilen_parcalar: List[Dict[str, Any]]) -> str:
        """Tam RAG prompt'unu hazırlar."""
        baglam_metni = self.baglam_olustur(getirilen_parcalar)
        return self.PROMPT_SABLONU.format(baglam_metni=baglam_metni, soru=soru)

    def yanit_sentezle(self, soru: str, getirilen_parcalar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bağlam parçalarından doğrulanabilir kaynak atıflı yanıt sentezler.
        Yetersiz veya düşük benzerlik durumunda kesin ret (refusal) üretir.
        """
        if not getirilen_parcalar:
            return {
                "yanit": "Belgeler içerisinde sorunuzla ilgili yeterli bilgi bulunamadı.",
                "kaynaklar": [],
                "guven_skoru": 0.0,
                "durum": "BULUNAMADI"
            }

        en_iyi_skor = getirilen_parcalar[0]["skor"]
        if en_iyi_skor < self.guven_esigi:
            return {
                "yanit": "Verilen kurumsal dokümanlarda bu soruyu doğrulamak için yeterli kanıt tespit edilemedi.",
                "kaynaklar": [],
                "guven_skoru": float(en_iyi_skor),
                "durum": "YETERSIZ_KANIT"
            }

        # Bilgi sentezi ve atıf çıkarma
        kaynaklar = [p["chunk_id"] for p in getirilen_parcalar]
        sentez_cumleleri = []

        for p in getirilen_parcalar[:2]:
            sentez_cumleleri.append(f"{p['metin']} [Kaynak: {p['chunk_id']}]")

        uretilen_yanit = f"{getirilen_parcalar[0]['baslik']} kapsamında: " + " ".join(sentez_cumleleri)

        return {
            "yanit": uretilen_yanit,
            "kaynaklar": kaynaklar,
            "guven_skoru": float(en_iyi_skor),
            "durum": "BASARILI"
        }
