"""
Chain-of-Thought (CoT) Akıl Yürütücü Modülü (Day 142 - Faz 8).
Özel <think> tokenleri ile farklı akıl yürütme yollarını (Reasoning Trajectories) örnekleyen motor.
"""

from typing import List, Dict, Any
import random
from .dusunce_tokenizatoru import DusunceTokenizatoru


class COTAkilYurutucu:
    """Çoklu sıcaklık örneklemesiyle çeşitli düşünce yolları üreten CoT motoru."""

    def __init__(self, tohum: int = 42):
        self.tokenizator = DusunceTokenizatoru()
        random.seed(tohum)

        # Farklı akıl yürütme stratejileri havuzu
        self._strateji_havuzu = {
            "sopave_top": [
                {
                    "strateji": "Cebirsel Modelleme",
                    "adimlar": [
                        "Değişken ata: Sopa = S, Top = T.",
                        "Denklemler: S + T = 1.10 ve S = T + 1.00.",
                        "Yerine koy: (T + 1.00) + T = 1.10 => 2T = 0.10 => T = 0.05.",
                        "Sağlama: 1.05 + 0.05 = 1.10 (Doğru).",
                    ],
                    "yanit": "5 cent ($0.05)",
                    "tahmin": "0.05",
                },
                {
                    "strateji": "Farktan Yola Çıkma (Aritmetik)",
                    "adimlar": [
                        "Toplam paradan aradaki 1.00 dolarlık farkı çıkar: 1.10 - 1.00 = 0.10.",
                        "Kalan tutar iki nesne arasında eşit bölünür: 0.10 / 2 = 0.05.",
                        "Top = 0.05 dolar, Sopa = 0.05 + 1.00 = 1.05 dolar.",
                    ],
                    "yanit": "5 cent ($0.05)",
                    "tahmin": "0.05",
                },
                {
                    "strateji": "Varsayım ve Çelişki Denetimi",
                    "adimlar": [
                        "Varsayım: Top 10 cent olsaydı, sopa 1.10 dolar olurdu.",
                        "Toplam: 1.10 + 0.10 = 1.20 dolar olurdu (Çelişki!).",
                        "Düzeltme: Top 5 cent olmalı. 1.05 + 0.05 = 1.10 dolar (Tutarlı).",
                    ],
                    "yanit": "5 cent ($0.05)",
                    "tahmin": "0.05",
                },
                {
                    "strateji": "Birim Dönüşümü (Cent Cinsinden)",
                    "adimlar": [
                        "Tüm tutarı cente çevir: 1.10 dolar = 110 cent.",
                        "Sopa toptan 100 cent pahalı: S = T + 100.",
                        "Denklem: 2T + 100 = 110 => 2T = 10 => T = 5 cent.",
                    ],
                    "yanit": "5 cent ($0.05)",
                    "tahmin": "0.05",
                },
                {
                    "strateji": "Hatalı Sezgisel Yol (Sapan / Gürültülü Örnek)",
                    "adimlar": [
                        "Hızlı çıkarma: 1.10 - 1.00 = 0.10.",
                        "Top doğrudan 10 cent olarak tahmin edildi.",
                    ],
                    "yanit": "10 cent ($0.10)",
                    "tahmin": "0.10",
                },
            ],
            "nilufer_golu": [
                {
                    "strateji": "Geriye Doğru Çıkarım (Backward Chaining)",
                    "adimlar": [
                        "Nilüfer her gün 2 katına çıkıyor (x2).",
                        "48. günde gölün %100'ü kaplıdır.",
                        "Bir gün önce (48 - 1 = 47. gün) alan tam yarısı kadardır (%50).",
                    ],
                    "yanit": "47 gün",
                    "tahmin": "47",
                },
                {
                    "strateji": "Üstel Fonksiyon Analizi",
                    "adimlar": [
                        "Alan formülü: A(t) = A_0 * 2^t.",
                        "A(48) = Tam alan ise, A(48) / 2 = A_0 * 2^(48-1) = A_0 * 2^47.",
                        "Yani 47. günde gölün yarısı kaplanır.",
                    ],
                    "yanit": "47 gün",
                    "tahmin": "47",
                },
                {
                    "strateji": "İleri Simülasyon",
                    "adimlar": [
                        "46. gün: %25 kaplı.",
                        "47. gün: %50 kaplı (Yarı göl).",
                        "48. gün: %100 kaplı (Tam göl).",
                    ],
                    "yanit": "47 gün",
                    "tahmin": "47",
                },
                {
                    "strateji": "Mantıksal Adım Doğrulama",
                    "adimlar": [
                        "Yarıdan tama geçiş tek bir ikiye katlanma adımıdır (+1 gün).",
                        "48 - 1 = 47 gün.",
                    ],
                    "yanit": "47 gün",
                    "tahmin": "47",
                },
                {
                    "strateji": "Hatalı Bölme Yolu (Sapan Örnek)",
                    "adimlar": [
                        "Gölün yarısı için 48 / 2 = 24 gün hesaplandı.",
                    ],
                    "yanit": "24 gün",
                    "tahmin": "24",
                },
            ],
        }

    def ornekle_coklu_yol(
        self,
        soru_anahtari: str,
        soru_metni: str,
        k: int = 5,
        sicaklik: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """K adet bağımsız akıl yürütme yolunu <think> formatında örnekler."""
        yollar = self._strateji_havuzu.get(soru_anahtari, [])
        if not yollar:
            return []

        orneklenen_yollar = []
        for i in range(min(k, len(yollar))):
            item = yollar[i]
            dusunce_blok = "\n".join([f"<step>{adim}</step>" for adim in item["adimlar"]])
            tam_cikti = self.tokenizator.birlestir(dusunce_blok, item["yanit"])
            ayrisim = self.tokenizator.ayristir(tam_cikti)
            token_bilgisi = self.tokenizator.token_dagilimi_hesapla(
                soru_metni, ayrisim["dusunce_metni"], ayrisim["nihai_yanit"]
            )

            orneklenen_yollar.append({
                "yol_no": i + 1,
                "strateji": item["strateji"],
                "sicaklik": sicaklik,
                "dusunce_metni": ayrisim["dusunce_metni"],
                "adimlar": ayrisim["adimlar"],
                "nihai_yanit": ayrisim["nihai_yanit"],
                "tahmin": item["tahmin"],
                "token_bilgisi": token_bilgisi,
                "tam_metin": tam_cikti,
            })

        return orneklenen_yollar
