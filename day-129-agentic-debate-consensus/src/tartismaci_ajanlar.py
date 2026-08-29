"""
Tartışmacı Ajanlar Modülü (Day 129 - Faz 7).
Muhafazakar (Güvenlik Odaklı), Yenilikçi (Hız/Ölçek Odaklı) ve Pragmatik (Maliyet/Operasyon Odaklı) tartışmacı personelleri.
"""

from typing import Dict, Any, List


class TemelTartismaciAjan:
    """Tüm tartışmacı ajanlar için temel sınıf."""

    def __init__(self, ad: str, ekol: str, odak: str):
        self.ad = ad
        self.ekol = ekol
        self.odak = odak

    def arguman_uret(
        self, konu: str, tur_no: int, diger_argumanlar: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MuhafazakarAjan(TemelTartismaciAjan):
    """Sıfır Güven (Zero-Trust), veri gizliliği ve güvenlik risklerini savunan ajan."""

    def __init__(self):
        super().__init__(
            ad="Ajan Alpha (Muhafazakar)",
            ekol="Zero-Trust & Veri Güvenliği",
            odak="Güvenlik açıkları, regülasyon uyumu ve en kötü durum senaryoları",
        )

    def arguman_uret(
        self, konu: str, tur_no: int, diger_argumanlar: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if tur_no == 1:
            sav = "Tüm servisler dış dünyaya kapatılmalı, mTLS ve Air-gapped şifreleme zorunlu kılınmalıdır."
            guven = 0.90
        elif tur_no == 2:
            sav = "Yenilikçi yaklaşımın önerdiği açık API'lar DDoS ve veri sızıntısına açıktır; sıkı Rate-Limiting ve WAF şarttır."
            guven = 0.92
        else:
            sav = "Uzlaşı Noktası: Kritik ödeme ve kimlik mikroservisleri izole edilsin; uç noktalarda API Gateway ile sıkı denetim yapılsın."
            guven = 0.95

        return {
            "ajan": self.ad,
            "tur": tur_no,
            "tez": sav,
            "tercih_edilen_secenek": "Guvenlik_Oncelikli_Mimari",
            "guven_skoru": guven,
            "vurgulanan_riskler": ["Veri Sızıntısı", "Yetkisiz Erişim", "Regülasyon Cezaları"],
        }


class YenilikciAjan(TemelTartismaciAjan):
    """Düşük gecikme, yüksek verim ve kullanıcı deneyimini savunan ajan."""

    def __init__(self):
        super().__init__(
            ad="Ajan Beta (Yenilikçi)",
            ekol="Yüksek Performans & Çeviklik",
            odak="Milisanlik gecikme süresi, küresel ölçeklenebilirlik ve kullanıcı deneyimi",
        )

    def arguman_uret(
        self, konu: str, tur_no: int, diger_argumanlar: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if tur_no == 1:
            sav = "Global Edge CDN, asenkron Event-Driven mimari ve minimum kimlik doğrulama katmanı ile <10ms yanıt hedeflenmelidir."
            guven = 0.88
        elif tur_no == 2:
            sav = "Aşırı güvenlik katmanları p99 gecikmeyi 200ms üzerine çıkarır; donanım hızlandırmalı token doğrulama kullanılmalıdır."
            guven = 0.91
        else:
            sav = "Uzlaşı Noktası: Edge seviyesinde hafif JWT doğrulaması, arka planda asenkron derin güvenlik taraması uygulansın."
            guven = 0.94

        return {
            "ajan": self.ad,
            "tur": tur_no,
            "tez": sav,
            "tercih_edilen_secenek": "Performans_Oncelikli_Mimari",
            "guven_skoru": guven,
            "vurgulanan_riskler": ["Yüksek Gecikme", "Müşteri Kaybı", "Ölçekleme Kilitlenmesi"],
        }


class PragmatikAjan(TemelTartismaciAjan):
    """Maliyet, operasyonel sürdürülebilirlik ve ekip yetkinliğini savunan ajan."""

    def __init__(self):
        super().__init__(
            ad="Ajan Gamma (Pragmatik)",
            ekol="Maliyet-Fayda & Operasyon",
            odak="Altyapı bütçesi, bakım kolaylığı ve aşamalı geçiş (Migration)",
        )

    def arguman_uret(
        self, konu: str, tur_no: int, diger_argumanlar: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if tur_no == 1:
            sav = "Mevcut monolit/hibrit yapı korunmalı, sadece darboğaz oluşturan servisler mikroservise bölünmelidir."
            guven = 0.85
        elif tur_no == 2:
            sav = "Tam izole Air-gapped veya saf küresel Edge mimarileri bütçeyi 3 katına çıkarır; yönetilen bulut servisleri seçilmelidir."
            guven = 0.89
        else:
            sav = "Uzlaşı Noktası: Hibrit yaklaşım; kritik 2 servis izole edilsin, diğerleri mevcut API Gateway üzerinden sunulsun."
            guven = 0.96

        return {
            "ajan": self.ad,
            "tur": tur_no,
            "tez": sav,
            "tercih_edilen_secenek": "Hibrit_Dengeli_Mimari",
            "guven_skoru": guven,
            "vurgulanan_riskler": ["Bütçe Aşımı", "Operasyonel Karmaşıklık", "Uzun Teslimat Süresi"],
        }
