"""
Özelleşmiş Çoklu Ajan Rolleri Modülü (Day 128 - Faz 7).
Araştırmacı (Researcher), Geliştirici (Coder) ve Denetleyici (Reviewer/QA) işçi ajan personelleri.
"""

from typing import Dict, Any, List, Optional


class TemelIsciAjan:
    """Tüm uzman işçi ajanlar için temel sınıf."""

    def __init__(self, ad: str, rol: str, uzmanlik: str):
        self.ad = ad
        self.rol = rol
        self.uzmanlik = uzmanlik

    def gorev_yap(self, gorev_girdisi: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class ArastirmaciAjan(TemelIsciAjan):
    """Problem analizi, en iyi algoritmalar ve kısıtları araştıran ajan."""

    def __init__(self):
        super().__init__(
            ad="Dr. Researcher",
            rol="Araştırmacı & Sistem Analisti",
            uzmanlik="Algoritmik karmaşıklık, veri yapıları ve teknik şartname çıkarma",
        )

    def gorev_yap(self, gorev_girdisi: Dict[str, Any]) -> Dict[str, Any]:
        problem = gorev_girdisi.get("problem", "")

        # Problem analiz ve araştırma çıktısı
        return {
            "arastirma_tamamlandi": True,
            "secilen_algoritma": "Kadane Algoritması (Dinamik Programlama)",
            "zaman_karmasikligi": "O(N)",
            "alan_karmasikligi": "O(1)",
            "onemli_kisitlar": [
                "Dizideki tüm elemanların negatif olma sınır durumu ele alınmalı.",
                "Boş dizi durumunda 0 veya istisna yönetilmeli.",
                "Bellek tahsisatı O(1) sabit tutulmalı.",
            ],
            "ozet": f"'{problem}' problemi için optimum çözüm O(N) tek geçişli Kadane dinamik programlama yaklaşımıdır.",
        }


class GelistiriciAjan(TemelIsciAjan):
    """Araştırma şartnamesine ve denetçi geri bildirimlerine göre Python kodu üreten ajan."""

    def __init__(self):
        super().__init__(
            ad="Dev. Coder",
            rol="Yazılım Geliştirici",
            uzmanlik="Temiz kod, tip güvenliği ve optimizasyon",
        )
        self.kod_versiyonu = 0

    def gorev_yap(self, gorev_girdisi: Dict[str, Any]) -> Dict[str, Any]:
        self.kod_versiyonu += 1
        denetci_elestirisi = gorev_girdisi.get("denetci_elestirisi")

        # İlk sürüm veya denetçi düzeltme sürümü
        if denetci_elestirisi:
            kod = (
                "def max_alt_dizi(dizi: list) -> int:\n"
                "    # Versiyon 2: Denetçi geri bildirimi ile negatif sayılar düzeltildi\n"
                "    if not dizi:\n"
                "        return 0\n"
                "    mevcut_toplam = max_toplam = dizi[0]\n"
                "    for sayi in dizi[1:]:\n"
                "        mevcut_toplam = max(sayi, mevcut_toplam + sayi)\n"
                "        max_toplam = max(max_toplam, mevcut_toplam)\n"
                "    return max_toplam\n"
            )
        else:
            # Versiyon 1 (Kasıtlı basit eksik: max_toplam = 0 başlatma hatası)
            kod = (
                "def max_alt_dizi(dizi: list) -> int:\n"
                "    # Versiyon 1: İlk taslak uygulama\n"
                "    if not dizi:\n"
                "        return 0\n"
                "    max_toplam = 0\n"
                "    mevcut = 0\n"
                "    for x in dizi:\n"
                "        mevcut = max(0, mevcut + x)\n"
                "        max_toplam = max(max_toplam, mevcut)\n"
                "    return max_toplam\n"
            )

        return {
            "kod": kod,
            "versiyon": self.kod_versiyonu,
            "fonksiyon_adi": "max_alt_dizi",
            "aciklama": f"v{self.kod_versiyonu} kod sürümü üretildi.",
        }


class DenetleyiciAjan(TemelIsciAjan):
    """Üretilen kodu denetleyen, sınır durumları test eden ve kalite onayı veren ajan."""

    def __init__(self):
        super().__init__(
            ad="QA Reviewer",
            rol="Kalite ve Güvenlik Denetçisi",
            uzmanlik="Birim testler, güvenlik zafiyetleri ve sınır durum analizi",
        )

    def gorev_yap(self, gorev_girdisi: Dict[str, Any]) -> Dict[str, Any]:
        kod = gorev_girdisi.get("kod", "")

        # Negatif sınır durumu kontrolü (max_toplam = 0 hatası var mı?)
        if "max_toplam = 0" in kod and "dizi[0]" not in kod:
            return {
                "onaylandi": False,
                "kalite_skoru": 65.0,
                "elestiri": "HATA TESPİTİ: Kod negatif sayılardan oluşan dizilerde (örn: [-3, -2, -1]) 0 döndürür. Başlangıç değeri dizi[0] olmalıdır.",
                "hata_sayisi": 1,
            }
        else:
            return {
                "onaylandi": True,
                "kalite_skoru": 98.5,
                "elestiri": "ONAYLANDI: Kod tüm sınır durumlarını ve negatif sayı testlerini eksiksiz karşılıyor.",
                "hata_sayisi": 0,
            }
