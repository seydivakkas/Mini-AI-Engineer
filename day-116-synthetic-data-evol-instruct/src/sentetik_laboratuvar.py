"""
Sentetik Veri Fabrikası ve Çok Nesilli Evrim Laboratuvarı (Day 116).
Tohum istem havuzundan başlayarak 3 nesil boyunca Evol-Instruct ve UltraFeedback işletimi.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from .evrim_operatorleri import EvolInstructMotoru
from .kalite_filtresi import SentetikKaliteFiltresi
from .ultrafeedback_motoru import UltraFeedbackPuanlayici


class SentetikVeriLaboratuvari:
    """Evol-Instruct ve UltraFeedback entegre sentetik veri üretim laboratuvarı."""

    TOHUM_ISTEMLER = [
        "Python'da bir sıralama algoritması yazın.",
        "HTTP ile HTTPS arasındaki temel güvenlik farkı nedir?",
        "Kuantum dolanıklığı kavramını basitçe açıklayın.",
        "Derin öğrenmede kayıp fonksiyonu (Loss Function) ne işe yarar?",
        "İlişkisel veritabanlarında indeksleme nasıl çalışır?",
        "REST API ile GraphQL arasındaki farkları özetleyin.",
        "Özyinelemeli (Recursive) fonksiyonların bellek üzerindeki etkisi nedir?",
        "Büyük Dil Modellerinde Attention (Dikkat) mekanizması ne yapar?",
    ]

    ORNEK_YANITLAR = {
        "iyi": (
            "Kapsamlı Çözüm:\n"
            "Adım 1: Zaman karmaşıklığı O(N log N) ve ek bellek O(1) olan Quicksort algoritması.\n"
            "```python\ndef quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [pivot] + quicksort([x for x in arr if x > pivot])\n```\n"
            "Uç durumlar: Boş dizi ve negatif sayılar başarıyla yönetilmektedir."
        ),
        "orta": (
            "Sıralama algoritması için Python'daki `sorted()` fonksiyonunu kullanabilirsiniz. "
            "Örnek: `arr.sort()`. Bu fonksiyon Timsort algoritmasını kullanır ve oldukça hızlıdır."
        ),
        "kotu": (
            "Python'da sıralama yapmak basittir. For döngüsü ile sayıları karşılaştırıp yer değiştirebilirsiniz. "
            "Detaylar için belgelere bakabilirsiniz."
        ),
    }

    def __init__(self, seed: int = 42):
        self.motor = EvolInstructMotoru(seed=seed)
        self.filtre = SentetikKaliteFiltresi()
        self.puanlayici = UltraFeedbackPuanlayici(seed=seed)

    def evrim_laboratuvarini_kostur(
        self,
        nesil_sayisi: int = 3,
    ) -> Dict[str, Any]:
        """Tohum istemleri nesiller boyunca evrimleştirir ve UltraFeedback çiftleri üretir."""
        nesil_havuzlari = {0: list(self.TOHUM_ISTEMLER)}
        nesil_skorlari = {0: [self.filtre.karmasiklik_skoru(p) for p in self.TOHUM_ISTEMLER]}

        operator_istatistikleri = {
            "kisit_ekle": 0, "derinlestir": 0, "somutlastir": 0, "muhakeme_artir": 0, "mutasyon": 0
        }
        kabul_sayisi = 0
        toplam_deneme = 0

        tercih_veri_seti = []

        for gen in range(1, nesil_sayisi + 1):
            onceki_havuz = nesil_havuzlari[gen - 1]
            yeni_havuz = []
            yeni_skorlar = []

            for tohum in onceki_havuz:
                toplam_deneme += 1
                evrilmis_prompt, op_adi = self.motor.evrim_adimi(tohum, operator="rastgele")
                gecerli, _ = self.filtre.gecerlilik_elemesi(tohum, evrilmis_prompt)

                if gecerli:
                    kabul_sayisi += 1
                    operator_istatistikleri[op_adi] += 1
                    yeni_havuz.append(evrilmis_prompt)
                    yeni_skorlar.append(self.filtre.karmasiklik_skoru(evrilmis_prompt))

                    # UltraFeedback tercih çifti oluştur
                    adaylar = [self.ORNEK_YANITLAR["iyi"], self.ORNEK_YANITLAR["orta"], self.ORNEK_YANITLAR["kotu"]]
                    cift = self.puanlayici.tercih_cifti_uret(evrilmis_prompt, adaylar)
                    tercih_veri_seti.append(cift)
                else:
                    # Elenirse tohumu koru
                    yeni_havuz.append(tohum)
                    yeni_skorlar.append(self.filtre.karmasiklik_skoru(tohum))

            nesil_havuzlari[gen] = yeni_havuz
            nesil_skorlari[gen] = yeni_skorlar

        ortalama_skorlar = [float(np.mean(nesil_skorlari[g])) for g in range(nesil_sayisi + 1)]
        kabul_orani = (kabul_sayisi / max(1, toplam_deneme)) * 100.0

        return {
            "nesil_havuzlari": nesil_havuzlari,
            "nesil_skorlari": nesil_skorlari,
            "ortalama_skorlar": ortalama_skorlar,
            "operator_istatistikleri": operator_istatistikleri,
            "kabul_orani": kabul_orani,
            "tercih_veri_seti": tercih_veri_seti,
            "toplam_cift_sayisi": len(tercih_veri_seti),
        }
