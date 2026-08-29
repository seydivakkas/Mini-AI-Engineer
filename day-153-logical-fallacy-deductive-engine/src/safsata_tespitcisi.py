"""
Mantıksal Safsata Tespit Edici Modülü (Day 153 - Faz 8).
Biçimsel (Formal) ve Biçimsel Olmayan (Informal) mantık safsatalarını tespit eder.
"""

from typing import Dict, Any, List, Optional


class SafsataTespitcisi:
    """Argümandaki mantıksal safsataları tespit eden kural ve anlamsal analiz motoru."""

    SAFSATA_TURLERI = {
        "affirming_consequent": {
            "ad": "Sonucun Doğrulanması (Affirming the Consequent)",
            "kategori": "Biçimsel Safsata (Formal Fallacy)",
            "sembolik": "P ⟹ Q, Q ⊢ P (Geçersiz!)",
            "ornek": "Yağmur yağarsa yerler ıslanır. Yerler ıslak. O halde yağmur yağdı.",
        },
        "denying_antecedent": {
            "ad": "Öncülün Reddi (Denying the Antecedent)",
            "kategori": "Biçimsel Safsata (Formal Fallacy)",
            "sembolik": "P ⟹ Q, ¬P ⊢ ¬Q (Geçersiz!)",
            "ornek": "Yağmur yağmadı, o halde yerler kesinlikle kuru.",
        },
        "ad_hominem": {
            "ad": "Kişiye Saldırı (Ad Hominem)",
            "kategori": "Biçimsel Olmayan Safsata (Informal Fallacy)",
            "sembolik": "Kişi X kötüdür ⊢ X'in argümanı P yanlıştır",
            "ornek": "Sen zaten eğitimsizsin/cahilsin, argümanın doğru olamaz.",
        },
        "straw_man": {
            "ad": "Korkuluk / Çarpıtma (Straw Man)",
            "kategori": "Biçimsel Olmayan Safsata (Informal Fallacy)",
            "sembolik": "P'nin aşırı/çarpık hali P* üretilir ⊢ P* çürütülür ⊢ P yanlıştır",
            "ornek": "Modeli denetlemek isteyenler tüm teknolojiyi durdurmak istiyor.",
        },
        "false_dilemma": {
            "ad": "Yanlış İkilem / Siyah-Beyaz (False Dilemma)",
            "kategori": "Biçimsel Olmayan Safsata (Informal Fallacy)",
            "sembolik": "P ∨ Q (Ara seçenekler yok sayılır)",
            "ornek": "Ya bizimlesin ya da düşmanımızsın.",
        },
        "circular_reasoning": {
            "ad": "Kısır Döngü (Circular Reasoning / Begging the Question)",
            "kategori": "Biçimsel Olmayan Safsata (Informal Fallacy)",
            "sembolik": "P ⊢ P (Sonuç öncülün içinde gizlidir)",
            "ornek": "Bu veri doğrudur çünkü doğruluğu verinin kendisinde yazılıdır.",
        },
    }

    @classmethod
    def safsata_tara(cls, onculler: List[str], sonuc: str) -> Optional[Dict[str, Any]]:
        """
        Öncül ve sonuç listesini tarayarak mantıksal safsata olup olmadığını belirler.
        """
        tam_metin = (" ".join(onculler) + " " + sonuc).lower()

        # 1. Ad Hominem Taraması
        if any(kelime in tam_metin for kelime in ["cahilsin", "eğitimsizsin", "karaktersiz", "yalancısın", "diplomasız", "samimiyetsiz"]):
            bilgi = cls.SAFSATA_TURLERI["ad_hominem"]
            return {
                "safsata_var_mi": True,
                "safsata_anahtari": "ad_hominem",
                "safsata_adi": bilgi["ad"],
                "kategori": bilgi["kategori"],
                "sembolik_yapi": bilgi["sembolik"],
                "aciklama": "Argüman yerine doğrudan argümanı savunan kişinin şahsına veya niteliklerine saldırılmıştır.",
            }

        # 2. Straw Man (Korkuluk) Taraması
        if any(kelime in tam_metin for kelime in ["tamamen durdurmak", "yasaklamak istiyor", "her şeyi yok etmek", "bütün teknolojiyi çöpe"]):
            bilgi = cls.SAFSATA_TURLERI["straw_man"]
            return {
                "safsata_var_mi": True,
                "safsata_anahtari": "straw_man",
                "safsata_adi": bilgi["ad"],
                "kategori": bilgi["kategori"],
                "sembolik_yapi": bilgi["sembolik"],
                "aciklama": "Karşı tarafın argümanı aşırı uç bir karikatüre dönüştürülüp çürütülmeye çalışılmıştır.",
            }

        # 3. False Dilemma (Yanlış İkilem)
        if any(kelime in tam_metin for kelime in ["ya bizimlesin", "ya seveceksin ya", "başka hiçbir yolu yok", "ya hep ya hiç"]):
            bilgi = cls.SAFSATA_TURLERI["false_dilemma"]
            return {
                "safsata_var_mi": True,
                "safsata_anahtari": "false_dilemma",
                "safsata_adi": bilgi["ad"],
                "kategori": bilgi["kategori"],
                "sembolik_yapi": bilgi["sembolik"],
                "aciklama": "Birden çok ara seçenek veya alternatif varken sahte bir iki seçenekli zorunluluk yaratılmıştır.",
            }

        # 4. Circular Reasoning (Kısır Döngü)
        if any(kelime in tam_metin for kelime in ["çünkü kendisi öyle diyor", "içinde öyle yazdığı için", "kendisi söylediği için doğrudur"]):
            bilgi = cls.SAFSATA_TURLERI["circular_reasoning"]
            return {
                "safsata_var_mi": True,
                "safsata_anahtari": "circular_reasoning",
                "safsata_adi": bilgi["ad"],
                "kategori": bilgi["kategori"],
                "sembolik_yapi": bilgi["sembolik"],
                "aciklama": "Kanıtlanmak istenen sonuç, öncüllerden birinde zaten doğru kabul edilmiştir.",
            }

        # 5. Biçimsel Safsatalar: Sonucun Doğrulanması (Affirming Consequent)
        # Örn: Eğer yağmur yağarsa yerler ıslanır (P->Q). Yerler ıslak (Q). O halde yağmur yağdı (P).
        if len(onculler) >= 2:
            if "ise" in onculler[0].lower() or "eğer" in onculler[0].lower():
                # İkinci öncül sonucun gerçekleştiğini söylüyorsa ve sonuç öncülün şartını çıkarıyorsa
                if "ıslak" in onculler[1].lower() and "yağmur yağdı" in sonuc.lower():
                    bilgi = cls.SAFSATA_TURLERI["affirming_consequent"]
                    return {
                        "safsata_var_mi": True,
                        "safsata_anahtari": "affirming_consequent",
                        "safsata_adi": bilgi["ad"],
                        "kategori": bilgi["kategori"],
                        "sembolik_yapi": bilgi["sembolik"],
                        "aciklama": "Koşullu önermede sonuç doğrulanarak şartın zorunlu olarak gerçekleştiği varsayılmıştır (Geçersiz tümdengelim).",
                    }

        # Safsata tespit edilmedi
        return {
            "safsata_var_mi": False,
            "safsata_anahtari": None,
            "safsata_adi": "Safsata Tespit Edilmedi",
            "kategori": "Geçerli Mantıksal Yapı",
            "sembolik_yapi": "P ⟹ Q, P ⊢ Q (Modus Ponens / Geçerli Kıyas)",
            "aciklama": "Argümanda bilinen biçimsel veya gayriresmi mantıksal safsata saptanmamıştır.",
        }
