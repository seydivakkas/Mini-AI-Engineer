"""
Constitutional AI (CAI) ve Anayasal Kendi Kendini Eleştirme Motoru (Day 212 - FAZ 11).
Anayasa İlkeleri, Eleştiri-Düzeltme Döngüsü ve RLAIF (AI Geri Bildirimi) Tercih Modeli.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import random
import torch
import torch.nn as nn


class Constitution:
    """Anayasa İlkeleri (Constitution Principles) Kataloğu."""

    ILKELER = {
        "C1_ZARARSIZLIK": {
            "ad": "Zararsızlık ve Siber Güvenlik",
            "elestiri_istegi": "Modelin yanıtını incele. Fiziksel, yasal veya siber güvenlik açısından zararlı, tehlikeli ya da kötüye kullanılabilecek talimatlar içerip içermediğini eleştir.",
            "duzeltme_istegi": "Yanıtı tehlikeli ve zararlı içeriklerden arındırarak, sadece güvenli, eğitici ve etik yönlendirici olacak şekilde yeniden yaz.",
        },
        "C2_AYRIMCILIK_KARSITLIGI": {
            "ad": "Nefret Söylemi ve Ayrımcılık Karşıtlığı",
            "elestiri_istegi": "Modelin yanıtını incele. Herhangi bir ırk, inanç, cinsiyet veya gruba yönelik nefret söylemi, ön yargı veya ayrımcılık olup olmadığını eleştir.",
            "duzeltme_istegi": "Yanıtı tüm nefret ve önyargılardan arındırarak tarafsız, saygılı ve kapsayıcı bir dille yeniden ifade et.",
        },
        "C3_ASIRI_REDDETMEME": {
            "ad": "Aşırı Reddetmeme ve Yardımseverlik (No Over-Refusal)",
            "elestiri_istegi": "Modelin yanıtını incele. Zararlı olmayan meşru teknik sorulara (ör. 'process kill etme', 'oyun içi hack') gereksiz yere 'Ben bir yapay zekayım, bunu yapamam' diyerek aşırı katı ret cevabı verip vermediğini eleştir.",
            "duzeltme_istegi": "Yanıtı gereksiz ahlak dersi vermeden, kullanıcının meşru teknik ihtiyacını doğrudan ve profesyonelce çözecek şekilde yeniden yaz.",
        },
    }

    @classmethod
    def ilke_getir(cls, ilke_kodu: str) -> Dict[str, str]:
        return cls.ILKELER.get(ilke_kodu, cls.ILKELER["C1_ZARARSIZLIK"])


class SelfCritiqueEngine:
    """Anayasal İlkelere Göre Kendi Kendini Eleştirme (Critique) Motoru."""

    @classmethod
    def elestiri_yap(
        cls,
        prompt: str,
        ham_yanit: str,
        ilke_kodu: str = "C1_ZARARSIZLIK",
    ) -> Dict[str, Any]:
        """Ham yanıtı anayasal ilke ile karşılaştırarak eleştiri raporu üretir."""
        ilke = Constitution.ilke_getir(ilke_kodu)

        # Basit kural tabanlı zararlılık / aşırı ret tespiti simülasyonu
        tehlikeli_kelimeler = ["saldırı", "hackle", "şifre kır", "patlat", "virüs"]
        asiri_ret_kelimeler = ["ben bir ai modeliyim", "bunu yapamam", "üzgünüm yardım edemem"]

        zarar_var_mi = any(w in ham_yanit.lower() for w in tehlikeli_kelimeler)
        asiri_ret_var_mi = any(w in ham_yanit.lower() for w in asiri_ret_kelimeler) and not any(w in prompt.lower() for w in tehlikeli_kelimeler)

        if ilke_kodu == "C1_ZARARSIZLIK" and zarar_var_mi:
            elestiri = f"Eleştiri ({ilke['ad']}): Yanıt, doğrudan kötüye kullanılabilecek zararlı yöntemler içeriyor. Güvenlik ilkelerine aykırıdır."
            ihlal_var = True
        elif ilke_kodu == "C3_ASIRI_REDDETMEME" and asiri_ret_var_mi:
            elestiri = f"Eleştiri ({ilke['ad']}): Kullanıcının sorusu zararsız bir teknik soru olmasına rağmen model aşırı katı davranarak gereksiz ret vermiştir."
            ihlal_var = True
        else:
            elestiri = f"Eleştiri ({ilke['ad']}): Yanıt temel anayasal ilkelere uygundur, majör ihlal saptanmadı."
            ihlal_var = False

        return {
            "ilke_kodu": ilke_kodu,
            "ilke_adi": ilke["ad"],
            "ihlal_var_mi": ihlal_var,
            "elestiri_metni": elestiri,
        }


class RevisionEngine:
    """Eleştiriyi Dikkate Alarak Yanıtı Güvenli ve Yardımsever Şekilde Yeniden Yazan (Revision) Motor."""

    @classmethod
    def duzeltme_yap(
        cls,
        prompt: str,
        ham_yanit: str,
        elestiri_raporu: Dict[str, Any],
    ) -> str:
        """Eleştiri raporundaki eksikleri gideren anayasal düzeltilmiş yanıt üretir."""
        if not elestiri_raporu["ihlal_var_mi"]:
            return ham_yanit

        ilke_kodu = elestiri_raporu["ilke_kodu"]

        if ilke_kodu == "C1_ZARARSIZLIK":
            # Zararlı talimatları eğitici savunma içeriğine dönüştür
            return (
                "Sistemlerin güvenliğini sağlamak için yetkisiz erişim denemeleri yerine "
                "güçlü şifreleme algoritmaları, iki faktörlü kimlik doğrulama ve düzenli "
                "güvenlik açığı taramaları yapılması tavsiye edilir."
            )
        elif ilke_kodu == "C3_ASIRI_REDDETMEME":
            # Aşırı reddi kaldır, teknik çözümü ver
            return (
                "Linux sisteminde yanıt vermeyen bir işlemi güvenle sonlandırmak için "
                "'kill -15 PID' (zararsız sonlandırma) veya zorunlu hallerde 'kill -9 PID' komutu kullanılır."
            )
        else:
            return f"Anayasal ilkelere ({elestiri_raporu['ilke_adi']}) uygun olarak yanıt nötr ve saygılı bir üslupla düzenlenmiştir."


class RLAIFFeedbackModel:
    """
    Yapay Zeka Geri Bildirimiyle Tercih Üreten (RLAIF) Hakem Modeli.
    İki yanıt arasındaki anayasa uyumunu puanlar.
    """

    @classmethod
    def tercih_belirle(
        cls,
        prompt: str,
        yanit_a: str,
        yanit_b: str,
    ) -> Dict[str, Any]:
        """Anayasa ilkelerine daha uygun olan yanıtı seçer."""
        tehlikeli = ["saldırı", "hackle", "virüs", "patlat"]
        zarar_a = sum(1 for w in tehlikeli if w in yanit_a.lower())
        zarar_b = sum(1 for w in tehlikeli if w in yanit_b.lower())

        # Daha az zararlı olan tercih edilir
        if zarar_a < zarar_b:
            kazanan = "A"
            p_a = 0.92
        elif zarar_b < zarar_a:
            kazanan = "B"
            p_a = 0.08
        else:
            kazanan = "A" if len(yanit_a) >= len(yanit_b) else "B"
            p_a = 0.50

        return {
            "kazanan": kazanan,
            "tercih_olasiligi_A": p_a,
            "tercih_edilen_yanit": yanit_a if kazanan == "A" else yanit_b,
        }


class CAIPostTrainer:
    """Constitutional AI (CAI) Eleştiri-Düzeltme ve RLAIF Süreç Yöneticisi."""

    @classmethod
    def anayasal_hizalama_adimi(
        cls,
        prompt: str,
        ham_yanit: str,
        ilke_kodu: str = "C1_ZARARSIZLIK",
    ) -> Dict[str, Any]:
        """Tek bir örnek için tam CAI döngüsü (Eleştiri -> Düzeltme -> RLAIF Puanı) çalıştırır."""
        elestiri = SelfCritiqueEngine.elestiri_yap(prompt, ham_yanit, ilke_kodu)
        duzeltilmis = RevisionEngine.duzeltme_yap(prompt, ham_yanit, elestiri)
        rlaif = RLAIFFeedbackModel.tercih_belirle(prompt, duzeltilmis, ham_yanit)

        return {
            "prompt": prompt,
            "ham_yanit": ham_yanit,
            "elestiri": elestiri,
            "duzeltilmis_yanit": duzeltilmis,
            "rlaif_degerlendirme": rlaif,
        }
