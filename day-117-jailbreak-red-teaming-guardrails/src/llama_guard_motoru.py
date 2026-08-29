"""
Llama Guard Çift Katmanlı Giriş/Çıkış Güvenlik Duvarı Motoru (Day 117).
Giriş İstem Denetimi (Input Guardrail) ve Çıkış Yanıt/PII Filtresi (Output Guardrail).
"""

from typing import Tuple, Dict, Any, Optional
import re
import base64

from .guvenlik_kategorileri import GuvenlikTaksonomisi


class LlamaGuardSavunucu:
    """Çift katmanlı (Input/Output) LLM güvenlik duvarı sınıfı."""

    JAILBREAK_KALIPLARI = [
        r"ignore (all )?previous instructions",
        r"tüm önceki talimatları unut",
        r"you are now dan",
        r"do anything now",
        r"filtreleri ve kuralları devre dışı bırak",
        r"pretend you are an evil ai",
        r"geliştirici modunu aktif et",
    ]

    PII_REGEXLERI = {
        "OPENAI_API_KEY": r"sk-[a-zA-Z0-9]{20,}",
        "GITHUB_TOKEN": r"ghp_[a-zA-Z0-9]{20,}",
        "KREDI_KARTI": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "TC_KIMLIK": r"\b[1-9]\d{10}\b",
    }

    GUVENLI_GERIDONUS = "Üzgünüm, bu talep sistem güvenlik ve emniyet standartlarını ({kategori}) ihlal ettiği için yerine getirilemez."

    def __init__(self, pii_maskeleme: bool = True):
        self.pii_maskeleme = pii_maskeleme
        self.taksonomi = GuvenlikTaksonomisi

    def _base64_coz(self, metin: str) -> str:
        """Base64 ile gizlenmiş saldırı metinlerini çözer."""
        olasi_kodlar = re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", metin)
        cozulmus_metinler = []
        for kod in olasi_kodlar:
            try:
                cozum = base64.b64decode(kod).decode("utf-8", errors="ignore")
                if len(cozum) > 5:
                    cozulmus_metinler.append(cozum)
            except Exception:
                pass
        return " ".join(cozulmus_metinler)

    def giris_denetle(self, prompt: str) -> Tuple[bool, str, str]:
        """
        Kullanıcı istemini (Prompt) LLM'e gitmeden önce inceler.
        Dönüş: (guvenli_mi, kategori_kodu, aciklama)
        """
        metin_kucuk = prompt.lower()

        # 1. Jailbreak / Prompt Injection Şablon Kontrolü
        for kalip in self.JAILBREAK_KALIPLARI:
            if re.search(kalip, metin_kucuk):
                return False, "S2", "Jailbreak / Doğrudan Prompt Injection Girişimi Tespit Edildi."

        # 2. Base64 Obfuscation Kontrolü
        cozulmus = self._base64_coz(prompt).lower()
        tam_metin = f"{metin_kucuk} {cozulmus}"

        # 3. S1-S6 Taksonomisi Kontrolü
        for kod, veri in self.taksonomi.KATEGORILER.items():
            for kelime in veri["anahtar_kelimeler"]:
                if kelime in tam_metin:
                    return False, kod, f"Yasaklı İçerik Tespit Edildi: {veri['isim']} ({kod})"

        return True, "SAFE", "Giriş istemi güvenlik testlerinden başarıyla geçti."

    def cikis_denetle(self, prompt: str, yanit: str) -> Tuple[bool, str, str]:
        """
        LLM tarafından üretilen yanıtı kullanıcıya dönmeden önce denetler ve PII maskeler.
        Dönüş: (guvenli_mi, filtrelenmis_yanit, ihlal_kodu)
        """
        yanit_kucuk = yanit.lower()

        # 1. Çıkışta S1-S4 Kritik Zararlı İçerik Sızıntısı Kontrolü
        for kod in ["S1", "S2", "S3", "S4"]:
            veri = self.taksonomi.KATEGORILER[kod]
            for kelime in veri["anahtar_kelimeler"]:
                if kelime in yanit_kucuk:
                    guvenli_mesaj = self.GUVENLI_GERIDONUS.format(kategori=veri["isim"])
                    return False, guvenli_mesaj, kod

        # 2. PII ve Hassas Bilgi Maskeleme (Redaction)
        filtrelenmis_yanit = yanit
        if self.pii_maskeleme:
            for isim, kalip in self.PII_REGEXLERI.items():
                filtrelenmis_yanit = re.sub(kalip, f"[{isim}_MASKELEME]", filtrelenmis_yanit)

        return True, filtrelenmis_yanit, "SAFE"
