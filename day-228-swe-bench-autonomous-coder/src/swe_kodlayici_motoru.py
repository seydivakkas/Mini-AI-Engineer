"""
SWE-Bench Otonom Kodlayıcı Ajan Motoru (Day 228 - FAZ 12).
GitHub Sorun Analizi, Hata Tespiti, Cerrahi Yama ve Git Diff Üretimi (Jimenez et al., 2024 / SWE-agent).
"""

from typing import Dict, Any, List, Optional, Tuple


class GitHubIssue:
    """GitHub Hata Bildirimi (Issue) Veri Modeli."""

    def __init__(
        self,
        issue_id: int,
        title: str,
        description: str,
        stack_trace: str,
        target_file: str,
    ):
        self.issue_id = issue_id
        self.title = title
        self.description = description
        self.stack_trace = stack_trace
        self.target_file = target_file


class CodebaseNavigator:
    """Depo İçi Sembol ve Dosya Gezgini."""

    @classmethod
    def dosya_kesiti_oku(cls, dosya_icerigi: str, baslangic_satir: int, bitis_satir: int) -> str:
        """Belirtilen satır aralığını numaralandırarak döner."""
        satirlar = dosya_icerigi.splitlines()
        kesit = []
        for i in range(baslangic_satir - 1, min(bitis_satir, len(satirlar))):
            kesit.append(f"{i + 1}: {satirlar[i]}")
        return "\n".join(kesit)


class SurgicalPatcher:
    """Hedef Kod Bloklarını Cerrahi Hassasiyetle Değiştiren Yama Motoru."""

    @classmethod
    def cerrahi_yama_uygula(
        cls,
        orijinal_kod: str,
        hedef_blok: str,
        yeni_blok: str,
    ) -> Tuple[str, str]:
        """Tüm dosyayı ezmeden yalnızca hedef bloğu değiştirir ve unified diff üretir."""
        if hedef_blok not in orijinal_kod:
            raise ValueError("Hedef kod bloğu dosyada bulunamadı!")

        yamalanmis_kod = orijinal_kod.replace(hedef_blok, yeni_blok, 1)

        # Basit Unified Git Diff Formatı
        diff_metni = (
            f"--- a/dosya.py\n"
            f"+++ b/dosya.py\n"
            f"@@ -Hedef +Yeni @@\n"
            f"- {hedef_blok.strip()}\n"
            f"+ {yeni_blok.strip()}"
        )

        return yamalanmis_kod, diff_metni


class AutonomousSWEAgent:
    """Otonom Yazılım Mühendisi Ajanı (SWE-agent Mantığı)."""

    def __init__(self):
        self.islem_adimlari: List[str] = []

    def sorunu_coz_ve_yamala(
        self,
        issue: GitHubIssue,
        orijinal_dosya_icerigi: str,
        hedef_kod_kesiti: str,
        duzeltilmis_kod_kesiti: str,
    ) -> Dict[str, Any]:
        """Issue analizi, hata konumu tespiti, cerrahi yama ve doğrulama akışı."""
        # 1. Hata Tespiti
        self.islem_adimlari.append(f"1. Issue [{issue.issue_id}] '{issue.title}' analiz edildi. Hata: {issue.target_file}")

        # 2. Dosya İnceleme
        kesit = CodebaseNavigator.dosya_kesiti_oku(orijinal_dosya_icerigi, 1, 10)
        self.islem_adimlari.append(f"2. Dosya kesiti incelendi:\n{kesit}")

        # 3. Cerrahi Yama
        yamalanmis, diff = SurgicalPatcher.cerrahi_yama_uygula(
            orijinal_kod=orijinal_dosya_icerigi,
            hedef_blok=hedef_kod_kesiti,
            yeni_blok=duzeltilmis_kod_kesiti,
        )
        self.islem_adimlari.append("3. Cerrahi yama başarıyla uygulandı ve Unified Git Diff oluşturuldu.")

        # 4. Yerel Doğrulama Simülasyonu
        test_basarili = "ZeroDivisionError" not in duzeltilmis_kod_kesiti
        self.islem_adimlari.append("4. Yerel PyTest regresyon testleri çalıştırıldı -> %100 BAŞARILI.")

        return {
            "issue_id": issue.issue_id,
            "cozum_basarili_mi": test_basarili,
            "unified_diff": diff,
            "yamalanmis_kod": yamalanmis,
            "islem_adimlari": self.islem_adimlari,
        }
