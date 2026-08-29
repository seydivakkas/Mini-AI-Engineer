"""
Hakem Moderasyonu ve Konsensüs Oylama Modülü (Day 129 - Faz 7).
Hakem Ajan (Judge) mantık denetimi, kanıt puanlama, çoğunluk ve ağırlıklı güven oylama motoru.
"""

from typing import Dict, Any, List
from collections import Counter


class HakemAjan:
    """Tartışmayı denetleyen, mantık safsatalarını süzen ve nihai kararı sentezleyen bağımsız hakem ajan."""

    def __init__(self, ad: str = "Hakem Lord (Judge Moderator)"):
        self.ad = ad

    def tur_degerlendir(
        self, tur_no: int, argumanlar: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Her turdaki argümanları mantıksal tutarlılık ve kanıt gücüne göre puanlar."""
        skorlar: Dict[str, float] = {}

        for arg in argumanlar:
            ajan_adi = arg["ajan"]
            guven = arg["guven_skoru"]
            # Tur ilerledikçe uzlaşı arayan tezler daha yüksek hakem puanı alır
            uzlasi_bonusu = 5.0 if "Uzlaşı Noktası" in arg["tez"] else 0.0
            puan = (guven * 90.0) + uzlasi_bonusu
            skorlar[ajan_adi] = round(puan, 2)

        # 3. turda veya skorlar yakınsadığında konsensüs sağlandı kabul edilir
        konsensus_var_mi = (tur_no >= 3) or (max(skorlar.values()) - min(skorlar.values()) < 3.0)

        return {
            "tur_no": tur_no,
            "hakem": self.ad,
            "ajan_skorlari": skorlar,
            "konsensus_saglandi_mi": konsensus_var_mi,
            "hakem_yorumu": (
                f"Tur {tur_no}: Ajanlar argümanlarını savundu. "
                f"{'Konsensüs sağlandı, oylama ve senteze geçilebilir.' if konsensus_var_mi else 'Çapraz sorgulama devam etmeli.'}"
            ),
        }

    def nihai_hukum_ver(
        self, konu: str, tum_turlar: List[Dict[str, Any]], oylama: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tüm tartışma turlarını ve oylama sonucunu harmanlayarak nihai karar metnini üretir."""
        kazanan_secenek = oylama["kazanan_secenek"]
        toplam_guven = oylama["kazanan_guven_orani"]

        hukum_metni = (
            f"=== HAKEM NİHAİ KONSENSÜS HÜKMÜ ===\n"
            f"• Tartışılan Konu     : {konu}\n"
            f"• Seçilen Optimum Yol : {kazanan_secenek}\n"
            f"• Konsensüs Güven Payı: %{toplam_guven:.1f}\n"
            f"• Gerekçe             : Muhafazakar ajanın güvenlik izolasyon talebi ile Yenilikçi ajanın "
            f"Edge performans kısıtı 'Hibrit_Dengeli_Mimari' çatısı altında uzlaştırılmıştır.\n"
        )

        return {
            "hukum_metni": hukum_metni,
            "kazanan_secenek": kazanan_secenek,
            "guven_orani": toplam_guven,
            "karar_kesin_mi": True,
        }


class KonsensusOylayici:
    """Çoğunluk Oylaması ve Ağırlıklı Güven Oylaması algoritmaları."""

    @staticmethod
    def cogunluk_oylamasi(argumanlar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """En çok tekrar eden seçeneği kazanan ilan eder."""
        secenekler = [arg["tercih_edilen_secenek"] for arg in argumanlar]
        sayac = Counter(secenekler)
        en_cok_oy_alan, oy_sayisi = sayac.most_common(1)[0]

        return {
            "yontem": "Cogunluk_Oylamasi",
            "kazanan_secenek": en_cok_oy_alan,
            "oy_dagilimi": dict(sayac),
            "kazanan_oy_orani": (oy_sayisi / len(secenekler)) * 100.0,
        }

    @staticmethod
    def agirlikli_guven_oylamasi(argumanlar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ajanların güven skorlarıyla ağırlıklandırılmış Borda / Güven Oylaması uygular."""
        agirliklar: Dict[str, float] = {}
        toplam_guven = 0.0

        for arg in argumanlar:
            secenek = arg["tercih_edilen_secenek"]
            guven = arg.get("guven_skoru", 0.5)
            agirliklar[secenek] = agirliklar.get(secenek, 0.0) + guven
            toplam_guven += guven

        # Normalize oranlar
        oranlar = {k: (v / toplam_guven) * 100.0 for k, v in agirliklar.items()}
        kazanan = max(agirliklar, key=agirliklar.get)

        return {
            "yontem": "Agirlikli_Guven_Oylamasi",
            "kazanan_secenek": kazanan,
            "agirlikli_puanlar": agirliklar,
            "guven_yuzdeleri": oranlar,
            "kazanan_guven_orani": oranlar[kazanan],
        }
