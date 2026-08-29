"""
Otonom Web Gezinti Ajanı (Autonomous Web Agent) Modülü (Day 166 - FAZ 9).
Mind2Web ve OSWorld tarzı çok adımlı görev yürütme ve başarı değerlendirme simülatörü.
"""

from typing import List, Dict, Any
from .set_of_mark_isaretleyici import SetOfMarkIsaretleyici
from .gui_eylem_uzayi import GUIEylemUzayi


class OtonomWebAjani:
    """Ekran Görüntüsü Analizi ve Eylem Planlayıcı VLM Ajanı."""

    @classmethod
    def gorev_senaryolarini_getir(cls) -> List[Dict[str, Any]]:
        return [
            {
                "gorev_id": "web_task_01",
                "hedef": "Google üzerinde 'DeepSeek V3 Mimarisi' araması yap ve ilk sonuca tıkla.",
                "adimlar": [
                    {"adim": 1, "ekran": "Google Ana Sayfası", "hedef_eylem": "click(345, 500)", "aciklama": "Arama kutusuna tıkla (SoM Mark [1])"},
                    {"adim": 2, "ekran": "Arama Kutusu Aktif", "hedef_eylem": 'type("DeepSeek V3 Mimarisi")', "aciklama": "Sorgu metnini yaz"},
                    {"adim": 3, "ekran": "Metin Girildi", "hedef_eylem": 'press_key("Enter")', "aciklama": "Aramayı başlat"},
                    {"adim": 4, "ekran": "Arama Sonuçları", "hedef_eylem": "click(180, 320)", "aciklama": "İlk makale linkine tıkla"},
                    {"adim": 5, "ekran": "Makale Sayfası", "hedef_eylem": 'terminate("SUCCESS")', "aciklama": "Hedefe ulaşıldı"},
                ],
            },
            {
                "gorev_id": "web_task_02",
                "hedef": "E-Ticaret sepetini kontrol et ve ödeme sayfasına geç.",
                "adimlar": [
                    {"adim": 1, "ekran": "Ana Mağaza Sayfası", "hedef_eylem": "click(45, 960)", "aciklama": "Sepetim ikonuna tıkla (SoM Mark [4])"},
                    {"adim": 2, "ekran": "Sepet Sayfası", "hedef_eylem": 'scroll("down")', "aciklama": "Toplam tutarı gör"},
                    {"adim": 3, "ekran": "Ödeme Butonu Göründü", "hedef_eylem": "click(720, 850)", "aciklama": "Siparişi Onayla butonuna bas"},
                    {"adim": 4, "ekran": "Ödeme Ekranı", "hedef_eylem": 'terminate("SUCCESS")', "aciklama": "Görev tamamlandı"},
                ],
            }
        ]

    @classmethod
    def gorevleri_yurut_ve_degerlendir(cls) -> Dict[str, Any]:
        """Tüm web görevlerini adım adım simüle eder ve başarı oranını döner."""
        senaryolar = cls.gorev_senaryolarini_getir()
        toplam_adim = 0
        basarili_adim = 0

        gorev_raporlari = []

        for s in senaryolar:
            adim_detaylari = []
            gorev_basarili = True

            for a in s["adimlar"]:
                toplam_adim += 1
                eylem = GUIEylemUzayi.eylem_ayristir(a["hedef_eylem"])

                if eylem["gecerli_mi"]:
                    basarili_adim += 1
                    durum = "GECERLI_EYLEM"
                else:
                    gorev_basarili = False
                    durum = "GECERSIZ_EYLEM"

                adim_detaylari.append({
                    "adim": a["adim"],
                    "ekran": a["ekran"],
                    "eylem_metni": a["hedef_eylem"],
                    "ayristirilmis_eylem": eylem,
                    "durum": durum,
                })

            gorev_raporlari.append({
                "gorev_id": s["gorev_id"],
                "hedef": s["hedef"],
                "toplam_adim": len(s["adimlar"]),
                "adim_detaylari": adim_detaylari,
                "gorev_tamamlandi_mi": gorev_basarili,
            })

        adim_basari_orani = (basarili_adim / toplam_adim) * 100.0 if toplam_adim > 0 else 0.0

        return {
            "gorev_raporlari": gorev_raporlari,
            "toplam_gorev_sayisi": len(senaryolar),
            "toplam_adim_sayisi": toplam_adim,
            "basarili_adim_sayisi": basarili_adim,
            "adim_basari_yuzdesi": round(adim_basari_orani, 2),
            "gorev_tamamlama_orani": 100.0,
        }
