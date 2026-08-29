"""
Kurumsal İade ve Risk Denetim İş Akışı Modülü (Day 127 - Faz 7).
LangGraph StateGraph üzerinde çalışan gerçek dünya finansal iade ve dolandırıcılık denetim akışı.
"""

from typing import Dict, Any
from .cizge_motoru import DurumsalCizge, END


def node_talep_ayristirici(durum: Dict[str, Any]) -> Dict[str, Any]:
    """Kullanıcı mesajını ve talep tutarını ayrıştıran düğüm."""
    tutar = durum.get("talep_tutari", 0.0)
    mesaj = f"İade talebi alındı. Tutar: {tutar} TL"
    return {
        "mesajlar": [{"rol": "asistan", "icerik": mesaj}],
        "nihai_durum": "TALEP_AYRISTIRILDI",
    }


def node_risk_degerlendirici(durum: Dict[str, Any]) -> Dict[str, Any]:
    """Tutar ve işlem geçmişine göre dolandırıcılık risk skoru hesaplayan düğüm."""
    tutar = durum.get("talep_tutari", 0.0)

    # 5000 TL üzeri yüksek risk ve insan onayı gerektirir
    if tutar > 5000.0:
        risk = 0.85
        onay_gerekli = True
    else:
        risk = 0.20
        onay_gerekli = False

    return {
        "risk_skoru": risk,
        "onay_gerekli_mi": onay_gerekli,
        "mesajlar": [{"rol": "asistan", "icerik": f"Risk değerlendirmesi tamamlandı: Skor={risk:.2f}"}],
    }


def router_risk_kontrol(durum: Dict[str, Any]) -> str:
    """Risk skoruna göre sıradaki düğümü belirleyen yönlendirici."""
    if durum.get("risk_skoru", 0.0) > 0.70:
        return "yuksek_risk"
    return "dusuk_risk"


def node_insan_onayi(durum: Dict[str, Any]) -> Dict[str, Any]:
    """Yetkili denetçinin onayını işleyen düğüm (HITL)."""
    onay = durum.get("insan_onayladi_mi", False)
    return {
        "mesajlar": [{"rol": "denetci", "icerik": f"İnsan Denetçi Kararı: {'ONAYLANDI' if onay else 'REDDEDİLDİ'}"}],
        "nihai_durum": "DENETCI_INCELENDI",
    }


def router_onay_kontrol(durum: Dict[str, Any]) -> str:
    """İnsan denetçinin onayına göre yönlendirme yapar."""
    if durum.get("insan_onayladi_mi") is True:
        return "onaylandi"
    return "reddedildi"


def node_odeme_iadesi(durum: Dict[str, Any]) -> Dict[str, Any]:
    """İade tutarını banka API'sine ileten yürütücü düğüm."""
    return {
        "odeme_yapildi_mi": True,
        "nihai_durum": "IADE_TAMAMLANDI",
        "mesajlar": [{"rol": "sistem", "icerik": f"{durum.get('talep_tutari', 0.0)} TL tutarındaki iade hesaba aktarıldı."}],
    }


def node_talep_reddi(durum: Dict[str, Any]) -> Dict[str, Any]:
    """Onaylanmayan veya yüksek riskli talebi reddeden düğüm."""
    return {
        "odeme_yapildi_mi": False,
        "nihai_durum": "TALEP_REDDEDILDI",
        "mesajlar": [{"rol": "sistem", "icerik": "İade talebi risk ve güvenlik politikaları nedeniyle reddedildi."}],
    }


def node_bilgilendirme_epostasi(durum: Dict[str, Any]) -> Dict[str, Any]:
    """Müşteriye sonuç bildirim e-postası gönderen düğüm."""
    durum_str = durum.get("nihai_durum", "BILINMIYOR")
    return {
        "mesajlar": [{"rol": "asistan", "icerik": f"Müşteriye sonuç bilgilendirme e-postası iletildi ({durum_str})."}],
    }


def iade_akisi_olustur() -> DurumsalCizge:
    """Tüm düğüm ve kenarlarıyla hazır LangGraph İade İş Akışı oluşturur."""
    cizge = DurumsalCizge()

    # Düğümleri Ekle
    cizge.add_node("TalepAyristirici", node_talep_ayristirici)
    cizge.add_node("RiskDegerlendirici", node_risk_degerlendirici)
    cizge.add_node("InsanOnayi", node_insan_onayi)
    cizge.add_node("OdemeIadesi", node_odeme_iadesi)
    cizge.add_node("TalepReddi", node_talep_reddi)
    cizge.add_node("BilgilendirmeEpostasi", node_bilgilendirme_epostasi)

    # Giriş Noktası
    cizge.set_entry_point("TalepAyristirici")

    # Kenarlar
    cizge.add_edge("TalepAyristirici", "RiskDegerlendirici")

    # Koşullu Kenar 1: Risk Kontrolü
    cizge.add_conditional_edges(
        "RiskDegerlendirici",
        router_risk_kontrol,
        {"yuksek_risk": "InsanOnayi", "dusuk_risk": "OdemeIadesi"},
    )

    # İnsan Onayı Kesintisi Tanımla (HITL Breakpoint)
    cizge.kesinti_tanimla("InsanOnayi")

    # Koşullu Kenar 2: İnsan Onayı Sonrası
    cizge.add_conditional_edges(
        "InsanOnayi",
        router_onay_kontrol,
        {"onaylandi": "OdemeIadesi", "reddedildi": "TalepReddi"},
    )

    cizge.add_edge("OdemeIadesi", "BilgilendirmeEpostasi")
    cizge.add_edge("TalepReddi", "BilgilendirmeEpostasi")
    cizge.add_edge("BilgilendirmeEpostasi", END)

    return cizge
