"""
Human-in-the-Loop (HITL) Güvenlik Bariyeri Motoru (Day 232 - FAZ 12).
Risk Derecelendirme, Akış Dondurma (Interrupt Gate) ve İnsan Onay Yönetimi.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class RiskLevel(Enum):
    """Eylem Risk Seviyeleri."""
    LOW = "DUSUK"          # Salt okuma, sorgu, loglama (Otomatik İcra)
    MEDIUM = "ORTA"        # Geçici dosya yazma, önbellek temizleme (Loglu İcra)
    HIGH = "YUKSEK"        # Dosya silme, konfigürasyon değiştirme (İnsan Onayı Zorunlu)
    CRITICAL = "KRITIK"    # Veritabanı drop, para transferi, e-posta yayını (Çift Onay)


class ActionRequest:
    """Ajan Tarafından Talep Edilen Eylem Modeli."""

    def __init__(
        self,
        arac_adi: str,
        parametreler: Dict[str, Any],
        risk_seviyesi: RiskLevel,
        gerekce: str,
    ):
        self.arac_adi = arac_adi
        self.parametreler = parametreler
        self.risk_seviyesi = risk_seviyesi
        self.gerekce = gerekce


class ApprovalDecision:
    """İnsan Tarafından Verilen Onay / Red Kararı."""

    def __init__(
        self,
        onaylandi_mi: bool,
        insan_yorumu: Optional[str] = None,
        revize_parametreler: Optional[Dict[str, Any]] = None,
    ):
        self.onaylandi_mi = onaylandi_mi
        self.insan_yorumu = insan_yorumu or ("İşlem onaylandı." if onaylandi_mi else "İşlem reddedildi.")
        self.revize_parametreler = revize_parametreler


class HITLGuardrailAgent:
    """Human-in-the-Loop Güvenlik ve Onay Bariyeri Ajanı."""

    KRITIK_ARACLAR = {
        "delete_database_table": RiskLevel.CRITICAL,
        "transfer_funds": RiskLevel.CRITICAL,
        "drop_table": RiskLevel.CRITICAL,
        "delete_file": RiskLevel.HIGH,
        "update_user_role": RiskLevel.HIGH,
        "write_temp_file": RiskLevel.MEDIUM,
        "read_file": RiskLevel.LOW,
        "query_database": RiskLevel.LOW,
    }

    @classmethod
    def risk_derecelendir(cls, arac_adi: str, parametreler: Dict[str, Any]) -> RiskLevel:
        """Araç adı ve parametrelerine göre dinamik risk seviyesi belirler."""
        if arac_adi in cls.KRITIK_ARACLAR:
            return cls.KRITIK_ARACLAR[arac_adi]

        # Anahtar kelime tabanlı sezgisel kural
        arac_kucuk = arac_adi.lower()
        if any(k in arac_kucuk for k in ["delete", "drop", "transfer", "revoke", "format"]):
            return RiskLevel.CRITICAL
        elif any(k in arac_kucuk for k in ["write", "update", "modify", "patch"]):
            return RiskLevel.HIGH
        elif any(k in arac_kucuk for k in ["read", "get", "query", "search", "list"]):
            return RiskLevel.LOW
        return RiskLevel.MEDIUM

    def eylem_talebi_olustur(
        self,
        arac_adi: str,
        parametreler: Dict[str, Any],
        gerekce: str,
    ) -> ActionRequest:
        risk = self.risk_derecelendir(arac_adi, parametreler)
        return ActionRequest(
            arac_adi=arac_adi,
            parametreler=parametreler,
            risk_seviyesi=risk,
            gerekce=gerekce,
        )

    def eylemi_denetle_ve_icra_et(
        self,
        talep: ActionRequest,
        insan_onayi: Optional[ApprovalDecision] = None,
    ) -> Dict[str, Any]:
        """Eylemi risk seviyesine göre doğrudan çalıştırır ya da insan onayını bekler."""
        # 1. Düşük ve Orta Risk -> Otomatik Güvenli İcra
        if talep.risk_seviyesi in [RiskLevel.LOW, RiskLevel.MEDIUM]:
            return {
                "durum": "OTOMATIK_ICRA",
                "mesaj": f"'{talep.arac_adi}' güvenli görüldü ve otomatik çalıştırıldı.",
                "risk": talep.risk_seviyesi.value,
                "icra_edildi_mi": True,
            }

        # 2. Yüksek ve Kritik Risk -> İnsan Onayı Yoksa Akışı Dondur (Interrupt)
        if insan_onayi is None:
            return {
                "durum": "INTERRUPT_BEKLEMEDE",
                "mesaj": f"🛑 [HITL DONDURMA]: '{talep.arac_adi}' ({talep.risk_seviyesi.value} Risk) insan onayı bekliyor!",
                "risk": talep.risk_seviyesi.value,
                "icra_edildi_mi": False,
            }

        # 3. İnsan Onayı Verildiyse
        if insan_onayi.onaylandi_mi:
            params = insan_onayi.revize_parametreler or talep.parametreler
            return {
                "durum": "ONAYLI_ICRA",
                "mesaj": f"✓ [İNSAN ONAYI VERİLDİ]: '{talep.arac_adi}' başarıyla çalıştırıldı. Not: {insan_onayi.insan_yorumu}",
                "risk": talep.risk_seviyesi.value,
                "icra_edildi_mi": True,
                "parametreler": params,
            }
        else:
            return {
                "durum": "REDDEDILDI",
                "mesaj": f"❌ [İNSAN REDDETTİ]: '{talep.arac_adi}' icrası iptal edildi. Gerekçe: {insan_onayi.insan_yorumu}",
                "risk": talep.risk_seviyesi.value,
                "icra_edildi_mi": False,
            }
