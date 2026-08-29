"""
Tip Güvenli Araç Yönlendirici (Tool Dispatcher) Modülü (Day 124 - Faz 7).
JSON Schema doğrulaması, araç icrası ve standart tool_call / tool_result yanıt yönetimi.
"""

from typing import Dict, Any, List, Callable, Optional
import uuid

from .arac_semasi import AracSemasi, ParametreOzelligi
from .json_ayristirici import GuvenliJsonAyristirici


class AracKaydi:
    """Tek bir kayıtlı aracın şeması ve yürütücü fonksiyonu."""

    def __init__(self, sema: AracSemasi, calistirici: Callable[..., Any]):
        self.sema = sema
        self.calistirici = calistirici


class AracYonlendirici:
    """Tüm tip güvenli araçları yöneten ve yönlendiren merkezi modül."""

    def __init__(self):
        self.araclar: Dict[str, AracKaydi] = {}
        self._varsayilan_araclari_kaydet()

    def arac_ekle(self, sema: AracSemasi, calistirici: Callable[..., Any]):
        self.araclar[sema.ad] = AracKaydi(sema=sema, calistirici=calistirici)

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Kayıtlı tüm araçların OpenAI JSON Schema listesini döndürür."""
        return [kayit.sema.to_openai_schema() for kayit in self.araclar.values()]

    def cagir(self, tool_call_veri: Any, tool_call_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Tool call isteğini ayrıştırır, doğrular ve çalıştırır.
        """
        cid = tool_call_id or f"call_{uuid.uuid4().hex[:8]}"

        # 1. JSON Ayrıştırma
        if isinstance(tool_call_veri, str):
            basarili, parsed, onarildi, hata = GuvenliJsonAyristirici.ayristir(tool_call_veri)
            if not basarili:
                return {
                    "tool_call_id": cid,
                    "basarili": False,
                    "hata_tipi": "JsonParseError",
                    "hata_mesaji": hata,
                    "sonuc": None,
                }
            veri = parsed
        else:
            veri = tool_call_veri

        arac_adi = veri.get("name") or veri.get("arac_adi")
        argumanlar = veri.get("arguments") or veri.get("argumanlar") or {}

        if isinstance(argumanlar, str):
            _, arg_parsed, _, _ = GuvenliJsonAyristirici.ayristir(argumanlar)
            argumanlar = arg_parsed or {}

        if not arac_adi or arac_adi not in self.araclar:
            return {
                "tool_call_id": cid,
                "arac_adi": arac_adi,
                "basarili": False,
                "hata_tipi": "ToolNotFoundError",
                "hata_mesaji": f"'{arac_adi}' adında bir araç bulunamadı. Mevcut araçlar: {list(self.araclar.keys())}",
                "sonuc": None,
            }

        kayit = self.araclar[arac_adi]

        # 2. JSON Schema & Tip Validasyonu
        dogru_mu, dogrulama_hatalari = kayit.sema.dogrula(argumanlar)
        if not dogru_mu:
            return {
                "tool_call_id": cid,
                "arac_adi": arac_adi,
                "basarili": False,
                "hata_tipi": "SchemaValidationError",
                "hata_mesaji": f"Şema doğrulama hatası: {'; '.join(dogrulama_hatalari)}",
                "sonuc": None,
            }

        # 3. Güvenli Yürütme
        try:
            cikti = kayit.calistirici(**argumanlar)
            return {
                "tool_call_id": cid,
                "arac_adi": arac_adi,
                "basarili": True,
                "hata_tipi": None,
                "hata_mesaji": None,
                "sonuc": cikti,
            }
        except Exception as e:
            return {
                "tool_call_id": cid,
                "arac_adi": arac_adi,
                "basarili": False,
                "hata_tipi": "ExecutionError",
                "hata_mesaji": f"Araç çalışma hatası: {str(e)}",
                "sonuc": None,
            }

    def _varsayilan_araclari_kaydet(self):
        """Örnek finansal ve operasyonel araçları sisteme kaydeder."""

        # Araç 1: Hisse Senedi Sorgulama
        sema_borsa = AracSemasi(
            ad="HisseSenediSorgula",
            aciklama="BIST hisse senedinin anlık fiyatını ve değişim oranını getirir.",
            parametreler={
                "sembol": ParametreOzelligi(tip="string", aciklama="Hisse kodu", enum=["THYAO", "ASELS", "GARAN", "KCHOL"]),
                "para_birimi": ParametreOzelligi(tip="string", aciklama="Fiyat para birimi", enum=["TRY", "USD"]),
            },
            zorunlu_alanlar=["sembol"],
        )

        def fn_borsa(sembol: str, para_birimi: str = "TRY") -> Dict[str, Any]:
            fiyatlar = {"THYAO": 312.50, "ASELS": 64.20, "GARAN": 118.00, "KCHOL": 245.80}
            fiyat = fiyatlar.get(sembol, 100.0)
            if para_birimi == "USD":
                fiyat = round(fiyat / 36.5, 2)
            return {"sembol": sembol, "fiyat": fiyat, "para_birimi": para_birimi, "durum": "Piyasa Açık"}

        self.arac_ekle(sema_borsa, fn_borsa)

        # Araç 2: Uçuş Rezervasyonu
        sema_ucus = AracSemasi(
            ad="UcusRezervasyonuYap",
            aciklama="Uçak bileti rezervasyonu oluşturur.",
            parametreler={
                "kalkis": ParametreOzelligi(tip="string", aciklama="Kalkış havalimanı"),
                "varis": ParametreOzelligi(tip="string", aciklama="Varış havalimanı"),
                "yolcu_sayisi": ParametreOzelligi(tip="integer", aciklama="Yolcu adedi (1-9)", minimum=1, maximum=9),
                "business_class": ParametreOzelligi(tip="boolean", aciklama="Business sınıfı mı?"),
            },
            zorunlu_alanlar=["kalkis", "varis", "yolcu_sayisi"],
        )

        def fn_ucus(kalkis: str, varis: str, yolcu_sayisi: int, business_class: bool = False) -> Dict[str, Any]:
            pnr = f"TK{uuid.uuid4().hex[:6].upper()}"
            return {
                "pnr": pnr,
                "rota": f"{kalkis} -> {varis}",
                "yolcu_sayisi": yolcu_sayisi,
                "sinif": "Business" if business_class else "Economy",
                "durum": "Onaylandı",
            }

        self.arac_ekle(sema_ucus, fn_ucus)
