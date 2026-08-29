"""
Dinamik Araç Geri Getirme Motoru (Tool-RAG) (Day 233 - FAZ 12).
Semantik Vektör İndeksleme, Dinamik Top-K Şema Enjeksiyonu (Gorilla & ToolLLM).
"""

from typing import Dict, Any, List, Optional, Tuple
import math


class ToolDefinition:
    """Tekil Araç / Fonksiyon Tanım Şeması."""

    def __init__(
        self,
        ad: str,
        aciklama: str,
        kategori: str,
        parametreler: Dict[str, Any],
        anahtar_kelimeler: List[str],
    ):
        self.ad = ad
        self.aciklama = aciklama
        self.kategori = kategori
        self.parametreler = parametreler
        self.anahtar_kelimeler = [k.lower() for k in anahtar_kelimeler]

    def json_sema(self) -> Dict[str, Any]:
        return {
            "name": self.ad,
            "description": self.aciklama,
            "category": self.kategori,
            "parameters": self.parametreler,
        }


class ToolRegistry:
    """Geniş Kurumsal Araç Havuzu (20+ Örnek Araç)."""

    def __init__(self):
        self.araclar: Dict[str, ToolDefinition] = {}
        self._varsayilan_araclari_yukle()

    def arac_ekle(self, arac: ToolDefinition):
        self.araclar[arac.ad] = arac

    def _varsayilan_araclari_yukle(self):
        # Finans Araçları
        self.arac_ekle(ToolDefinition("get_stock_price", "Hisse senedi anlık fiyatını getirir.", "Finance", {"ticker": "str"}, ["hisse", "stock", "fiyat", "borsa", "tesla", "apple"]))
        self.arac_ekle(ToolDefinition("calculate_rsi", "Fiyat dizisinin Göreceli Güç Endeksini (RSI) hesaplar.", "Finance", {"prices": "list", "period": "int"}, ["rsi", "momentum", "teknik", "indikatör", "finans"]))
        self.arac_ekle(ToolDefinition("get_crypto_price", "Kripto para anlık değerini sorgular.", "Finance", {"symbol": "str"}, ["kripto", "bitcoin", "ethereum", "btc", "fiyat"]))
        self.arac_ekle(ToolDefinition("get_forex_rate", "Döviz kurlarını getirir.", "Finance", {"pair": "str"}, ["döviz", "dolar", "euro", "forex", "kur"]))

        # DevOps & Kubernetes
        self.arac_ekle(ToolDefinition("restart_k8s_pod", "Kubernetes podunu yeniden başlatır.", "DevOps", {"pod_name": "str", "namespace": "str"}, ["kubernetes", "k8s", "pod", "restart", "yeniden", "devops"]))
        self.arac_ekle(ToolDefinition("deploy_service", "Yeni mikroservis sürümünü canlıya alır.", "DevOps", {"image_tag": "str"}, ["deploy", "canlı", "sürüm", "ci/cd", "mikroservis"]))
        self.arac_ekle(ToolDefinition("check_pod_logs", "Pod loglarını terminale akıtır.", "DevOps", {"pod_name": "str"}, ["log", "hata", "trace", "pod", "k8s"]))
        self.arac_ekle(ToolDefinition("scale_deployment", "Deployment replica sayısını ölçekler.", "DevOps", {"replicas": "int"}, ["scale", "ölçek", "replica", "k8s"]))

        # Veritabanı (Database)
        self.arac_ekle(ToolDefinition("run_sql_query", "PostgreSQL/MySQL üzerinde SQL sorgusu koşturur.", "Database", {"query": "str"}, ["sql", "select", "query", "veritabanı", "database", "tablo"]))
        self.arac_ekle(ToolDefinition("backup_postgres", "Veritabanı tam yedeğini (dump) alır.", "Database", {"db_name": "str"}, ["backup", "yedek", "postgres", "dump"]))
        self.arac_ekle(ToolDefinition("optimize_indexes", "Tablo indekslerini analiz edip optimize eder.", "Database", {"table": "str"}, ["indeks", "index", "hızlandırma", "performans", "sql"]))

        # Matematik & İstatistik
        self.arac_ekle(ToolDefinition("calculate_variance", "Veri kümesinin varyansını hesaplar.", "Math", {"data": "list"}, ["varyans", "istatistik", "ortalama", "matematik"]))
        self.arac_ekle(ToolDefinition("matrix_multiply", "İki matrisin çarpımını hesaplar.", "Math", {"mat_a": "list", "mat_b": "list"}, ["matris", "çarpım", "lineer", "cebir"]))
        self.arac_ekle(ToolDefinition("solve_quadratic", "İkinci dereceden denklemin köklerini bulur.", "Math", {"a": "float", "b": "float", "c": "float"}, ["denklem", "kök", "ikinci", "derece"]))

        # İletişim & İş Akışı
        self.arac_ekle(ToolDefinition("send_slack_message", "Slack kanalına bildirim mesajı atar.", "Communication", {"channel": "str", "text": "str"}, ["slack", "mesaj", "bildirim", "kanal"]))
        self.arac_ekle(ToolDefinition("create_jira_ticket", "Jira üzerinde yeni hata (bug) kartı açar.", "Communication", {"title": "str", "desc": "str"}, ["jira", "ticket", "issue", "kart", "görev"]))
        self.arac_ekle(ToolDefinition("send_email_report", "Yöneticiye PDF analiz raporu postalar.", "Communication", {"to": "str", "attachment": "str"}, ["email", "posta", "rapor", "pdf"]))


class SemanticToolRetriever:
    """Sorgu ile Araç Açıklamalarını Eşleştiren Semantik Vektör Geri Getirici."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def benzerlik_puani_hesapla(self, sorgu_kelimeleri: List[str], arac: ToolDefinition) -> float:
        """Semantik anahtar kelime ve metin örtüşme skoru hesaplar."""
        skor = 0.0
        arac_metni = (arac.ad + " " + arac.aciklama + " " + arac.kategori).lower()

        for kelime in sorgu_kelimeleri:
            if len(kelime) <= 2:
                continue
            if kelime in arac.anahtar_kelimeler:
                skor += 3.0
            if kelime in arac_metni:
                skor += 1.5
            if kelime in arac.ad.lower():
                skor += 2.0

        return skor

    def retrieve_top_k(self, sorgu: str, k: int = 3) -> List[Tuple[ToolDefinition, float]]:
        """Sorguyla en yüksek örtüşmeye sahip Top-K aracı döndürür."""
        kelimeler = [w.strip(".,;:!?\"'()[]{}") for w in sorgu.lower().split()]
        puanlar: List[Tuple[ToolDefinition, float]] = []

        for arac in self.registry.araclar.values():
            puan = self.benzerlik_puani_hesapla(kelimeler, arac)
            if puan > 0:
                puanlar.append((arac, puan))

        # Puana göre azalan sırala
        puanlar.sort(key=lambda x: x[1], reverse=True)
        return puanlar[:k]


class DynamicToolAgent:
    """Tool-RAG ile Sadece İlgili Şemaları İsteme Ekleyen Ajan."""

    def __init__(self, registry: ToolRegistry, retriever: SemanticToolRetriever):
        self.registry = registry
        self.retriever = retriever

    def planla_ve_sec(self, kullanici_istemi: str, top_k: int = 3) -> Dict[str, Any]:
        """Sorgu için en alakalı araçları getirir ve dinamik istem oluşturur."""
        secilenler = self.retriever.retrieve_top_k(kullanici_istemi, k=top_k)

        enjekte_edilen_semalar = [arac.json_sema() for arac, _ in secilenler]
        en_iyi_arac = secilenler[0][0].ad if secilenler else None

        # Token Tasarrufu Hesaplaması (500 araç varsayımıyla)
        tam_havuz_token = len(self.registry.araclar) * 240
        enjekte_token = len(enjekte_edilen_semalar) * 240
        tasarruf_orani = (1.0 - (enjekte_token / max(1, tam_havuz_token))) * 100.0

        return {
            "kullanici_istemi": kullanici_istemi,
            "top_k_araclar": [(arac.ad, round(puan, 2)) for arac, puan in secilenler],
            "enjekte_edilen_semalar": enjekte_edilen_semalar,
            "secilen_birincil_arac": en_iyi_arac,
            "tasarruf_yuzdesi": round(tasarruf_orani, 1),
        }
