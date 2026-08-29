"""
Varlık Çıkarıcı (Entity Extraction) Modülü (Day 136 - Faz 7 - GraphRAG-1).
Metinlerden teknik varlıkları, tiplerini ve açıklamalarını çıkaran modül.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
import re


@dataclass
class Varlik:
    """Bilgi Grafı Düğümü (Entity / Node)."""
    isim: str
    tip: str  # TEKNOLOJI, ALGORITMA, KAVRAM, ORGANIZASYON, METRIK
    aciklama: str = ""
    aliaslar: List[str] = field(default_factory=list)
    frekans: int = 1


class VarlikCikarici:
    """Teknik metinlerden varlıkları (Entities) çıkaran kural ve örüntü motoru."""

    VARLIK_KATALOGU: Dict[str, Dict[str, Any]] = {
        "Raft": {
            "tip": "ALGORITMA",
            "aciklama": "Dağıtık sistemlerde durum makinesi çoğaltması ve lider seçimi protokolü.",
            "aliaslar": ["Raft Protokolü", "Raft Konsensüs"],
        },
        "Quorum": {
            "tip": "KAVRAM",
            "aciklama": "Ağ bölünmesinde çoğunluk kuralı ile bölünmüş beyin durumunu engelleyen mekanizma.",
            "aliaslar": ["Çoğunluk Kuralı"],
        },
        "Vision Transformer": {
            "tip": "TEKNOLOJI",
            "aciklama": "Görüntüleri piksel yamalarına bölerek dikkat mekanizmasıyla işleyen derin öğrenme mimarisi.",
            "aliaslar": ["ViT", "Vision Transformers"],
        },
        "Self-Attention": {
            "tip": "ALGORITMA",
            "aciklama": "Token ilişkilerini QKV matrisleri üzerinden küresel modelleyen dikkat mekanizması.",
            "aliaslar": ["Öz-Dikkat", "Öz Dikkat"],
        },
        "PostgreSQL": {
            "tip": "TEKNOLOJI",
            "aciklama": "Gelişmiş B-Tree ve GIN indeksleme desteğine sahip açık kaynaklı ilişkisel veritabanı.",
            "aliaslar": ["Postgres"],
        },
        "B-Tree": {
            "tip": "ALGORITMA",
            "aciklama": "Veritabanlarında logaritmik sürede arama ve aralık sorgusu sunan dengeli ağaç yapısı.",
            "aliaslar": ["B-Tree İndeksi"],
        },
        "Limit Order Book": {
            "tip": "KAVRAM",
            "aciklama": "Finansal piyasalarda alım-satım emirlerini fiyat ve zaman önceliğine göre sıralayan defter.",
            "aliaslar": ["LOB", "Emir Defteri"],
        },
        "FPGA": {
            "tip": "TEKNOLOJI",
            "aciklama": "Yüksek frekanslı ticarette mikrosaniye altı işlem gecikmesi sunan programlanabilir donanım.",
            "aliaslar": ["Donanım Hızlandırıcı"],
        },
        "NDCG@5": {
            "tip": "METRIK",
            "aciklama": "Bilgi getirmede sıralı sonuçların alaka düzeyini ve pozisyonunu ölçen değerlendirme metriği.",
            "aliaslar": ["NDCG"],
        },
    }

    @classmethod
    def cikar(cls, metin: str) -> List[Varlik]:
        """Metinden katalog ve büyük harf/özel kalıplarla varlıkları çıkarır."""
        bulunan_varliklar: Dict[str, Varlik] = {}

        # 1. Katalog Bazlı Eşleme
        for ana_isim, detay in cls.VARLIK_KATALOGU.items():
            olasi_isimler = [ana_isim] + detay.get("aliaslar", [])
            for isim in olasi_isimler:
                if re.search(r"\b" + re.escape(isim) + r"\b", metin, re.IGNORECASE):
                    if ana_isim not in bulunan_varliklar:
                        bulunan_varliklar[ana_isim] = Varlik(
                            isim=ana_isim,
                            tip=detay["tip"],
                            aciklama=detay["aciklama"],
                            aliaslar=detay.get("aliaslar", []),
                            frekans=1,
                        )
                    else:
                        bulunan_varliklar[ana_isim].frekans += 1
                    break

        # 2. Heuristik Büyük Harfli Özel İsim Çıkarımı
        kaliplar = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b", metin)
        for kalip in kaliplar:
            if kalip not in bulunan_varliklar and len(kalip) > 3 and kalip not in ["Bölüm", "KAYNAK", "AŞAMA"]:
                bulunan_varliklar[kalip] = Varlik(
                    isim=kalip,
                    tip="KAVRAM",
                    aciklama=f"Metin içinden dinamik olarak keşfedilen teknik varlık: {kalip}",
                    aliaslar=[],
                    frekans=1,
                )

        return list(bulunan_varliklar.values())
