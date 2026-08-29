"""
Day 137: GraphRAG-2: Bilgi Grafı (Knowledge Graph), Cypher Sorgulama ve Multi-Hop Gezinti Ana Akışı.
Labeled Property Graph (LPG) üzerinde deklaratif sorgulama ve akıl yürütme zinciri.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ozellikli_graf_deposu import OzellikliGrafDeposu
from src.cypher_ayristirici_ve_motor import CypherMotoru
from src.graf_gezgini import GrafGezgini
from src.gorsellestirici import CypherGraphGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 137: GraphRAG-2 - Knowledge Graph, Cypher Query Engine & Multi-Hop Traversal")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # ADIM 1: Property Graph Veritabanını Kur ve Doldur
    # -------------------------------------------------------------
    print("\n[1/4] Labeled Property Graph (LPG) Veritabanı Yükleniyor...")
    depo = OzellikliGrafDeposu()

    # Düğümler (Nodes & Properties)
    depo.dugum_ekle("Vision Transformer", etiket="TEKNOLOJI", aciklama="Piksel yamalarını transformer bloklarıyla işleyen model.")
    depo.dugum_ekle("Self-Attention", etiket="ALGORITMA", aciklama="Tokenlar arası küresel korelasyonu modelleyen dikkat katmanı.")
    depo.dugum_ekle("FlashAttention", etiket="ALGORITMA", aciklama="GPU SRAM bellek bant genişliğini optimize eden tam dikkat motoru.")
    depo.dugum_ekle("NVIDIA GPU", etiket="DONANIM", aciklama="Tensor çekirdekleri ve yüksek bellek bant genişliği sunan donanım.")
    depo.dugum_ekle("Raft", etiket="ALGORITMA", aciklama="Dağıtık sistemlerde lider seçimi ve durum çoğaltması protokolü.")
    depo.dugum_ekle("Quorum", etiket="KAVRAM", aciklama="Ağ bölünmelerinde çoğunluk kararı ile tutarlılık sağlayan kural.")
    depo.dugum_ekle("PostgreSQL", etiket="TEKNOLOJI", aciklama="ACID uyumlu ilişkisel veritabanı yönetim sistemi.")
    depo.dugum_ekle("B-Tree", etiket="ALGORITMA", aciklama="Logaritmik sürede veri erişimi sunan dengeli ağaç indeksi.")

    # Kenarlar (Directed Relationships)
    depo.kenar_ekle("Vision Transformer", "Self-Attention", "KULLANIR", agirlik=1.0)
    depo.kenar_ekle("Self-Attention", "FlashAttention", "HIZLANDIRIR", agirlik=1.0)
    depo.kenar_ekle("FlashAttention", "NVIDIA GPU", "CALISIR", agirlik=1.0)
    depo.kenar_ekle("Raft", "Quorum", "UYGULAR", agirlik=1.0)
    depo.kenar_ekle("PostgreSQL", "B-Tree", "DESTEKLER", agirlik=1.0)

    print(f"  [✓] Toplam Düğüm Sayısı: {len(depo.tum_dugumler())}")
    print(f"  [✓] Toplam Kenar Sayısı: {len(depo.tum_kenarlar())}")

    # -------------------------------------------------------------
    # ADIM 2: Cypher Deklaratif Sorgularını İcra Et
    # -------------------------------------------------------------
    print("\n[2/4] Cypher Deklaratif Sorguları İcra Ediliyor (Query Execution)...")
    motor = CypherMotoru(depo)

    # 1-Hop Sorgusu
    sorgu_1 = "MATCH (a)-[r:KULLANIR]->(b) WHERE a.id = 'Vision Transformer' RETURN b"
    sonuc_1 = motor.sorgula(sorgu_1)
    print(f"\n  [Cypher-1] {sorgu_1}")
    for s in sonuc_1:
        print(f"    -> Bulunan İlişki: ({s['kaynak']}) ──[{s['iliski']}]──► ({s['hedef']})")

    # 2-Hop Zincirleme Sorgusu
    sorgu_2 = "MATCH (a)-[r1]->(b)-[r2]->(c) WHERE a.id = 'Vision Transformer' RETURN c"
    sonuc_2 = motor.sorgula(sorgu_2)
    print(f"\n  [Cypher-2 (2-Hop)] {sorgu_2}")
    for s in sonuc_2:
        print(f"    -> Akıl Yürütme Yolu: {s['yol']}")

    # -------------------------------------------------------------
    # ADIM 3: Graf Gezintisi & En Kısa Yol (Multi-Hop Traversal)
    # -------------------------------------------------------------
    print("\n[3/4] Multi-Hop BFS Gezintisi ve LLM Alt-Grafik Bağlamı Üretiliyor...")
    gezgini = GrafGezgini(depo)

    # En Kısa Yol Akıl Yürütme Zinciri
    baslangic = "Vision Transformer"
    hedef = "NVIDIA GPU"
    en_kisa_yol = gezgini.en_kisa_yol(baslangic, hedef)
    print(f"\n  • [{baslangic}] -> [{hedef}] En Kısa Akıl Yürütme Zinciri (Shortest Path):")
    print(f"    {' ──► '.join(en_kisa_yol or [])}")

    # Alt-Grafik LLM Prompt Bağlamı
    altgraf_baglami = gezgini.altgraf_baglami_olustur(baslangic, max_derinlik=2)
    print(f"\n  --- [LLM İçin Üretilen Yapısal Alt-Grafik Bağlamı] ---")
    print(altgraf_baglami)

    # -------------------------------------------------------------
    # ADIM 4: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli GraphRAG-2 Teşhis Panosu Çiziliyor...")
    bench = gezgini.benchmark_karsilastir()
    altgraf = gezgini.k_hop_komsuluk(baslangic, max_derinlik=2)

    gorsellestirici = CypherGraphGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "knowledge_graph_cypher_paneli.png")
    gorsellestirici.pano_olustur(
        karsilastirma=bench,
        en_kisa_yol=en_kisa_yol or [],
        altgraf=altgraf,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 137: GRAPHRAG-2 KNOWLEDGE GRAPH & CYPHER ENGINE BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
