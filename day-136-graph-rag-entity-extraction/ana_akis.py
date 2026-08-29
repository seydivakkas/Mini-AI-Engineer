"""
Day 136: GraphRAG-1: Metinden Varlık (Entity) ve İlişki (Relationship) Çıkarma Ana Akışı.
Yapılandırılmamış teknik metinlerden yapısal Bilgi Grafı (Knowledge Graph) inşası.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.bilgi_grafi_olusturucu import BilgiGrafiOlusturucu
from src.gorsellestirici import GraphRAGGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 136: GraphRAG-1 - Unstructured Text to Knowledge Graph Construction")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    olusturucu = BilgiGrafiOlusturucu()

    # Çok Konulu ve İlişkisel Teknik Belge
    belge = (
        "Bölüm 1: Dağıtık Sistemlerde Konsensüs. "
        "Raft konsensüs protokolü Quorum kuralını uygular ve durum makinesi çoğaltması yürütür. "
        "Raft protokolü ağ bölünmesi durumunda veri tutarsızlığını engeller. "
        "Bölüm 2: Bilgisayarlı Görü ve Dikkat. "
        "Vision Transformer mimarisi Self-Attention mekanizmasını kullanır. "
        "ViT modelleri görüntü yamalarını gömme uzayına projekte ederek doğrusal projeksiyon uygular. "
        "Bölüm 3: Veritabanı Mimarisi ve İndeksleme. "
        "PostgreSQL ilişkisel veritabanı B-Tree indeksleme desteği sunar ve logaritmik aramayı hızlandırır. "
        "Bölüm 4: Yüksek Frekanslı Finansal Sistemler. "
        "Limit Order Book alım ve satım emirlerini fiyat önceliğiyle yönetir. "
        "FPGA donanım hızlandırıcıları Limit Order Book eşleştirme motorunu hızlandırır."
    )

    print("\n[1/3] Metinden Varlıklar ve İlişki Üçlüleri Çıkarılıyor (Graph Construction)...")
    graf_sonucu = olusturucu.metinden_graf_olustur(belge)

    print(f"  [✓] Toplam Çıkarılan Düğüm (Nodes)  : {graf_sonucu['toplam_dugum_sayisi']}")
    print(f"  [✓] Toplam Çıkarılan Kenar (Edges)  : {graf_sonucu['toplam_kenar_sayisi']}")
    print(f"  [✓] Graf İnşa Süresi                : {graf_sonucu['cikarim_suresi_ms']:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 2: Çıkarılan Düğümleri ve Kenarları Yazdır
    # -------------------------------------------------------------
    print("\n--- [A] Bilgi Grafı Düğümleri (Knowledge Graph Entities) ---")
    print(f"{'DÜĞÜM ADI':<22} | {'TİP':<14} | {'DERECE':<8} | {'AÇIKLAMA'}")
    print("-" * 95)
    for d in graf_sonucu["dugumler"]:
        print(f"{d['isim']:<22} | {d['tip']:<14} | {d['derece']:<8} | {d['aciklama'][:45]}...")

    print("\n--- [B] Bilgi Grafı İlişki Kenarları (Directed Triplets: (S, P, O)) ---")
    for k in graf_sonucu["kenarlar"]:
        print(f"  • ({k['ozne']}) ──[{k['yuklem']} (Ağırlık: {k['agirlik']:.1f})]──► ({k['nesne']})")

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli GraphRAG-1 Teşhis Panosu Çiziliyor...")
    karsilastirma = olusturucu.benchmark_karsilastir()

    gorsellestirici = GraphRAGGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "graph_rag_entity_extraction_paneli.png")
    gorsellestirici.pano_olustur(
        graf_sonucu=graf_sonucu,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 136: GRAPHRAG-1 ENTITY & RELATIONSHIP EXTRACTION BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
