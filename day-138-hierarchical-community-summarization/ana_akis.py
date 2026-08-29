"""
Day 138: GraphRAG-3: Leiden Topluluk Tespiti ve Hiyerarşik Küme Özetleme Ana Akışı.
Microsoft GraphRAG mimarisi ile küresel anlamlandırma (Global Sensemaking & Map-Reduce).
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.leiden_topluluk_tespiti import LeidenToplulukDedektoru
from src.hiyerarsik_ozetleyici import HiyerarsikOzetleyici
from src.kuresel_arama_motoru import KureselAramaMotoru
from src.gorsellestirici import CommunitySummarizationGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 138: GraphRAG-3 - Leiden Hierarchical Community Detection & Summarization (Microsoft GraphRAG)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # ADIM 1: Bilgi Grafı Düğümleri ve Kenarları
    # -------------------------------------------------------------
    print("\n[1/4] Bilgi Grafı Varlıkları ve İlişkileri Yükleniyor...")
    dugumler = [
        "Vision Transformer", "Self-Attention", "FlashAttention", "NVIDIA GPU",
        "Raft", "Quorum", "PostgreSQL", "B-Tree",
        "Limit Order Book", "FPGA"
    ]
    kenarlar = [
        ("Vision Transformer", "Self-Attention", 1.0),
        ("Self-Attention", "FlashAttention", 1.0),
        ("FlashAttention", "NVIDIA GPU", 1.0),
        ("Raft", "Quorum", 1.0),
        ("PostgreSQL", "B-Tree", 1.0),
        ("Limit Order Book", "FPGA", 1.0),
    ]
    dugum_detaylari = {
        "Vision Transformer": "Görüntüleri piksel yamalarıyla işleyen derin öğrenme modeli.",
        "Self-Attention": "QKV tensörleriyle çalışan küresel dikkat katmanı.",
        "FlashAttention": "SRAM bellek optimizasyonu ile tam dikkat hesaplayan kernel.",
        "NVIDIA GPU": "Yüksek bellek bant genişliğine sahip donanım hızlandırıcı.",
        "Raft": "Dağıtık sistemlerde lider seçimi ve durum çoğaltması protokolü.",
        "Quorum": "Ağ bölünmesinde çoğunluk kararı ile tutarlılık sağlayan kural.",
        "PostgreSQL": "İlişkisel veritabanı yönetim sistemi.",
        "B-Tree": "Logaritmik arama ve aralık sorgusu sunan dengeli ağaç indeksi.",
        "Limit Order Book": "Alım-satım emirlerini fiyat/zaman önceliğiyle sıralayan defter.",
        "FPGA": "Mikrosaniye altı gecikmeyle emir eşleştiren donanım.",
    }

    # -------------------------------------------------------------
    # ADIM 2: Leiden Hiyerarşik Topluluk Tespiti
    # -------------------------------------------------------------
    print("\n[2/4] Leiden Algoritması ile Hiyerarşik Topluluklar Tespit Ediliyor...")
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)
    modulerlik = LeidenToplulukDedektoru.modulerlik_hesapla(hiyerarsi[1], len(kenarlar))

    print(f"  [✓] Seviye-1 (Meso Alt-Alan) Topluluk Sayısı : {len(hiyerarsi[1])}")
    print(f"  [✓] Seviye-2 (Macro Kök) Topluluk Sayısı     : {len(hiyerarsi[2])}")
    print(f"  [✓] Graf Modülerlik Skoru (Modularity Q)     : {modulerlik:.3f}")

    for t in hiyerarsi[1]:
        print(f"    • [{t.id}] {t.alan_adi} -> Düğümler: {', '.join(t.dugumler)}")

    # -------------------------------------------------------------
    # ADIM 3: Bottom-Up Hiyerarşik Topluluk Raporları Üretimi
    # -------------------------------------------------------------
    print("\n[3/4] Hiyerarşik Topluluk Özet Raporları Üretiliyor (Community Reports)...")
    raporlar = HiyerarsikOzetleyici.raporlari_uret(hiyerarsi, dugum_detaylari)

    for r_id, r in raporlar.items():
        print(f"\n  --- {r.baslik} (Yapısal Ağırlık: {r.yapısal_agirlik:.1f}) ---")
        print(f"  Özet: {r.ozet[:110]}...")

    # -------------------------------------------------------------
    # ADIM 4: Map-Reduce Küresel Arama ve Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[4/4] Map-Reduce Küresel Soru İcra Ediliyor (Global Search Q&A)...")
    kuresel_soru = "Sistemdeki genel yapay zeka çıkarımı, dağıtık veri tutarlılığı ve donanım hızlandırma mimarisi nasıl çalışır?"
    arama_motoru = KureselAramaMotoru(raporlar)
    sorgu_sonucu = arama_motoru.kuresel_sorgula(kuresel_soru)

    print(f"\n  [KÜRESEL SORU]: '{kuresel_soru}'")
    print(f"  [Sorgu Süresi]: {sorgu_sonucu['sorgu_suresi_ms']:.2f} ms")
    print(f"\n  --- [NİHAİ KÜRESEL SENTEZ YANITI] ---")
    print(sorgu_sonucu["nihai_kuresel_yanit"])

    bench = arama_motoru.benchmark_karsilastir()
    gorsellestirici = CommunitySummarizationGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "community_summarization_paneli.png")
    gorsellestirici.pano_olustur(
        karsilastirma=bench,
        hiyerarsi=hiyerarsi,
        sorgu_sonucu=sorgu_sonucu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 138: GRAPHRAG-3 COMMUNITY SUMMARIZATION BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
