"""
Day 139: Hibrit RAG: Vektör Arama + Bilgi Grafı Gezintisi (Hybrid Retrieval & RRF) Ana Akışı.
Çift kanallı getirme, dinamik sorgu yönlendirici ve Reciprocal Rank Fusion entegrasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.hibrit_rag_yoneticisi import HibritRAGYoneticisi
from src.gorsellestirici import HybridRAGGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 139: Hybrid Vector + Knowledge Graph RAG with Reciprocal Rank Fusion (RRF)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # ADIM 1: Çok Alanlı Belge Havuzu ve Graf Kenarları
    # -------------------------------------------------------------
    print("\n[1/4] Belge Havuzu ve Bilgi Grafı İndeksleniyor...")
    belgeler = [
        {
            "id": "DOC_01_VIT_CORE",
            "metin": "Vision Transformer mimarisi görüntü yamalarını Self-Attention dikkat mekanizması ile işler.",
            "varliklar": ["Vision Transformer", "Self-Attention"],
        },
        {
            "id": "DOC_02_FLASH_OPT",
            "metin": "FlashAttention algoritması GPU SRAM bellek bant genişliğini optimize ederek Self-Attention'ı hızlandırır.",
            "varliklar": ["FlashAttention", "Self-Attention", "NVIDIA GPU"],
        },
        {
            "id": "DOC_03_RAFT_CONS",
            "metin": "Raft konsensüs protokolü dağıtık sistemlerde Quorum çoğunluk kuralını uygular ve lider seçimi yapar.",
            "varliklar": ["Raft", "Quorum"],
        },
        {
            "id": "DOC_04_POSTGRES",
            "metin": "PostgreSQL ilişkisel veritabanı B-Tree ve GIN indeksleme desteği ile hızlı arama sağlar.",
            "varliklar": ["PostgreSQL", "B-Tree"],
        },
        {
            "id": "DOC_05_LOB_FPGA",
            "metin": "Limit Order Book alım-satım emir defteri FPGA donanım hızlandırıcıları ile mikrosaniyede eşleştirilir.",
            "varliklar": ["Limit Order Book", "FPGA"],
        },
    ]

    graf_kenarlari = [
        {"ozne": "Vision Transformer", "yuklem": "KULLANIR", "nesne": "Self-Attention"},
        {"ozne": "Self-Attention", "yuklem": "HIZLANDIRIR", "nesne": "FlashAttention"},
        {"ozne": "FlashAttention", "yuklem": "CALISIR", "nesne": "NVIDIA GPU"},
        {"ozne": "Raft", "yuklem": "UYGULAR", "nesne": "Quorum"},
        {"ozne": "PostgreSQL", "yuklem": "DESTEKLER", "nesne": "B-Tree"},
        {"ozne": "Limit Order Book", "yuklem": "HIZLANDIRIR", "nesne": "FPGA"},
    ]

    yonetici = HibritRAGYoneticisi()
    yonetici.indeksle(belgeler, graf_kenarlari)

    print(f"  [✓] Toplam İndekslenen Belge Sayısı: {len(belgeler)}")
    print(f"  [✓] Toplam İndekslenen Graf Kenarı : {len(graf_kenarlari)}")

    # -------------------------------------------------------------
    # ADIM 2: Çoklu Atlama ve İlişkisel Hibrit Sorgu
    # -------------------------------------------------------------
    sorgu = "Vision Transformer dikkat mekanizmasını GPU üzerinde hızlandıran algoritma nedir ve nasıl bağlanır?"
    print(f"\n[2/4] Hibrit Sorgu İcra Ediliyor: '{sorgu}'")

    sonuc = yonetici.ara(sorgu, top_k=4)

    print(f"  [✓] Tespit Edilen Sorgu Tipi : {sonuc['sorgu_tipi']}")
    print(f"  [✓] Uygulanan Ağırlıklar     : Vektör: {sonuc['agirliklar']['w_vec']}, Graf: {sonuc['agirliklar']['w_graph']}")
    print(f"  [✓] Getirme Gecikmesi        : {sonuc['getirme_suresi_ms']:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 3: RRF Füzyon ve Sıralama Kayması Tablosu
    # -------------------------------------------------------------
    print("\n--- [A] Reciprocal Rank Fusion (RRF) Sonuç Sıralaması ---")
    print(f"{'HİBRİT SIRA':<12} | {'BELGE ID':<18} | {'VEKTÖR SIRA':<12} | {'GRAF SIRA':<10} | {'RRF SKORU':<12} | {'KAYMA'}")
    print("-" * 85)
    for d in sonuc["hibrit_sonuclar"]:
        v_sira = d.get("vektor_sirasi", "-")
        g_sira = d.get("graf_sirasi", "-")
        kayma = f"+{d['siralama_kaymasi']}" if d["siralama_kaymasi"] > 0 else str(d["siralama_kaymasi"])
        print(f"#{d['nihai_sira']:<11} | {d['id']:<18} | #{v_sira:<11} | #{g_sira:<9} | {d['rrf_skoru']:<12.5f} | {kayma}")

    # -------------------------------------------------------------
    # ADIM 4: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Hibrit Vector + Graph RAG Teşhis Panosu Çiziliyor...")
    bench = yonetici.benchmark_karsilastir()

    gorsellestirici = HybridRAGGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "hybrid_vector_graph_rag_paneli.png")
    gorsellestirici.pano_olustur(
        karsilastirma=bench,
        arama_sonucu=sonuc,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 139: HYBRID VECTOR + GRAPH RAG (RRF FUSION) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
