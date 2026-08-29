"""
Day 133: HyDE (Hypothetical Document Embeddings) & Sıfır-Atış Soru Zenginleştirme Ana Akışı.
Kullanıcı sorusundan varsayımsal yanıt pasajı üreterek belge manifoldunda hassas getirme.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.hyde_getirici import HyDERAGGetirici
from src.gorsellestirici import HyDEGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 133: HyDE (Hypothetical Document Embeddings) for Zero-Shot Dense Retrieval")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    getirici = HyDERAGGetirici(vektor_boyutu=128)

    # Korpus Belgeleri
    korpus = [
        {
            "doc_id": "DOC_RAFT",
            "metin": (
                "Raft Konsensüs Protokolü: Dağıtık sistemlerde durum makinesi çoğaltması (State Machine Replication) "
                "için lider seçimi, günlük çoğaltma (Log Replication) ve güvenlik mekanizmalarını yönetir. "
                "Lider düğüm ağ bölünmesi (Network Partition) anında çoğunluk (Quorum) kuralı ile bölünmüş beyin (Split-Brain) "
                "durumunu engeller."
            ),
            "kategori": "Dağıtık Sistemler",
        },
        {
            "doc_id": "DOC_VIT",
            "metin": (
                "Vision Transformer (ViT) Mimarisi: 2D görüntüleri 16x16 yamalara (Patches) ayırarak düzleştirir ve "
                "doğrusal projeksiyonla gömme uzayına aktarır. Konumsal kodlama (Positional Encoding) ve çok başlı öz-dikkat "
                "(Multi-Head Self-Attention) ile piksel ilişkilerini küresel olarak modeller."
            ),
            "kategori": "Bilgisayarlı Görü",
        },
        {
            "doc_id": "DOC_HFT",
            "metin": (
                "Yüksek Frekanslı Ticaret (HFT) ve Emir Defteri: Limit Emir Defteri (Limit Order Book - LOB) "
                "alım ve satım emirlerini fiyat-zaman önceliğine göre eşleştirir. Düşük gecikmeli FPGA donanımları ve "
                "piyasa yapıcı algoritmalar likidite sağlayarak spread farkından kâr elde eder."
            ),
            "kategori": "Finansal Teknolojiler",
        },
        {
            "doc_id": "DOC_QUANTUM",
            "metin": (
                "Kuantum Dolanıklığı ve Bell Testi: İki veya daha fazla parçacığın kuantum durumlarının birbirine "
                "bağlı olması durumudur. Bir parçacığın durumunun ölçülmesi, mesafeden bağımsız olarak diğer parçacığın durumunu "
                "anında belirler; Einstein bunu hayaletimsi uzaktan etki olarak nitelemiştir."
            ),
            "kategori": "Kuantum Fiziği",
        },
    ]

    print("\n[1/3] Korpus Belgeleri Vektör Veritabanına İndeksleniyor...")
    getirici.toplu_belge_ekle(korpus)
    print(f"  [✓] Toplam İndekslenen Belge Sayısı: {len(korpus)}")

    # -------------------------------------------------------------
    # ADIM 2: Soru-Belge Asimetrisi Olan Kısa Sorgu
    # -------------------------------------------------------------
    sorgu = "Quorum ve bölünmüş beyin savunması nasıl yapılır?"
    print(f"\n[2/3] Soru-Belge Asimetrisi Taşıyan Sorgu İcra Ediliyor: '{sorgu}'\n")

    # A) Standart Dense Arama
    standart_sonuclar = getirici.standart_arama(sorgu, top_k=2)
    print("--- [A] Standart Dense Arama (E(q) · E(d)) ---")
    for doc in standart_sonuclar:
        print(f"  • {doc['doc_id']} (Benzerlik Skoru: %{doc['skor']*100:.1f}) | {doc['kategori']}")

    # B) HyDE Arama
    hyde_sonuc = getirici.hyde_arama(sorgu, hipotez_sayisi=3, top_k=2)
    print("\n--- [B] HyDE (Sıfır-Atış Hipotez Centroid) Araması (E(d̂) · E(d)) ---")
    for doc in hyde_sonuc["getirilen_belgeler"]:
        print(f"  ✓ {doc['doc_id']} (HyDE Benzerlik Skoru: %{doc['hyde_skor']*100:.1f}) | {doc['kategori']}")

    print("\n--- [C] Üretilen Varsayımsal Hipotez Pasajı Örneği (d̂₁) ---")
    print(f"  \"{hyde_sonuc['hipotezler'][0][:130]}...\"")
    print(f"  [⚡] HyDE Hipotez & Centroid Arama Süresi: {hyde_sonuc['arama_suresi_ms']:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli HyDE Teşhis Panosu Çiziliyor...")
    karsilastirma = getirici.benchmark_karsilastir()

    gorsellestirici = HyDEGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "hyde_embeddings_paneli.png")
    gorsellestirici.pano_olustur(
        hyde_sonucu=hyde_sonuc,
        standart_sonuclar=standart_sonuclar,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 133: HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
