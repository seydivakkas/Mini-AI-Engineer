"""
Day 134: İki Aşamalı Hassas Getirme: Bi-Encoder (Vektör) + Cross-Encoder (Re-ranker) Ana Akışı.
Hızlı aday çıkarımı ve derin çapraz dikkat re-ranking ile getirme optimizasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.iki_asamali_getirici import IkiAsamaliRAGGetirici
from src.gorsellestirici import CrossEncoderGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 134: Two-Stage Precision Retrieval (Bi-Encoder + Cross-Encoder Re-ranking)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    getirici = IkiAsamaliRAGGetirici(vektor_boyutu=128, cross_gizli_boyut=64)

    # Korpus Belgeleri
    korpus = [
        {
            "doc_id": "DOC_01_RAFT_CORE",
            "metin": "Raft protokolü lider seçimi ve günlük çoğaltma mekanizması ile dağıtık veri tutarlılığı sağlar.",
            "metadata": {"kategori": "Dağıtık Sistemler", "onem": "Yuksek"},
        },
        {
            "doc_id": "DOC_02_PAXOS",
            "metin": "Paxos ve Multi-Paxos algoritması lider tabanlı konsensüs ve durum makinesi çoğaltması yürütür.",
            "metadata": {"kategori": "Dağıtık Sistemler", "onem": "Orta"},
        },
        {
            "doc_id": "DOC_03_POSTGRES",
            "metin": "PostgreSQL ilişkisel veritabanlarında B-Tree ve GIN indeksleri sorgu optimizasyonu sunar.",
            "metadata": {"kategori": "Veritabanı", "onem": "Yuksek"},
        },
        {
            "doc_id": "DOC_04_VIT_VISION",
            "metin": "Vision Transformer modelleri görüntü yamalarını öz-dikkat katmanları ile küresel işler.",
            "metadata": {"kategori": "Bilgisayarlı Görü", "onem": "Orta"},
        },
        {
            "doc_id": "DOC_05_HFT_ORDER",
            "metin": "Limit Emir Defteri piyasa alım satım emirlerini fiyat ve zaman önceliğine göre sıralar.",
            "metadata": {"kategori": "Finans", "onem": "Orta"},
        },
        {
            "doc_id": "DOC_06_RAFT_SAFETY",
            "metin": "Raft algoritmasında günlük eşleştirme özelliği ve seçim kısıtları lider güvenliğini garanti eder.",
            "metadata": {"kategori": "Dağıtık Sistemler", "onem": "Kritik"},
        },
    ]

    print("\n[1/3] Korpus Belgeleri Bi-Encoder Veritabanına İndeksleniyor...")
    getirici.toplu_belge_ekle(korpus)
    print(f"  [✓] Toplam İndekslenen Belge Sayısı: {len(korpus)}")

    # -------------------------------------------------------------
    # ADIM 2: İki Aşamalı Getirme ve Re-ranking İcrası
    # -------------------------------------------------------------
    sorgu = "Raft protokolü lider seçimi ve günlük çoğaltma güvenliği nasıl çalışır?"
    print(f"\n[2/3] Sorgu İcra Ediliyor: '{sorgu}'\n")

    sonuc = getirici.getir_ve_yeniden_sirala(sorgu, aday_k=5, nihai_k=3)

    print("--- [A] 1. AŞAMA: Bi-Encoder Hızlı Aday Sıralaması (K=5) ---")
    for doc in sonuc["asama_1_adaylar"]:
        print(f"  #{doc['asama_1_sira']} {doc['doc_id']:<18} | Bi-Encoder Skor: %{doc['bi_encoder_skor']*100:.1f}")

    print("\n--- [B] 2. AŞAMA: Cross-Encoder Derin Re-ranking Sıralaması (Top-3) ---")
    for doc in sonuc["nihai_sonuclar"]:
        shift_str = f"+{doc['sira_degisimi']}" if doc['sira_degisimi'] > 0 else str(doc['sira_degisimi'])
        print(f"  ★ #{doc['asama_2_sira']} {doc['doc_id']:<18} | Cross-Encoder Skor: %{doc['cross_encoder_skor']*100:.1f} | Rank Shift: {shift_str}")

    print(f"\n  [⚡] Süreler: Bi-Encoder: {sonuc['sureler']['bi_encoder_ms']} ms | Cross-Encoder: {sonuc['sureler']['cross_encoder_ms']} ms | Toplam: {sonuc['sureler']['toplam_ms']} ms")

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Cross-Encoder Re-ranking Teşhis Panosu Çiziliyor...")
    karsilastirma = getirici.benchmark_karsilastir()

    _, dikkat_matrisi = getirici.cross_encoder.puanla(sorgu, sonuc["nihai_sonuclar"][0]["metin"])
    s_tokens = getirici.cross_encoder._tokenlestir(sorgu)
    b_tokens = getirici.cross_encoder._tokenlestir(sonuc["nihai_sonuclar"][0]["metin"])

    gorsellestirici = CrossEncoderGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "cross_encoder_reranking_paneli.png")
    gorsellestirici.pano_olustur(
        getirme_sonucu=sonuc,
        dikkat_matrisi=dikkat_matrisi,
        sorgu_tokenlari=s_tokens,
        belge_tokenlari=b_tokens,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 134: TWO-STAGE BI-ENCODER + CROSS-ENCODER RERANKING BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
