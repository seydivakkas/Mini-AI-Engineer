"""
Day 135: Dinamik Bağlam Sıkıştırma (Contextual Compression & Extraction for RAG) Ana Akışı.
Alınan belgelerden alakasız dolgu cümlelerini eleme, token tasarrufu ve temiz LLM girdisi sağlama.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.sikistirilmis_rag_getirici import SikistirilmisRAGGetirici
from src.gorsellestirici import ContextualCompressionGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 135: Contextual Compression & Extraction for High-Signal RAG Context")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    getirici = SikistirilmisRAGGetirici(vektor_boyutu=128, esik_skoru=0.25)

    # Gürültülü ve Dolgu Cümleleri İçeren Gerçekçi Korpus Belgeleri
    korpus = [
        {
            "doc_id": "DOC_01_RAFT",
            "metin": (
                "Raft konsensüs protokolü dağıtık durumlarda lider seçimi ve günlük çoğaltma işlemlerini yönetir. "
                "Lider düğüm ağ bölünmesi anında çoğunluk kuralı (Quorum) ile bölünmüş beyin sorununu engeller. "
                "Şirket içi öğle yemeği molası saat 12:30'da başlayacaktır. "
                "Sunucu odasındaki sıcaklık sensörleri her 10 dakikada bir kontrol edilir."
            ),
        },
        {
            "doc_id": "DOC_02_POSTGRES",
            "metin": (
                "PostgreSQL ilişkisel veritabanında B-Tree indeksleri hızlı eşitlik ve aralık sorguları sunar. "
                "Ofis içi kahve makinelerinin periyodik bakımı bu cuma yapılacaktır. "
                "GIN ve GiST indeksleri ise tam metin ve coğrafi arama yükleri için optimize edilmiştir."
            ),
        },
        {
            "doc_id": "DOC_03_VIT",
            "metin": (
                "Vision Transformer görüntüleri 16x16 piksel yamalarına ayırarak öz-dikkat katmanlarına iletir. "
                "Hafta sonu hava sıcaklıklarının mevsim normallerinin üzerinde seyretmesi bekleniyor. "
                "Konumsal kodlama matrisi görüntüdeki mekânsal piksel ilişkilerini korur."
            ),
        },
    ]

    print("\n[1/3] Gürültülü Belgeler Vektör Veritabanına İndeksleniyor...")
    getirici.toplu_belge_ekle(korpus)
    print(f"  [✓] Toplam İndekslenen Belge Sayısı: {len(korpus)}")

    # -------------------------------------------------------------
    # ADIM 2: Sorgu İcrası ve Cümle Düzeyinde Dinamik Sıkıştırma
    # -------------------------------------------------------------
    sorgu = "Raft protokolü lider seçimi ve quorum çoğunluk kuralı nasıl çalışır?"
    print(f"\n[2/3] Sorgu İcra Ediliyor: '{sorgu}'\n")

    sonuc = getirici.sorgula_ve_sikistir(sorgu, top_k=2)

    print("--- [A] Cümle Düzeyinde Anlamsal Puanlama ve Eleme Sonuçları ---")
    for cumle, skor in sonuc["puanli_cumleler"]:
        durum = "[KORUNDU]" if skor >= sonuc["esik_skoru"] else "[BUDANDI]"
        print(f"  {durum:<10} (Skor: {skor:.4f}) [{cumle.doc_id}] -> \"{cumle.metin[:65]}...\"")

    print("\n--- [B] Bağlam Sıkıştırma ve Token Tasarruf İstatistikleri ---")
    print(f"  • Toplam Ayrıştırılan Cümle : {sonuc['toplam_cumle_sayisi']}")
    print(f"  • Kabul Edilen Yüksek Sinyal: {sonuc['secilen_cumle_sayisi']} Cümle")
    print(f"  • Budanan Alakasız Gürültü  : {sonuc['elenen_cumle_sayisi']} Cümle")
    print(f"  • Ham Prompt Token          : {sonuc['ham_token']} Token")
    print(f"  • Sıkıştırılmış Token       : {sonuc['sikistirilmis_token']} Token")
    print(f"  • NET TOKEN TASARRUFU       : %{sonuc['token_tasarrufu_yuzde']:.2f}")

    print("\n--- [C] LLM'e Gönderilecek Tertemiz Sıkıştırılmış Bağlam ---")
    print(sonuc["nihai_baglam"])

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Contextual Compression Teşhis Panosu Çiziliyor...")
    karsilastirma = getirici.benchmark_karsilastir()

    gorsellestirici = ContextualCompressionGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "contextual_compression_paneli.png")
    gorsellestirici.pano_olustur(
        sikistirma_sonucu=sonuc,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 135: CONTEXTUAL COMPRESSION & EXTRACTION BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
