"""
Day 40: Tekstil ve Üretim Teknik Dokümanları Üzerinde Sektörel RAG Sistemi Ana Yürütme Betiği.
"""

import os
from src.sektor_korpusu import TEKSTIL_TEKNIK_KORPUS
from src.semantik_parcalayici import SemantikMetinParcalayici
from src.vektor_deposu import TekstilVektorDeposu
from src.rag_asistani import SektorelRAGAsistani
from src.gorsellestirici import SektorelRAGGorsellestirici


def main():
    print("=" * 80)
    print(">>> 40.1: Tekstil Teknik Doküman Korpusu ve Semantik Parçalama (Chunking)")
    print("=" * 80)

    parcalayici = SemantikMetinParcalayici(max_karakter=350, overlap_karakter=50)
    tum_chunklar = parcalayici.korpus_parcala(TEKSTIL_TEKNIK_KORPUS)

    print(f"[+] Orijinal Doküman Sayısı : {len(TEKSTIL_TEKNIK_KORPUS)} Adet")
    print(f"[+] Üretilen Chunk Sayısı   : {len(tum_chunklar)} Adet Semantik Parça")
    for i, ch in enumerate(tum_chunklar[:3]):
        print(f"    - [{ch['chunk_id']}] {ch['alt_baslik']:<35} | {ch['karakter_uzunlugu']} Karakter | Standart: {ch['kaynak_standart']}")

    print("\n" + "=" * 80)
    print(">>> 40.2: Hibrit Dense-Sparse Vektör İndeksi ve Depolama")
    print("=" * 80)

    vektor_deposu = TekstilVektorDeposu(dense_agirligi=0.60, sparse_agirligi=0.40)
    vektor_deposu.indeksle(tum_chunklar)
    print(f"[+] İndeksleme Tamamlandı: {len(vektor_deposu.kelime_sozlugu)} Kelimelik Teknik Sözlük Oluşturuldu.")

    print("\n" + "=" * 80)
    print(">>> 40.3: Sektörel RAG Soru-Cevap, Context Injection ve Halüsinasyon Kontrolü")
    print("=" * 80)

    rag_asistani = SektorelRAGAsistani(vektor_deposu, guven_esigi=0.20)

    # Örnek Sorgu 1: Kurutma Sıcaklığı
    soru_1 = "Akrilik halı apre ve kurutma fiksaj sıcaklığı kaç derece olmalıdır?"
    print(f"[?] SORU 1: '{soru_1}'")
    sonuc_1 = rag_asistani.yanit_uret(soru_1, top_k=2)

    print(f"[+] Durum           : {sonuc_1['durum']}")
    print(f"[+] Güven Skoru     : %{sonuc_1['en_yuksek_skor']*100:.2f}")
    print(f"[+] Doğrulanmış Yanıt:\n{sonuc_1['yanit']}")

    # Örnek Sorgu 2: Alan Dışı Soru (Reject Guardrail Testi)
    soru_2 = "Uzay mekiklerinde kullanılan titanyum alaşım oranı nedir?"
    print(f"\n[?] SORU 2 (Alan Dışı): '{soru_2}'")
    sonuc_2 = rag_asistani.yanit_uret(soru_2, top_k=2)
    print(f"[+] Durum           : {sonuc_2['durum']}")
    print(f"[+] Güven Skoru     : %{sonuc_2['en_yuksek_skor']*100:.2f}")
    print(f"[+] Koruma Yanıtı   : {sonuc_2['yanit']}")

    print("\n" + "=" * 80)
    print(">>> 40.4: 6 Panelli Sektörel RAG Teşhis Panosunun Üretilmesi")
    print("=" * 80)

    cikis_resmi = SektorelRAGGorsellestirici.rag_paneli_ciz(
        arama_sonucu=sonuc_1,
        tum_chunklar=tum_chunklar,
        hedef_path="day-40-carpet-knowledge-rag/ciktilar/sektorel_rag_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 40: TEKSTİL TEKNİK DOKÜMANLARI RAG SİSTEMİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
