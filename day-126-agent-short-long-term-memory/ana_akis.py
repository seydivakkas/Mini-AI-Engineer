"""
Day 126: Multi-Tier Agent Memory Systems (Working, Episodic & Semantic Vector Memory) Ana Akışı.
Mem0 tarzı olgu çıkarma, çelişki çözümleme (ADD/UPDATE), Ebbinghaus unutma eğrisi ve kişiselleştirilmiş hatırlama.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.bellek_ajani import HafizaliAjan
from src.gorsellestirici import BellekGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 126: Multi-Tier Agent Memory Systems (Mem0 / Zep Memory Architecture)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # ADIM 1: Çok Turlu Konuşma ve Bellek Evrimi Simülasyonu
    # -------------------------------------------------------------
    print("\n[1/3] Hafızalı Otonom Ajan (Memory-Augmented Agent) Başlatılıyor...")
    ajan = HafizaliAjan()

    turlar = [
        # Tur 1: Kullanıcı profili ve ilk tercihler
        "Projelerimde derin öğrenme için PyTorch ve CUDA kullanıyorum, veritabanı olarak PostgreSQL tercih ediyorum.",
        # Tur 2: İlk tercihe dayalı soru istemi
        "Bana yeni bir görüntü sınıflandırma modeli mimarisi kur.",
        # Tur 3: Tercih Güncellemesi / Çelişki Bildirimi
        "Artık PyTorch yerine JAX ve TPU altyapısını tercih ediyorum.",
        # Tur 4: 10 tur sonra uzun vadeli hatırlama testi
        "Hangi makine öğrenimi kütüphanesini ve hızlandırıcıyı kullanıyordum?",
    ]

    print("\n" + "=" * 95)
    print("                 🧠 ÇOK TURLU KİŞİSELLEŞTİRİLMİŞ KONUŞMA VE BELLEK EVRİMİ                 ")
    print("=" * 95)

    son_rapor = None
    for i, tur_mesaji in enumerate(turlar, 1):
        print(f"\n>>> TUR {i} | KULLANICI: '{tur_mesaji}'")
        rapor = ajan.mesaj_isle(tur_mesaji)
        son_rapor = rapor

        print("  [🔍] Bellek İşlemleri:")
        for op in rapor["olgu_islemleri"]:
            print(f"       * {op['islem']} (Benzerlik: {op['benzerlik']}) -> '{op['metin'][:40]}...'")

        if rapor["hatirlanan_anilar"]:
            print("  [💡] Çekilen Uzun Süreli Anılar:")
            for ani in rapor["hatirlanan_anilar"]:
                print(f"       {ani}")

        print(f"  [🤖] Asistan Yanıtı: {rapor['asistan_yaniti']}")
    print("-" * 95)

    # -------------------------------------------------------------
    # ADIM 2: Aktif Bellek Katmanlarının Envanter Durumu
    # -------------------------------------------------------------
    print("\n[2/3] Çok Katmanlı Bellek Durum Raporu:")
    print(f"  * Çalışma Belleği (Working Memory) : {son_rapor['calisma_bellegi_boyutu']} Mesaj")
    print(f"  * Semantik Bellek (Semantic Memory): {son_rapor['semantik_bellek_sayisi']} Aktif Tercih / Olgu")
    print(f"  * Episodik Bellek (Episodic Memory): {son_rapor['episodik_bellek_sayisi']} Oturum Kaydı")

    # -------------------------------------------------------------
    # ADIM 3: Stateless vs Çok Katmanlı Bellek Kıyaslaması ve Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[3/3] Stateless Kayan Pencere vs Çok Katmanlı Vektör Bellek Kıyaslaması Çiziliyor...")
    karsilastirma = ajan.benchmark_karsilastir()

    print("\n" + "=" * 95)
    print(f"{'METRİK':<35} | {'KAYAN PENCERE (%)':<22} | {'ÇOK KATMANLI BELLEK (%)':<26}")
    print("-" * 95)
    for m, s, v in zip(
        karsilastirma["metrikler"],
        karsilastirma["stateless_kayan_pencere"],
        karsilastirma["cok_katmanli_vektor_bellek"],
    ):
        print(f"{m:<35} | %{s:>19.1f} | %{v:>23.1f}")
    print("-" * 95)

    gorsellestirici = BellekGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "agent_memory_paneli.png")
    gorsellestirici.pano_olustur(
        calisma_ozeti=son_rapor,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 126: ÇOK KATMANLI AJAN BELLEK SİSTEMLERİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
