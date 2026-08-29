"""
Day 238: Asenkron Olay Güdümlü Ajan Kuyruğu Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.kuyruk_ajani_motoru import (
    AgentJob,
    DeadLetterQueue,
    AsyncAgentQueue,
)
from src.kuyruk_profilleyici import KuyrukProfilleyici
from src.gorsellestirici import KuyrukGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 238 (FAZ 12): ASENKRON OLAY GÜDÜMLÜ AJAN KUYRUĞU - REDIS/CELERY DAYANIKLI İŞÇİ HAVUZU & DLQ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Kuyruk ve İşçi Havuzu Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] Olay Güdümlü Ajan Kuyruğu ve DLQ Başlatılıyor...")
    queue = AsyncAgentQueue()
    print("  ✓ Redis/Celery Tabanlı Mesaj Kuyruğu ve İşçi Havuzu Hazır.")

    # -------------------------------------------------------------
    # ADIM 2: Asenkron Görevlerin Kuyruğa Eklenmesi (HTTP 202)
    # -------------------------------------------------------------
    print("\n[2/4] İstemciden Gelen Görevler Kuyruğa Ekleniyor (5ms Yanıt)...")
    j1_id = queue.gorev_ekle("web_scraping", {"url": "https://deepmind.google/research", "derinlik": 3})
    j2_id = queue.gorev_ekle("code_repair", {"repo": "mini-ai-engineer", "hata": "Deadlock"}, maks_deneme=2)

    print(f"  • İstemci Yanıtı: HTTP 202 Accepted -> Görev ID: [{j1_id}]")
    print(f"  • İstemci Yanıtı: HTTP 202 Accepted -> Görev ID: [{j2_id}]")

    # -------------------------------------------------------------
    # ADIM 3: Arka Planda Görevlerin İcrası ve DLQ Yönlendirmesi
    # -------------------------------------------------------------
    print("\n[3/4] Arka Plan İşçileri Görevleri Tüketiyor...")

    def worker_logic(job: AgentJob):
        if job.gorev_tipi == "code_repair":
            raise TimeoutError("LLM API 429 Too Many Requests (Rate Limit Aşıldı)")
        return {"durum": "TAMAMLANDI", "kazinan_sayfa_sayisi": 24}

    queue.gorev_calistir(worker_logic)
    queue.gorev_calistir(worker_logic)

    print("\n--- [Kuyruk ve İşçi Günlüğü] ---")
    for log in queue.islem_gunlugu:
        print("  " + log)

    print(f"\n  📦 Ölü Mektup Kuyruğundaki (DLQ) Görev Sayısı: {queue.dlq.boyut()}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Asenkron Kuyruk Teşhis Panosu Oluşturuluyor...")
    profil_raporu = KuyrukProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "kuyruk_ajani_paneli.png")

    KuyrukGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Kuyruk Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 238 (FAZ 12): ASENKRON OLAY GÜDÜMLÜ AJAN KUYRUĞU BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
