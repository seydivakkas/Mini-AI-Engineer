"""
Çok Turlu (Multi-Turn) Diyalog RLHF Motoru (Day 211 - FAZ 11).
Markov Karar Süreci (MDP), Zamansal Kredi Dağılımı ve Uzun Konuşma Tutarlılığı.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import random
import torch
import torch.nn as nn


class DialogueState:
    """Çok Turlu Konuşma Geçmişi ve Durum (s_t) Yöneticisi."""

    def __init__(self, sistem_mesaji: str = "Sen yardımsever ve tutarlı bir AI asistanısın."):
        self.sistem_mesaji = sistem_mesaji
        self.turlar: List[Dict[str, str]] = []

    def tur_ekle(self, rol: str, icerik: str):
        """Konuşma geçmişine yeni bir kullanıcı veya model turu ekler."""
        self.turlar.append({"rol": rol, "icerik": icerik})

    def tam_baglami_getir(self) -> str:
        """Tüm konuşma geçmişini tek bir bağlam metni olarak derler."""
        metin = f"[SİSTEM]: {self.sistem_mesaji}\n"
        for t in self.turlar:
            metin += f"[{t['rol'].upper()}]: {t['icerik']}\n"
        return metin

    def tur_sayisi(self) -> int:
        return len(self.turlar)


class UserSimulator:
    """Gerçekçi Kullanıcı Tepkileri ve Hedef Takibi Üreten Çevre Modeli."""

    def __init__(self, hedef: str = "Veritabanı İndeks Optimizasyonu"):
        self.hedef = hedef
        self.adim = 0
        self.senaryo_sorulari = [
            "PostgreSQL'de yavaş çalışan bir sorgum var, nasıl hızlandırırım?",
            "EXPLAIN ANALYZE çıktısında Seq Scan görüyorum, ne yapmalıyım?",
            "B-Tree index ekledim ama hala yavaş, bileşik (composite) indeks nasıl tanımlanır?",
            "Harika, şimdi sorgu 5ms'ye düştü. Teşekkürler!",
        ]

    def sonraki_kullanici_mesaji(self) -> Optional[str]:
        """Kullanıcının bir sonraki sorusunu veya bitiş mesajını döndürür."""
        if self.adim < len(self.senaryo_sorulari):
            msg = self.senaryo_sorulari[self.adim]
            self.adim += 1
            return msg
        return None

    def hedef_tamamlandi_mi(self) -> bool:
        return self.adim >= len(self.senaryo_sorulari)


class MultiTurnRewardModel:
    """
    Çok Turlu Konuşma Ödül Modeli.
    Adım bazlı tutarlılık ve nihai hedef tamamlama ödülünü hesaplar.
    """

    @classmethod
    def ara_adim_odulu(
        cls,
        gecmis_baglam: str,
        yeni_yanit: str,
        kullanici_sorusu: str,
    ) -> float:
        """
        Her tur için anlık kalite ve tutarlılık ödülü:
        - Tekrara düşme cezası
        - Konuyla ilgililik ve açıklık ödülü
        """
        odul = 0.20

        # Tekrar kontrolü
        if yeni_yanit.lower() in gecmis_baglam.lower():
            odul -= 0.50  # Ciddi tekrara düşme cezası

        # İlgili anahtar kelimeler
        ilgili_kelimeler = ["index", "explain", "b-tree", "composite", "optimizasyon", "sorgu"]
        if any(k in yeni_yanit.lower() for k in ilgili_kelimeler):
            odul += 0.30

        return float(odul)

    @classmethod
    def nihai_hedef_odulu(cls, user_sim: UserSimulator) -> float:
        """Kullanıcının ana hedefine başarıyla ulaşıldığında verilen terminal ödül."""
        return 2.50 if user_sim.hedef_tamamlandi_mi() else 0.0


class TemporalCreditAssigner:
    """
    Zamansal Kredi Dağıtımı (Temporal Credit Assignment).
    G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} ... formülüyle geçmiş turlara getiri atar.
    """

    @classmethod
    def birikimli_getiri_hesapla(
        cls,
        tur_odulleri: List[float],
        gamma: float = 0.95,
    ) -> List[float]:
        """Geriye doğru indirimli birikimli getirileri (G_t) hesaplar."""
        T = len(tur_odulleri)
        getiriler = [0.0] * T
        birikimli = 0.0

        for t in reversed(range(T)):
            birikimli = tur_odulleri[t] + gamma * birikimli
            getiriler[t] = round(birikimli, 4)

        return getiriler


class MultiTurnRLHFTrainer:
    """Çok Turlu Diyalog Simülasyonu ve RLHF Eğiticisi."""

    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma

    def tam_diyalog_yurut(self) -> Dict[str, Any]:
        """Tam bir çok turlu konuşma epizodu koşturur ve getirileri hesaplar."""
        state = DialogueState()
        user = UserSimulator()
        tur_odulleri = []
        diyalog_adimlari = []

        while True:
            u_msg = user.sonraki_kullanici_mesaji()
            if u_msg is None:
                break

            state.tur_ekle("user", u_msg)

            # Model yanıt simülasyonu
            if user.adim == 1:
                a_msg = "Sorgunuzu hızlandırmak için EXPLAIN ANALYZE çıktısını incelemeli ve eksik indeksleri belirlemeliyiz."
            elif user.adim == 2:
                a_msg = "Seq Scan tablonun baştan sona tarandığını gösterir. Sık filtrelenen sütuna B-Tree indeksi eklemelisiniz."
            elif user.adim == 3:
                a_msg = "Birden fazla sütun WHERE koşulunda birlikte kullanılıyorsa CREATE INDEX idx_adi ON tablo (col1, col2) ile composite index oluşturun."
            else:
                a_msg = "Rica ederim! Sorgu süresinin 5ms'ye inmesi harika bir sonuç. Başka bir sorunuz olursa buradayım."

            state.tur_ekle("assistant", a_msg)

            # Ara ödül hesapla
            r_t = MultiTurnRewardModel.ara_adim_odulu(state.tam_baglami_getir(), a_msg, u_msg)
            tur_odulleri.append(r_t)
            diyalog_adimlari.append({"tur": user.adim, "user": u_msg, "assistant": a_msg, "ara_odul": r_t})

        # Nihai terminal ödülü son tura ekle
        r_terminal = MultiTurnRewardModel.nihai_hedef_odulu(user)
        if tur_odulleri:
            tur_odulleri[-1] += r_terminal

        # Zamansal kredi dağıtımı
        indirimli_getiriler = TemporalCreditAssigner.birikimli_getiri_hesapla(tur_odulleri, gamma=self.gamma)

        return {
            "toplam_tur": len(diyalog_adimlari),
            "diyalog_adimlari": diyalog_adimlari,
            "tur_odulleri": tur_odulleri,
            "indirimli_getiriler": indirimli_getiriler,
            "terminal_odul": r_terminal,
            "hedef_basarildi_mi": user.hedef_tamamlandi_mi(),
        }
