"""
Bellek Yöneticisi ve Hibrit Erişim Motoru Modülü (Day 126 - Faz 7).
Mem0/Zep tarzı olgu çıkarma, çelişki giderme (ADD/UPDATE/NOOP) ve Ebbinghaus unutma eğrili hibrit erişim.
"""

import time
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .bellek_katmanlari import BellekTipi, BellekKaydi, CalismaBellegi, EpisodikBellek, SemantikBellek


class BellekYoneticisi:
    """Çok katmanlı bellek yönetim, çıkarma, güncelleme ve hibrit erişim sistemi."""

    def __init__(
        self,
        vektor_boyutu: int = 128,
        benzerlik_agirligi: float = 0.6,
        tazelik_agirligi: float = 0.2,
        onem_agirligi: float = 0.2,
        unutma_katsayisi: float = 0.001,  # Lambda
    ):
        self.vektor_boyutu = vektor_boyutu
        self.w_sim = benzerlik_agirligi
        self.w_rec = tazelik_agirligi
        self.w_imp = onem_agirligi
        self.lambda_decay = unutma_katsayisi

        self.calisma = CalismaBellegi(kapasite=5)
        self.episodik = EpisodikBellek()
        self.semantik = SemantikBellek()
        self.durdurma_kelimeleri = {"olarak", "için", "ve", "bir", "ile", "bu", "de", "da", "mi", "mu", "kullanıyorum", "tercih", "ediyorum", "hangi"}

    def _metin_vektorlestir(self, metin: str) -> np.ndarray:
        """Deterministik ve anlamsal alt kelime n-gram hash tabanlı gömme vektörü üretir."""
        v = np.zeros(self.vektor_boyutu, dtype=np.float32)
        temiz = metin.lower()
        kelimeler = re.findall(r"\w+", temiz)
        for kelime in kelimeler:
            agirlik = 0.5 if kelime in self.durdurma_kelimeleri else 4.0
            h = int(abs(hash(kelime))) % self.vektor_boyutu
            v[h] += agirlik
            for i in range(max(1, len(kelime) - 2)):
                gram = kelime[i : i + 3]
                h_gram = int(abs(hash(gram))) % self.vektor_boyutu
                v[h_gram] += agirlik * 0.5
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v /= norm
        return v

    def olgu_cikar_ve_kaydet(self, kullanici_mesaji: str) -> List[Dict[str, Any]]:
        """
        Kullanıcı mesajındaki tercih, olgu ve kısıtları çıkarır, çelişki denetimi yaparak kaydeder.
        """
        islem_raporu = []

        # Basit kural tabanlı olgu çıkarma simülasyonu (Mem0 LLM Extractor dengi)
        ilgili_ifadeler = []
        if "kullanıyorum" in kullanici_mesaji.lower() or "tercih ediyorum" in kullanici_mesaji.lower():
            ilgili_ifadeler.append((kullanici_mesaji, 8.0, ["tercih", "arac"]))
        elif "çalışıyorum" in kullanici_mesaji.lower() or "uzmanıyım" in kullanici_mesaji.lower():
            ilgili_ifadeler.append((kullanici_mesaji, 7.5, ["profil", "rol"]))
        elif "asla" in kullanici_mesaji.lower() or "istemiyorum" in kullanici_mesaji.lower():
            ilgili_ifadeler.append((kullanici_mesaji, 9.0, ["kisit", "guvenlik"]))
        else:
            ilgili_ifadeler.append((kullanici_mesaji, 5.0, ["genel"]))

        for metin, onem, etiketler in ilgili_ifadeler:
            vektor = self._metin_vektorlestir(metin)
            aktif_kayitlar = self.semantik.aktif_kayitlar()

            en_benzer_kayit: Optional[BellekKaydi] = None
            en_yuksek_benzerlik = -1.0

            for k in aktif_kayitlar:
                sim = float(np.dot(vektor, k.vektor))
                if sim > en_yuksek_benzerlik:
                    en_yuksek_benzerlik = sim
                    en_benzer_kayit = k

            # 1. Çelişki / Güncelleme Kararı
            if en_benzer_kayit and en_yuksek_benzerlik > 0.90:
                islem = "NOOP (Tekrar Eden Olgu)"
            elif en_benzer_kayit and en_yuksek_benzerlik > 0.72 and ("yerine" in metin.lower() or "artık" in metin.lower()):
                # Eski kaydı güncelle (Eski kaydı geçersiz kıl, yenisini ekle)
                self.semantik.gecersiz_kil(en_benzer_kayit.id)
                yeni_kayit = BellekKaydi(metin=metin, vektor=vektor, tip=BellekTipi.SEMANTIK, onem_puani=onem, etiketler=etiketler)
                self.semantik.ekle_veya_guncelle(yeni_kayit)
                islem = f"UPDATE (Eski ID: {en_benzer_kayit.id} güncellendi)"
            else:
                yeni_kayit = BellekKaydi(metin=metin, vektor=vektor, tip=BellekTipi.SEMANTIK, onem_puani=onem, etiketler=etiketler)
                self.semantik.ekle_veya_guncelle(yeni_kayit)
                islem = f"ADD (Yeni Bellek ID: {yeni_kayit.id})"

            islem_raporu.append({"metin": metin, "islem": islem, "benzerlik": round(en_yuksek_benzerlik, 3)})

        return islem_raporu

    def hibrit_arama(self, sorgu: str, top_k: int = 3) -> List[Tuple[BellekKaydi, float]]:
        """
        Sorgu ile semantik bellek kayıtlarını Benzerlik + Tazelik + Önem bileşimiyle sıralar.
        Puan = w_sim * Sim + w_rec * Recency + w_imp * Importance
        """
        sorgu_vektoru = self._metin_vektorlestir(sorgu)
        aktif_kayitlar = self.semantik.aktif_kayitlar()
        simdiki_zaman = time.time()

        puanlanmis: List[Tuple[BellekKaydi, float]] = []

        for k in aktif_kayitlar:
            # 1. Anlamsal Benzerlik
            sim = float(np.dot(sorgu_vektoru, k.vektor))
            sim = max(0.0, sim)

            # 2. Tazelik (Ebbinghaus Decay: e^(-lambda * dt))
            gecen_zaman = simdiki_zaman - k.son_erisim_zamani
            tazelik = float(np.exp(-self.lambda_decay * gecen_zaman))

            # 3. Önem Puanı Normalize (0 - 1)
            onem = k.onem_puani / 10.0

            # Hibrit Skor
            toplam_skor = (self.w_sim * sim) + (self.w_rec * tazelik) + (self.w_imp * onem)
            puanlanmis.append((k, toplam_skor))

        # Puana göre azalan sırala
        puanlanmis.sort(key=lambda x: x[1], reverse=True)
        secilenler = puanlanmis[:top_k]

        # Erişilen kayıtların zaman damgasını güncelle
        for k, _ in secilenler:
            k.erisim_kaydet()

        return secilenler
