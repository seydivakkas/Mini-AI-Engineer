"""
Self-Play RL ve Sentetik Veri Döngüsü Motoru (Day 210 - FAZ 11).
Kendi Kendine Problem Üretimi, Akıl Yürütme Çözümü ve Dinamik Müfredat Güncellemesi.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import random
import torch
import torch.nn as nn


class SyntheticProblemGenerator:
    """
    Dinamik Zorluk Seviyesine (δ in [1..10]) Göre Sentetik Problem Üreticisi.
    """

    @classmethod
    def problem_uret(cls, zorluk: float = 1.0) -> Dict[str, Any]:
        """
        Zorluk seviyesine göre sentetik matematik veya mantık problemi oluşturur.
        """
        z_int = max(1, min(int(round(zorluk)), 10))

        if z_int <= 3:
            # Seviye 1-3: Temel Aritmetik & Lineer Denklem
            a = random.randint(2, 10 * z_int)
            b = random.randint(1, 15 * z_int)
            x_val = random.randint(1, 10)
            c = a * x_val + b
            soru = f"{a}*x + {b} = {c} denkleminde x kaçtır?"
            cevap = str(x_val)
        elif z_int <= 7:
            # Seviye 4-7: İkinci Dereceden Denklem ve Polinom
            x1 = random.randint(1, 5)
            x2 = random.randint(1, 5)
            b_coeff = -(x1 + x2)
            c_coeff = x1 * x2
            soru = f"x^2 + ({b_coeff})*x + ({c_coeff}) = 0 denkleminin pozitif köklerinin toplamı kaçtır?"
            cevap = str(x1 + x2)
        else:
            # Seviye 8-10: Modüler Aritmetik ve Çok Adımlı Sistemler
            mod = random.choice([7, 11, 13, 17])
            base = random.randint(2, mod - 1)
            exp = random.randint(3, 8)
            cevap = str(pow(base, exp, mod))
            soru = f"{base}^{exp} mod {mod} ifadesinin eşiti kaçtır?"

        return {
            "soru": soru,
            "dogru_cevap": cevap,
            "zorluk": float(zorluk),
            "kategori": f"Seviye_{z_int}_Problem",
        }


class ReasoningSolver:
    """
    Üretilen Problemleri Adım Adım Çözen Akıl Yürütme Modeli.
    Modelin yeteneği (capability θ) ile soru zorluğu (δ) arasındaki lojistik olasılıkla çözülür.
    """

    def __init__(self, yetenek_theta: float = 2.0):
        self.theta = yetenek_theta

    def cozumu_yurut(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Soruyu çözer ve düşünce zinciriyle birlikte yanıtı üretir."""
        delta = problem["zorluk"]
        # Lojistik başarı fonksiyonu: P(başarı) = 1 / (1 + exp(-(theta - delta)))
        fark = self.theta - delta
        p_basari = 1.0 / (1.0 + math.exp(-max(min(fark, 10.0), -10.0)))

        dogru_mu = random.random() < p_basari
        if dogru_mu:
            uretilen_cevap = problem["dogru_cevap"]
            dusunce_yolu = f"<think>\n1. Problem incelendi: {problem['soru']}\n2. Adım adım cebirsel çözüm yapıldı.\n3. Sonuç teyit edildi.\n</think>\nSonuç: {uretilen_cevap}"
        else:
            # Hatalı rastgele cevap
            uretilen_cevap = str(int(problem["dogru_cevap"]) + random.choice([-2, -1, 1, 2])) if problem["dogru_cevap"].isdigit() else "Hata"
            dusunce_yolu = f"<think>\n1. Hatalı varsayım yapıldı.\n</think>\nSonuç: {uretilen_cevap}"

        return {
            "soru": problem["soru"],
            "uretilen_cevap": uretilen_cevap,
            "dusunce_yolu": dusunce_yolu,
            "dogru_mu": dogru_mu,
            "p_basari_tahmini": p_basari,
        }


class SelfPlayReferee:
    """
    Deterministik Hakem: Çözücü ve Üretici için Ödülleri Hesaplar.
    """

    @classmethod
    def odul_hesapla(
        cls,
        problem: Dict[str, Any],
        cozum: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        R_solver = 1.0 (Doğruysa), 0.0 (Yanlışsa)
        R_generator = Zorluk sınırında (%50 başarı bölgesinde) en yüksek ödülü alır.
        """
        dogru_mu = cozum["uretilen_cevap"] == problem["dogru_cevap"]
        r_solver = 1.0 if dogru_mu else 0.0

        # Üretici ödülü: Ne çok kolay ne de imkansız olmalıdır (Hedef başarı ~ %50)
        p_tahmin = cozum["p_basari_tahmini"]
        r_generator = max(0.0, 1.0 - 2.0 * abs(p_tahmin - 0.5))

        return {
            "r_solver": r_solver,
            "r_generator": r_generator,
            "dogru_mu": dogru_mu,
        }


class CurriculumScheduler:
    """
    Dinamik Müfredat Yöneticisi: Zorluğu Sürekli Sınırda (Frontier) Tutar.
    """

    def __init__(self, baslangic_zorluk: float = 1.0):
        self.zorluk = baslangic_zorluk
        self.gecmis_basarilar: List[bool] = []

    def sonucu_kaydet_ve_zorlugu_guncelle(self, dogru_mu: bool) -> float:
        """Pencere içi başarıya göre zorluğu artırır veya azaltır."""
        self.gecmis_basarilar.append(dogru_mu)
        if len(self.gecmis_basarilar) > 10:
            self.gecmis_basarilar.pop(0)

        basari_orani = sum(self.gecmis_basarilar) / len(self.gecmis_basarilar)

        if basari_orani >= 0.75:
            self.zorluk = min(self.zorluk + 0.4, 10.0)
        elif basari_orani <= 0.35:
            self.zorluk = max(self.zorluk - 0.3, 1.0)

        return self.zorluk


class SelfPlayRLTrainer:
    """
    Kendi Kendine Öğrenen Self-Play Döngüsü Yöneticisi.
    """

    def __init__(self, baslangic_yetenek: float = 2.0, ogrenme_hizi: float = 0.1):
        self.solver = ReasoningSolver(yetenek_theta=baslangic_yetenek)
        self.curriculum = CurriculumScheduler(baslangic_zorluk=1.0)
        self.lr = ogrenme_hizi

    def adim_yurut(self) -> Dict[str, Any]:
        """Tek bir Self-Play döngüsü koşturur ve modeli günceller."""
        # 1. Problem üret
        problem = SyntheticProblemGenerator.problem_uret(self.curriculum.zorluk)
        # 2. Çöz
        cozum = self.solver.cozumu_yurut(problem)
        # 3. Ödülleri hesapla
        oduller = SelfPlayReferee.odul_hesapla(problem, cozum)
        # 4. Müfredatı ve çözücü yeteneğini güncelle
        yeni_zorluk = self.curriculum.sonucu_kaydet_ve_zorlugu_guncelle(oduller["dogru_mu"])

        # Çözücü yeteneği pekiştirmeli olarak artar
        if oduller["dogru_mu"]:
            self.solver.theta += self.lr * 0.5

        return {
            "problem": problem,
            "cozum": cozum,
            "oduller": oduller,
            "yeni_zorluk": yeni_zorluk,
            "guncel_yetenek": self.solver.theta,
        }
