"""
Evol-Instruct Evrim Operatörleri Çekirdek Modülü (Day 116).
WizardLM tarzı In-Depth (Derinlemesine) ve In-Breadth (Genişlemesine) evrimsel istem dönüşümleri.
"""

from typing import Dict, List, Tuple
import random


class EvolInstructMotoru:
    """Sentetik talimatları evrimleştiren otonom kural ve şablon motoru."""

    KISIT_SABLONLARI = [
        "Lütfen çözümünüzde ek bellek karmaşıklığının O(1) olduğunu kanıtlayın ve yerleşik kütüphaneleri kullanmayın.",
        "Yanıtınızı yalnızca 3 maddelik kesin bir teknik analiz ve bir Python kod bloğu ile sınırlandırın.",
        "Tüm potansiyel uç durumları (edge cases: None, negatif değerler, taşma) ele alan istisna yönetimini dahil edin.",
        "Çözümü hem zaman optimizasyonu hem de mikroişlemci önbellek (cache locality) verimliliği açısından değerlendirin.",
    ]

    DERINLESTIRME_SABLONLARI = [
        "Bu kavramın teorik temellerini, matematiksel ispatını ve kuantum/dağıtık sistemlerdeki izdüşümünü açıklayın.",
        "Yalnızca temel tanımı değil, arka plandaki düşük seviyeli işletim sistemi ve bellek yönetim süreçlerini detaylandırın.",
        "Bu algoritmanın sınırlarını, asimptotik alt sınırını (Omega) ve NP-Zorluk ilişkisini tartışın.",
    ]

    SOMUTLASTIRMA_SABLONLARI = [
        "Örnek olarak: 10 milyon eşzamanlı kullanıcının olduğu bir e-ticaret ödeme ağ geçidini senaryo olarak ele alın.",
        "Somut bir vaka olarak: Yüksek frekanslı borsa alım-satım (HFT) sunucusunda 5 mikrosaniyelik gecikme bütçesiyle uygulayın.",
        "Gerçek dünya vakası: Otonom bir aracın LiDAR sensöründen gelen 30 FPS nokta bulutu verisini işleyen bir mimari kurun.",
    ]

    MUHAKEME_SABLONLARI = [
        "Adım adım akıl yürütün: Önce 3 alternatif hipotez geliştirin, ardından her birini eleyerek nihai çözümü türetin.",
        "Tersine mühendislik mantığıyla yaklaşın: Sonuçtan başlayarak başlangıç koşullarını dedüktif mantıkla kanıtlayın.",
        "Önce sistemin neden başarısız olabileceğini (Failure Mode Analysis) listeleyin, ardından bunu engelleyen yapıyı inşa edin.",
    ]

    MUTASYON_SABLONLARI = [
        "Bu mantığı tamamen farklı bir disipline (örneğin biyoinformatik veya kriptografik protokoller) uyarlayan yeni bir görev türetin.",
        "Bu problemin ikiz (dual) karşılığı olan ters optimizasyon problemini tanımlayın ve çözün.",
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def in_depth_kisit_ekle(self, prompt: str) -> str:
        secilen_kisit = random.choice(self.KISIT_SABLONLARI)
        return f"{prompt.strip()} [KISIT]: {secilen_kisit}"

    def in_depth_derinlestir(self, prompt: str) -> str:
        secilen_derinlik = random.choice(self.DERINLESTIRME_SABLONLARI)
        return f"{prompt.strip()} [DERİNLEŞTİRME]: {secilen_derinlik}"

    def in_depth_somutlastir(self, prompt: str) -> str:
        secilen_somut = random.choice(self.SOMUTLASTIRMA_SABLONLARI)
        return f"{prompt.strip()} [SOMUTLAŞTIRMA]: {secilen_somut}"

    def in_depth_muhakeme_artir(self, prompt: str) -> str:
        secilen_muhakeme = random.choice(self.MUHAKEME_SABLONLARI)
        return f"{prompt.strip()} [MUHAKEME ADIMI]: {secilen_muhakeme}"

    def in_breadth_mutasyon(self, prompt: str) -> str:
        secilen_mutasyon = random.choice(self.MUTASYON_SABLONLARI)
        return f"{prompt.strip()} [MUTASYON ÇEŞİTLİLİK]: {secilen_mutasyon}"

    def evrim_adimi(self, prompt: str, operator: str = "rastgele") -> Tuple[str, str]:
        """Verilen istemi belirtilen veya rastgele seçilen operatörle evrimleştirir."""
        operatorler = {
            "kisit_ekle": self.in_depth_kisit_ekle,
            "derinlestir": self.in_depth_derinlestir,
            "somutlastir": self.in_depth_somutlastir,
            "muhakeme_artir": self.in_depth_muhakeme_artir,
            "mutasyon": self.in_breadth_mutasyon,
        }

        if operator == "rastgele" or operator not in operatorler:
            secilen_op_adi = random.choice(list(operatorler.keys()))
        else:
            secilen_op_adi = operator

        evrilmis_prompt = operatorler[secilen_op_adi](prompt)
        return evrilmis_prompt, secilen_op_adi
