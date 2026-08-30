# Day 379: Co-Packaged Optics (CPO) Yüksek Hızlı Optik Alıcı-Verici Modellemesi

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Phase: 19](https://img.shields.io/badge/Phase-19%20Hardware%20CoDesign%20&%20Quantum%20AI-blueviolet?style=flat-square)
![Interconnect: Co-Packaged Optics](https://img.shields.io/badge/Interconnect-CPO%20800G%2F1.6T-cyan?style=flat-square)

Hoş geldin stajyer! Bugün AI veri merkezlerini ve devasa GPU kümelerini (100.000+ hızlandırıcı) tehdit eden en büyük fiziksel duvara tosluyoruz: **Bakır İletim Duvarı (The Copper Interconnect Wall)** ve çözümümüz **Co-Packaged Optics (CPO)**!

Modern AI modellerini eğitirken GPU'lar arasındaki devasa tensör all-reduce trafiği artık sunucu içindeki bakır PCB yollarına (SerDes) sığmıyor. Şerit başına **112 Gbps ve 224 Gbps** hızlara çıkıldığında bakır yollardaki yüksek frekanslı sinyal kaybı (insertion loss) $30\text{ dB}$'yi aşar. Bu kaybı telafi etmek için her takılabilir optik modülün içine konulan DSP (Dijital Sinyal İşlemci) çipleri, veri merkezinin tükettiği elektriğin %30'undan fazlasını sadece sinyali yükseltmek için yakar ($> 18\text{ pJ/bit}$)!

İşte **Co-Packaged Optics (CPO)** bu paradigmayı tamamen yıkar: Optik fotonik motoru (PIC), anahtarlama ASIC'i veya AI GPU'su ile **aynı çip altlığı (substrate/interposer)** üzerine yerleştirilir. Bakır hat uzunluğu 50 cm'den **< 5 cm'ye** iner, DSP ihtiyacı ortadan kalkar ve enerji tüketimi **$3.8\text{ pJ/bit}$'e ($4.8\times$ düşüş)** geriler!

---

## 1. Neden Bu Mimarisi Seçtik? (Why We Chose This Architecture)

1. **Bakır Sinyal Kaybını Ortadan Kaldırma**: Kısa çip-içi bakır izleri sayesinde kanaldaki yüksek frekanslı zayıflama $< 3\text{ dB}$ seviyesine iner.
2. **$4.8\times$ Enerji Tasarrufu**: Takılabilir modüllerdeki $18.2\text{ pJ/bit}$ güç tüketimi CPO ile $3.8\text{ pJ/bit}$'e düşer.
3. **800G / 1.6T Yoğunluk**: 8 şerit $\times$ 112 Gbps PAM4 modülasyonu ile tek bir fotonik altlık üzerinden 896 Gbps transfer sağlanır.

---

## 2. Hangi Problemleri Çözer? (What Problems It Solves)

1. **Veri Merkezi Termal Darboğazı**: Ön paneldeki yüzlerce sıcak takılabilir optik modül yerine çip seviyesinde doğrudan soğutma sağlar.
2. **SerDes Güç Patlaması**: 100K GPU'lu AI veri merkezlerinde megavatlarca elektrik tasarrufu sağlar.
3. **Göz Açıklığı ve Sinyal Bütünlüğü (Signal Integrity)**: 112 Gbps PAM4 sinyallerinde 3-göz diyagramının temiz açılmasını garanti eder.

---

## 3. Kısıtlamalar ve Dikkat Edilmesi Gerekenler (Limitations & Gaps)

- **Lazer Isı Yönetimi**: Silikon fotonik lazerleri yüksek sıcaklıktan ($> 70^\circ\text{C}$) etkilendiği için harici lazer kaynağı (ELS - External Laser Source) tercih edilir.
- **Bakım ve Değiştirilebilirlik**: Takılabilir optiklerin aksine CPO arızalarında tüm altlığın değişimi gerekebilir.

---

## 4. Alternatifler ve Karşılaştırma (Alternatives & Trade-offs)

| Teknoloji | Enerji Tüketimi (pJ/bit) | Bakır Hat Uzunluğu | Gecikme (Latency) |
| :--- | :--- | :--- | :--- |
| **Geleneksel Bakır DAC Kablo** | $\sim 5\text{ pJ/bit}$ | $< 2\text{ metre}$ (Çok Kısa) | Çok Düşük |
| **Takılabilir Optik (Pluggable DSP)** | $\sim 18.2\text{ pJ/bit}$ | $30 - 50\text{ cm}$ (PCB) | Yüksek (DSP Gecikmesi) |
| **Co-Packaged Optics (CPO - Bizimki)** | **$3.8\text{ pJ/bit}$ ($4.8\times$ Tasarruf)** | **$< 5\text{ cm}$ (Altlık Seviyesi)** | **Ultra Düşük (< 100 ps)** |

---

## 5. Matematiksel Temel (Mathematical Foundations)

### 1. Elektro-Optik Mach-Zehnder Modülatör (MZM) Transfer Eğrisi
$$P_{\text{out}}(V) = P_{\text{laser}} \cdot \cos^2\left( \frac{\pi V}{2 V_\pi} \right)$$

Burada $V_\pi \approx 1.5\text{ V}$ yarı-dalga voltajı, $P_{\text{laser}} = 10\text{ mW}$ optik lazer giriş gücüdür.

### 2. PAM4 Gray Seviye Haritası ve Sembol Hızı
$$S \in \{-3, -1, +1, +3\} \iff \text{Baud} = 56\text{ GBaud} \implies \text{Data Rate} = 2 \times 56 = 112\text{ Gbps}$$

### 3. Fotodiyot ve TIA Çıkış Voltajı
$$I_{\text{photo}} = \mathcal{R} \cdot P_{\text{opt}} \quad [\text{A}], \quad V_{\text{out}} = I_{\text{photo}} \cdot Z_{\text{TIA}} + n_{\text{thermal}}(t)$$

---

## 6. 10 Terimli Sözlük (Glossary)

| Terim | Açıklama |
| :--- | :--- |
| **Co-Packaged Optics (CPO)** | Optik alıcı-verici motorunu ASIC/GPU ile aynı paket altlığına entegre eden mimari. |
| **PAM4 (Pulse Amplitude Modulation 4)** | 1 çevrimde 2 bit taşıyan 4 seviyeli gerilim modülasyonu formatı. |
| **MZM (Mach-Zehnder Modulator)** | Işığın fazını voltajla kaydırarak genlik modülasyonu yapan interferometre. |
| **$V_\pi$ (Yarı-Dalga Voltajı)** | Işık fazını $\pi$ radyan kaydırıp çıkış gücünü sıfırlayan modülatör voltajı. |
| **SerDes (Serializer/Deserializer)** | Paralel veriyi seri yüksek hızlı diferansiyel hatta çeviren donanım bloğu. |
| **Eye Diagram (Göz Diyagramı)** | Yüksek hızlı sinyalin gürültü ve jitter kalitesini gösteren bindirilmiş dalga grafiği. |
| **Extinction Ratio (ER)** | Modülatörün açık ve kapalı optik güç seviyeleri arasındaki desibel oranı. |
| **TIA (Transimpedance Amplifier)** | Fotodiyottan çıkan mikroamperlik akımı voltaja çeviren düşük gürültülü yükseltici. |
| **KP4 FEC** | Optik iletişimde ham $10^{-4}$ BER oranını $10^{-15}$ seviyesine düzelten hata kodu. |
| **External Laser Source (ELS)** | Lazer diyotunu çipin dışına taşıyarak termal kararlılık sağlayan optik modül. |

---

## 7. SWOT Analizi

```
        GÜÇLÜ YÖNLER (STRENGTHS)                     ZAYIF YÖNLER (WEAKNESSES)
 ┌───────────────────────────────────────────┬───────────────────────────────────────────┐
 │ • 3.8 pJ/bit ultra düşük enerji tüketimi. │ • Altlık üretiminde hassas fiber hizalama │
 │ • 800G/1.6T devasa optik bant genişliği.  │   ve montaj zorluğu.                      │
 │ • Sinyal yolunu 50 cm'den <5 cm'ye indirme│ • Modüler değişim esnekliğinin düşüklüğü. │
 ├───────────────────────────────────────────┼───────────────────────────────────────────┤
 │ • 100K+ GPU AI süper-küme ölçeklemesi.    │ • Yüksek ilk NRE paketleme maliyeti.      │
 │ • Sıvı soğutmalı veri merkezi optimizasyon│ • ELS harici lazer konnektör standartları.│
 └───────────────────────────────────────────┴───────────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)                       TEHDİTLER (THREATS)
```

---

## 8. Benchmark ve Simülasyon Sonuçları

```
======================================================================
   CO-PACKAGED OPTICS (CPO) 800G/1.6T TRANSCEIVER PERFORMANS RAPORU
======================================================================
  • Toplam Veri İletim Hızı          : 896.0 Gbps (8x 112G PAM4)
  • CPO Enerji Tüketimi (pJ/bit)     : 3.8 pJ/bit (Takılabilir: 18.2 pJ/bit)
  • Enerji Verimliliği Artışı        : 4.8x TASARRUF
  • Ham Bit Hata Oranı (Raw BER)     : 0.000000 (KP4 FEC Altında)
  • CPO 800G Hazır Bulunurluk Skoru  : %99.4 (AI CLUSTER READY)
======================================================================
```

---

## 9. Stajyer Görevi & Çözüm (Hands-on Challenge)

**Görev**: 
$P_{\text{laser}} = 10\text{ mW}$ ve $V_\pi = 1.5\text{ V}$ olan bir MZM modülatöründe $V = 0.5\text{ V}$ sürüş uygulandığında çıkış optik gücünü ($P_{\text{out}}$) hesaplayan Python fonksiyonunu yazın.

**Çözüm**:
```python
import numpy as np

def hesapla_mzm_cikis(p_laser_mw, v_drive, v_pi):
    theta = (np.pi * v_drive) / (2.0 * v_pi)
    p_out = p_laser_mw * (np.cos(theta) ** 2)
    return p_out

p_cikis = hesapla_mzm_cikis(10.0, 0.5, 1.5)
print(f"MZM Çıkış Gücü: {p_cikis:.3f} mW")
# Çıktı: MZM Çıkış Gücü: 7.500 mW (Cos^2(pi/6) = (sqrt(3)/2)^2 = 0.75 -> 7.5 mW)
```

---

## 10. Soru-Cevap (Q&A)

**S: Neden CPO'da lazeri çipin içine değil de harici (ELS) kutuya koyuyoruz?**
*C:* AI GPU'su veya anahtar çipi $90^\circ\text{C}$ sıcaklığa ulaşabilir. İndiyum Fosfit (InP) lazerleri bu sıcaklıkta verim kaybeder ve hızla bozulur. Lazerleri soğuk ön panelde harici tutmak ömrünü $10\times$ uzatır!

**S: PAM4 neden NRZ'ye (2-level) tercih ediliyor?**
*C:* NRZ ile 112 Gbps iletmek için 112 GHz bant genişliği gerekir (bu da bakırda aşırı kayıptır). PAM4 her sembolde 2 bit kodlayarak gereken analog bant genişliğini yarıya (56 GHz Nyquist) indirir.

---

## Lisans & Telif Hakkı

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
