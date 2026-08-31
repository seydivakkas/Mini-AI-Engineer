# 🚗 Tesla FSD Otonom Sürüş | Gün 60: Hız Profili Optimizasyonu: Konfor, Enerji Verimliliği ve Trafik Akışı Dengeleme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Speed Profile](https://img.shields.io/badge/Speed%20Profile-Forward--Backward%20Pass-red.svg?style=flat-square)](https://www.tesla.com/)
[![Comfort](https://img.shields.io/badge/Comfort-a__lat%20%3C%3D%202.0%20m%2Fs%C2%B2-blue.svg?style=flat-square)](https://www.sae.org/)
[![Regen](https://img.shields.io/badge/Energy-Regenerative%20Braking%2085%25-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"60. günümüze hoş geldin stajyer!  
> Harika bir 2D yol çizgisi çizmiş olsak bile, eğer araç bu viraja $120\text{ km/h}$ hızla girerse merkezkaç kuvveti ($a_{\text{lat}} = v^2 \kappa$) aracı yoldan fırlatır veya yolcuları dehşete düşürür.  
> Bu nedenle yolun her bir $s$ metresinde hızın ($v(s)$) ne olması gerektiği **Hız Profili Optimizasyonu (Speed Profile Optimization)** ile belirlenir:  
> 1. **Maksimum Güvenli Viraj Hızı ($v = \sqrt{a_{\text{lat}} / \kappa}$):** Eğrilik arttıkça hız fizik yasalarına uygun olarak otomatik düşürülür ($a_{\text{lat}} \le 2.0\text{ m/s}^2$).  
> 2. **İleri Geçiş (Forward Pass - Hızlanma Kısıtı):** Motor tork limitlerini aşmayacak şekilde ($a_{\text{acc}} \le 2.0\text{ m/s}^2$) hız artışı zincirlenir.  
> 3. **Geri Geçiş (Backward Pass - Frenleme Kısıtı):** Viraja yaklaşırken aracın ne zaman ve ne kadar önceden yumuşakça fren yapmaya başlaması gerektiği geriye doğru taranır.  
> 4. **Rejeneratif Frenleme Enerji Verimliliği:** Fren balatalarını aşındırmadan kinetik enerjiyi bataryaya (%85 verimle) geri yükler.  
> Bugün Tesla'nın hem konforlu hem de menzil tasarruflu hız beynini inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Maksimum Güvenli Viraj Hızı Sınırı

$$v_{\text{corner}}(\kappa) = \sqrt{\frac{a_{\text{lat, max}}}{\kappa}}, \quad a_{\text{lat, max}} = 2.0\text{ m/s}^2$$

### 2. İleri Geçiş (Forward Pass: Hızlanma Kısıtı)

$$v_{i+1} = \min\left( v_{\text{limit}, i+1}, \sqrt{v_i^2 + 2 a_{\text{acc, max}} \Delta s} \right)$$

### 3. Geri Geçiş (Backward Pass: Frenleme Kısıtı)

$$v_i = \min\left( v_i, \sqrt{v_{i+1}^2 + 2 a_{\text{dec, max}} \Delta s} \right)$$

### 4. Boyuna İvme ve Rejeneratif Enerji Geri Kazanımı

$$a_{\text{long}, i} = \frac{v_{i+1}^2 - v_i^2}{2 \Delta s}$$

$$E_{\text{regen}} = \eta_{\text{regen}} \cdot \frac{1}{2} m \left( v_{\text{straight}}^2 - v_{\text{corner}}^2 \right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Virajlara güvenli ve sarsıntısız hızla girmek, lastik kaymasını önlemek ve rejeneratif frenlemeyi maksimize ederek menzili uzatmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Viraj İçi Panik Frenleri:** Sürücünün veya otopilotun virajın ortasında sert fren yapmasını engelledi; frenleme virajdan önce tamamlandı.
- **Batarya Menzil Kaybı:** Mekanik sürtünme freni yerine rejenerasyon profilini optimize ederek her yavaşlamada enerji geri kazandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Islak/Karlı Zemin:** Sürtünme katsayısı $\mu$ düştüğünde $a_{\text{lat, max}}$ değerinin adaptif olarak $1.0\text{ m/s}^2$ seviyesine çekilmesi gerekir.
- **Trafik Işıkları ve Dur-Kalk:** İleri-geri geçişe ek olarak dinamik engeller için S-T (Zaman-Mekan) grafiği eklenmelidir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Karesel Programlama (QP) Hız Optimizasyonu:** Dışbükey çözücülerle (OSQP) çözülür; benzer sonuç verir ancak daha fazla bellek kullanır.
- **Sabit Hız Kontrolü (Cruise Control):** Viraj geometrisini dikkate almaz; keskin virajlarda savrulma riski doğurur.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Speed Profile $v(s)$** | Yol boyunca her $s$ metresinde hedeflenen boyuna hız dağılımı eğrisi. |
| **Forward Pass** | Başlangıç noktasından ileriye doğru maksimum hızlanma kapasitesini hesaplayan tarama adımı. |
| **Backward Pass** | Hedef viraj kısıtından geriye doğru erken frenleme gereksinimini hesaplayan tarama adımı. |
| **Lateral Acceleration ($a_{\text{lat}}$)** | Araç viraj dönerken oluşan $v^2 \kappa$ merkezkaç ivmesi. |
| **Longitudinal Acceleration ($a_{\text{long}}$)**| Gaz ve fren pedalıyla sağlanan boyuna hızlanma/yavaşlama ivmesi. |
| **Regenerative Braking** | Elektrik motorunu jeneratör olarak çalıştırıp kinetik enerjiyi bataryaya şarj etme mekanizması. |
| **Curvature ($\kappa$)** | Virajın darlığını belirten yol eğriliği ($1/R$). |
| **Kinetic Energy ($E_k$)** | Aracın hareket enerjisi ($\frac{1}{2} m v^2$). |
| **Comfort Boundary** | Yolcu konforu için kabul edilen $2.0\text{ m/s}^2$ yanal ve boyuna ivme tavanı. |
| **Dynamic Programming (DP)**| Karmaşık hız profili problemini ileri-geri alt problemlere bölerek çözen optimizasyon yöntemi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 fiziksel ve kinematik garantili hız profili    | • Yalnızca statik yol geometrisi kısıtını içerir      |
| • Rejeneratif enerji ile maksimum batarya menzili     | • Önümüzdeki araçların hız değişimlerini anlık        |
| • 25 µs ultra hızlı iki yönlü dinamik tarama         |   güncellemek için sürekli adaptasyon gerekir         |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla FSD Enerji/Menzil Tahmin Algoritmasına        | • Gizli buzlanma veya aşınmış lastiklerde             |
|   milimetrik harita tabanlı hız profili besleme       |   sürtünme limitinin aniden aşılması                  |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Hız Profili Optimizasyonu Akış Şeması

```
[ Yol Eğriliği kappa(s) & Ham Hız Limitleri: v_lim = sqrt(a_lat / kappa) ]
                                    |
                                    v
       [ 1. İleri Geçiş (Forward Pass): v_{i+1} <= sqrt(v_i^2 + 2*a_acc*ds) ]
                                    |
                                    v
       [ 2. Geri Geçiş (Backward Pass): v_i <= sqrt(v_{i+1}^2 + 2*a_dec*ds) ]
                                    |
                                    v
       [ 3. Boyuna ve Yanal İvme Analizi: a_lat <= 2.0 m/s² Onayı ]
                                    |
                                    v
     [ 4. Rejeneratif Frenleme Enerji Tasarrufu (%85 Verimle Bataryaya) ]
                                    |
                                    v
        [ %100 PREMIUM KONFORLU VE ENERJİ VERİMLİ HIZ PROFİLİ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Hız Profili simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
