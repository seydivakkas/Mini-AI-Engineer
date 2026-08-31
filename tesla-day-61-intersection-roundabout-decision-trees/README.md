# 🚗 Tesla FSD Otonom Sürüş | Gün 61: Karmaşık Şehir İçi Kavşak ve Döner Kavşak (Roundabout) Karar Ağaçları

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Decision Engine](https://img.shields.io/badge/Decision%20Engine-Hierarchical%20FSM-red.svg?style=flat-square)](https://www.tesla.com/)
[![Gap Acceptance](https://img.shields.io/badge/Gap%20Acceptance-TTC%20%3E%3D%203.5s-blue.svg?style=flat-square)](https://www.sae.org/)
[![Roundabout](https://img.shields.io/badge/Roundabout-Right--of--Way%20Rules-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"61. günümüze hoş geldin stajyer!  
> Otoyolda şerit takip etmek göreceli olarak basittir. Ancak şehir içindeki çok kollu kontrolsüz kavşaklar ve döner kavşaklar (Roundabouts), otonom sürüşün en çetin sınavlarından biridir!  
> Burada yalnızca fizik değil, **Trafik Hukuku ve İnsan Davranışı Modellemesi** devrededir:  
> 1. **Geçiş Önceliği Hiyerarşisi (Right-of-Way Rules):** Döner kavşak içindeki araç her zaman önceliklidir; kavşağa giren araç yol verme çizgisinde (Yield Line) beklemelidir.  
> 2. **Time-To-Collision (TTC) & Güvenli Aralık Kabulü (Gap Acceptance):** Kavşak içinde dönen aracın hızı ($v$) ve mesafesi ($d$) üzerinden $TTC = \frac{d}{v}$ hesaplanır. $TTC \ge 3.5\text{ saniye}$ ise kavşağa giriş onaylanır.  
> 3. **Hiyerarşik Sonlu Durum Makinesi (HFSM):** `APPROACHING` $\to$ `YIELDING` $\to$ `ENTERING` $\to$ `CIRCULATING` $\to$ `EXITING` durumları arasında deterministik geçiş yapılır.  
> 4. **Tereddüt Önleme (Anti-Hesitation Logic):** Karar verildikten sonra gereksiz frenleme yapılmaz, akıcı ve kararlı bir şekilde manevra tamamlanır.  
> Bugün Tesla FSD'nin en yoğun döner kavşaklarda bile güvenle akmasını sağlayan karar beynini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Time-To-Collision (TTC) Denklemi

$$TTC = \frac{d_{\text{approaching}}}{v_{\text{rel}}} = \frac{d_{\text{approaching}}}{\max(v_{\text{approaching}}, 0.1)}$$

### 2. Güvenli Aralık Kabul Kriteri (Gap Acceptance Model)

$$\text{Karar} = \begin{cases} \text{GİRİŞ (ENTER)}, & \min_i TTC_i \ge 3.5\text{ Saniye} \\ \text{YOL VER (YIELD)}, & \min_i TTC_i < 3.5\text{ Saniye} \end{cases}$$

### 3. Yol Verme Çizgisine Yaklaşma Fren İvmesi

$$a_{\text{target}} = -\frac{v_{\text{ego}}^2}{2 \cdot d_{\text{yield\_line}}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Döner kavşak ve kontrolsüz kavşaklarda trafik akışını kilitlemeden (Deadlock) ve kazaya sebebiyet vermeden güvenli geçiş aralıklarını matematiksel kesinlikle değerlendirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Tereddüt Kilitlenmesi (Hesitation Deadlock):** Otonom araçların kavşak girişinde sonsuza kadar beklemesini önledi; 3.5s üzeri her güvenli aralığı cesurca kullandı.
- **Kavşak İçi Çarpışmalar:** Görüş açısındaki tüm araçların TTC değerini sürekli takip ederek tehlikeli girişleri anında engelledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Kör Noktalar ve Görüş Engelleri:** Ağaç, tabela veya büyük kamyon arkasından aniden hızla çıkan araçlar için ekstra ihtiyat payı (Occlusion Buffer) eklenmelidir.
- **Yaya Geçitleri:** Döner kavşak giriş ve çıkışındaki yayalar ile bisikletlilerin de karar ağacına entegre edilmesi gerekir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Derin Pekiştirmeli Öğrenme (RL - Reinforcement Learning):** Yoğun trafikte çok başarılıdır ancak ASIL-D fonksiyonel güvenlik garantisi ve deterministik açıklanabilirlik sunamaz.
- **Kural Tabanlı Basit Mesafe Eşiği:** Yaklaşan aracın hızını dikkate almadığı için yüksek hızlı trafikte yetersiz kalır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Roundabout** | Trafiğin tek yönlü bir ada etrafında aktığı modern döner kavşak. |
| **Right-of-Way** | Trafik kanunlarına göre kavşakta öncelikli geçiş hakkı. |
| **Time-To-Collision (TTC)** | İki aracın mevcut hız ve doğrultularında çarpışmasına kalan tahmini süre. |
| **Gap Acceptance** | Sürücünün veya otonom sistemin kavşağa güvenle girmek için kabul ettiği zaman aralığı. |
| **Yield Line** | Kavşak girişinde diğer araçlara yol vermek için durulan çizgi. |
| **Hierarchical FSM** | Alt ve üst durumlardan oluşan hiyerarşik sonlu durum makinesi mimarisi. |
| **Deadlock** | İki veya daha fazla aracın birbirine yol vermek için durup trafiği tamamen kilitlemesi. |
| **Critical Vehicle** | Kavşak içinde ego araca en düşük TTC ile yaklaşan ve en büyük tehdidi oluşturan araç. |
| **Anti-Hesitation** | Kararsızlık ve titrek frenleme hareketlerini önleyen sürüş tutarlılık filtresi. |
| **Circulating Traffic** | Döner kavşağın dairesel şeridi içinde dönmekte olan araçlar. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 deterministik ve ISO 26262 güvenlik kuralı     | • Agresif insan sürücülerin yol vermeme durumunda     |
| • 10 µs ultra hızlı RTOS karar çevrimi               |   bekleme süresinin uzaması                           |
| • Çoklu araç TTC füzyonu ile sıfır kaza riski         |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Robotaksi operasyonlarında karmaşık Avrupa tipi     | • Kavşak içinde aşırı hız yapan veya şerit ihlali     |
|   döner kavşaklarda insan benzeri akıcı sürüş         |   yapan kural tanımaz diğer araçlar                   |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Kavşak Karar Ağacı Akış Şeması

```
[ Kavşak İçi Araçların Algılanması (Mesafe ve Hızlar) ]
                           |
                           v
        [ Time-To-Collision (TTC = d / v) Hesabı ]
                           |
                           v
         [ En Kritik Araç TTC Değerinin Seçimi ]
                           |
            +--------------+--------------+
            |                             |
            v                             v
    [ TTC >= 3.5 Saniye ]         [ TTC < 3.5 Saniye ]
            |                             |
            v                             v
[ Karar: ENTERING ]               [ Karar: YIELDING ]
- Hedef İvme: +1.5 m/s²          - Hedef İvme: -2.0 m/s²
- Güvenli Kavşak Girişi           - Yol Verme Çizgisinde Bekle
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Kavşak Karar Ağacı simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
