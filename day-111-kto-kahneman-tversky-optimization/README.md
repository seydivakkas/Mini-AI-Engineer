# Day 111: Kahneman-Tversky Optimization (KTO) ile LLM Hizalaması

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; Nobel ödüllü Daniel Kahneman ve Amos Tversky'nin **Beklenti Teorisi (Prospect Theory)** üzerine kurulan, çiftli veri zorunluluğunu kaldırarak **Tekil İkili (Beğenildi / Beğenilmedi - Binary Feedback)** geri bildirimlerle LLM hizalaması yapan **Kahneman-Tversky Optimization (KTO)** algoritmasını sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Davranışsal İktisat ve Başparmak Yukarı/Aşağı Analojisi"

Gerçek dünyada bir ChatGPT kullanıcısı aynı soruya 2 farklı cevap ürettirip "A cevabı B'den daha iyi" diye çiftli kıyaslama yapmaz! Kullanıcı tek bir cevabı okur ve basitçe **👍 Beğendim (+1)** veya **👎 Beğenmedim (-1)** butonuna basar.

DPO ve PPO bu tekil verileri eğitemez çünkü her zaman $(x, y_w, y_l)$ şeklinde iki cevabın eşleşmesini şart koşar.

**KTO (Kahneman-Tversky Optimization)** tam olarak burada devreye girer:
1. 👍 **Eşleştirilmemiş Tekil Veri (Unpaired Data):** Herhangi bir prompt ve ona verilen tekil bir yanıt $(x, y)$ ve $\pm 1$ etiketi KTO için yeterlidir.
2. 📉 **Kayıp Kaçınması (Loss Aversion):** İnsan psikolojisinde bir şeyi kaybetmenin acısı, aynı şeyi kazanmanın sevincinden **%30 - %100 daha fazladır** ("Losses loom larger than gains"). KTO bu yüzden kötü bir cevabı cezalandırırken ($\lambda_U = 1.33$), iyi bir cevabı ödüllendirmekten ($\lambda_D = 1.0$) daha sert davranır!
3. 🎯 **Referans Noktası ($z_{\text{ref}}$):** Model her cevabı mutlak olarak değil, ortalama genel beklentiye göre değerlendirir.

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Beklenti Teorisi (Prospect Theory)
İnsan kararları beklenen fayda teorisine değil, referans noktasına göre kazanç/kayıp algısına dayanır. KTO'da değer fonksiyonu S-şekillidir:
$$v(r - z_{\text{ref}}) = \begin{cases} 1 - \sigma(r_\theta(x, y) - z_{\text{ref}}), & y \in \mathcal{Y}_D \text{ (Desirable)} \\ 1 - \sigma(z_{\text{ref}} - r_\theta(x, y)), & y \in \mathcal{Y}_U \text{ (Undesirable)} \end{cases}$$

### 2. Eşleştirilmemiş İkili Veri (Unpaired Binary Feedback)
DPO veri toplama maliyeti $O(2N)$ iken, KTO $O(N)$ tekil verilerle çalışır. E-ticaret yorumları, kullanıcı tıklamaları, Reddit upvote/downvote verileri doğrudan KTO eğitim setine dönüştürülebilir.

### 3. Asimetrik Kayıp Kaçınması ($\lambda_U > \lambda_D$) ve Referans Noktası
- **$\lambda_D = 1.0$ (Kazanç Ağırlığı):** Beğenilen yanıtların olasılığını artırır.
- **$\lambda_U = 1.33$ (Kayıp Ağırlığı):** Beğenilmeyen yanıtları daha agresif şekilde bastırır.
- **$z_{\text{ref}} = \mathbb{E}[r_\theta(x, y)]:$** Hareketli ortalamayla güncellenen referans noktası.

### 4. Endüstriyel Entegrasyon (Contextual AI, TRL `KTOTrainer`, Üretim Sistemleri)
- **Contextual AI:** KTO'yu ilk geliştiren ekip olarak Llama-7B/13B modellerinde DPO'ya denk ve bazı benchmarklarda daha üstün sonuçlar elde etti.
- **Hugging Face TRL (`KTOTrainer`):** KTO desteği ekleyerek üretimdeki telemetri ve kullanıcı geri bildirim verilerini doğrudan hizalama hattına bağladı.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **KTO (Kahneman-Tversky Optimization)** | Beklenti teorisi değer fonksiyonuyla tekil ikili geri bildirimler üzerinden LLM hizalayan algoritma. |
| **Prospect Theory (Beklenti Teorisi)** | İnsanların risk ve kayıp altındaki psikolojik karar alma mekanizmasını açıklayan Nobel ödüllü iktisat teorisi. |
| **Loss Aversion (Kayıp Kaçınması)** | Kayıpların yarattığı negatif etkinin, eşit miktardaki kazançların pozitif etkisinden daha büyük olması ilkesi. |
| **Reference Point ($z_{\text{ref}}$)** | Ödüllerin kazanç mı yoksa kayıp mı olduğunu belirleyen dinamik taban beklenti eşiği. |
| **Desirable Sample ($y \in \mathcal{Y}_D$)** | Kullanıcı veya denetçi tarafından beğenilmiş (+1 / Thumbs Up) kaliteli yanıt. |
| **Undesirable Sample ($y \in \mathcal{Y}_U$)** | Kullanıcı veya denetçi tarafından reddedilmiş (-1 / Thumbs Down) kalitesiz yanıt. |
| **Unpaired Alignment** | Prompt başına iki alternatif üretmeden, bağımsız tekil örneklerle yapılan tercih optimizasyonu. |
| **Asymmetric Loss Weighting ($\lambda_D, \lambda_U$)** | Desirable ve Undesirable kayıplarına atanan farklılaştırılmış ceza katsayıları ($\lambda_U > \lambda_D$). |
| **S-Shaped Value Curve** | Kazançlarda konkav (azalan verim), kayıplarda konveks ve dik olan psikolojik fayda eğrisi. |
| **Implicit Reward ($\hat{r}$)** | Politikanın referans modele göre log-olasılık oranı: $\beta(\log \pi_\theta - \log \pi_{\text{ref}})$. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Eşleştirilmemiş tekil veri         │ • z_ref referans noktası dengesiz    │
 │   (👍 / 👎) ile doğrudan çalışabilme.│   ise eğitim kararsızlaşabilir.      │
 │ • DPO'ya denk hizalama başarısı.     │ • Sınıf dengesizliğine (örn. %90 👍  │
 │ • Gerçek dünya telemetrisine tam uyum│   %10 👎) karşı hiperparametre ayarı.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Üretim loglarından otomatik        │ • Çok ince ayrıntılı (fine-grained)  │
 │   hizalama veri seti devşirme.       │   nüanslarda çiftli DPO'nun bazen    │
 │ • Düşük veri etiketleme maliyeti.    │   daha keskin ayrışması.             │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/kto_alignment_paneli.png` dosyası üretilir:
1. **KTO Asimetrik Kayıp Eğrileri (Total, Desirable $\mathcal{L}_D$, Undesirable $\mathcal{L}_U$)**
2. **İkili Tercih Doğruluğu (% Binary Alignment Accuracy)**
3. **Örtük Ödüllerin ve Marjinin Ayrışması ($\hat{r}_D$ vs $\hat{r}_U$)**
4. **Beklenti Teorisi S-Eğrisi ve Kayıp Kaçınması ($\lambda_U = 1.33 > \lambda_D = 1.0$)**
5. **KTO Matematik ve Asimetrik Beklenti Kartı**
6. **Stajyer Notu & KTO Karar Sertifikası**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Ana kıyaslama ve görselleştirme akışını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
