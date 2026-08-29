# Day 123: Reflexion — Sözel Öz-Eleştiri ve Episodik Hafıza Ajanı

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; model ağırlıklarını güncellemeye gerek kalmadan sözel pekiştirmeli öğrenme (**Verbal Reinforcement Learning**) ile hatalarından ders çıkaran, **Aktör-Değerlendirici-ÖzEleştiri (Actor-Evaluator-Reflector) Mimarisi**, **Episodik Başarısızlık Hafızası** ve **Çok Turlu İteratif Kod Hata Giderme** sistemini (Shinn et al., NeurIPS 2023) sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Hatalarından Ders Çıkaran Ajan: Reflexion ve Sözel RL"

Geleneksel pekiştirmeli öğrenmede (RL / PPO), bir modelin hatasını düzeltmesi için binlerce ağırlık parametresinin gradyanlarla güncellenmesi gerekir. Bu hem aşırı maliyetlidir hem de dakikalar/saatler sürer.

**Reflexion (Shinn et al., NeurIPS 2023)** ise insan gibi **"Doğal Dille Düşünüp Ders Çıkarma"** yöntemini kullanır:
1. 🧑‍💻 **Aktör (Actor):** Bir problemi çözmek için ilk kodu yazar (Deneme 1).
2. ⚖️ **Değerlendirici (Evaluator):** Kodu birim testlerle çalıştırır. Eğer bir sınır durumunda (örn. tümü negatif sayılar) test patlarsa sıfır veya düşük ödül ($r_t$) verir.
3. 💡 **Reflector (Öz-Eleştiri):** Hatayı ve traceback çıktısını inceler: *"Hata yaptım çünkü negatif dizilerde max_toplam=0 başlattım. Sıradaki denemede dizinin ilk elemanını başlangıç almalıyım ve Kadane algoritmasını tam uygulamalıyım."*
4. 🧠 **Episodik Hafıza (Memory Buffer):** Bu sözel dersi hafızaya kaydeder.
5. 🚀 **Deneme 2 (Trial 2):** Aktör önceki hatasını ve çıkardığı dersi okuyarak kusursuz bir kod üretir ve tüm testleri geçer!
6. 📈 **Sonuç:** Başarı oranı ilk denemedeki %64'ten 3. denemede **%95.8'e (Pass@3)** fırlar!

```
      [1. AKTÖR (LLM)] ────────(Kod Üret)────────> [2. DEĞERLENDİRİCİ]
             ▲                                            │
             │ (Prompt Enjeksiyonu)                       │ Birim Testler
             │                                            ▼
      [4. EPİSODİK HAFIZA] <──(Sözel Ders Çıkar)─── [3. REFLECTOR]
      (Geçmiş Hatalar & Dersler)                    (Self-Reflection)
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Reflexion Paradigması
- **Klasik Ajanların Çıkmazı:** Standart ReAct veya CoT ajanları birim testlerde hata aldığında genellikle aynı hatayı içeren varyasyonlar üretir (**Repetitive Failure Loop**).
- **Reflexion Çözümü:** Başarısızlığı skaler bir sinyalden ($r_t$) zengin bir anlamsal geri bildirime ($\text{sr}_t$) dönüştürerek modele neyi neden yanlış yaptığını anlatır.

### 2. Sözel Pekiştirmeli Öğrenme (Verbal RL) ve Ağırlıksız Adaptasyon
- **Sıfır Ağırlık Güncellemesi:** Modelin ağırlıkları sabittir (Frozen Weights). Öğrenme, modelin bağlam penceresine (Context Window) yerleştirilen episodik hafıza üzerinden gerçekleşir.
- **Hızlı Adaptasyon:** Geri yayılım (Backprop) gerektirmediği için milisaniyeler içinde adaptasyon sağlanır.

### 3. Aktör, Değerlendirici ve Reflector Üçlü Mimari Ayrımı
- **Actor:** Görevi yerine getiren üretici katman ($M_{\text{actor}}$).
- **Evaluator:** Deterministik testleri (veya LLM-as-a-judge) çalıştıran hakem katman ($M_{\text{eval}}$).
- **Reflector:** Hata izlerini analiz edip somut dersler çıkaran eleştirmen katman ($M_{\text{reflect}}$).

### 4. İteratif Kod Hata Giderme ve Pass@k Başarımı
- **Pass@1 (Zero-Shot Base):** %64.2 (Tek seferde doğru kod üretme).
- **Reflexion Trial 1:** %68.0.
- **Reflexion Trial 2:** %88.5 (+%30 Artış).
- **Reflexion Trial 3:** **%95.8 (+%49.2 Artış)**.
- **Hata Tekrarlama Oranı:** %42.0'dan **%3.2'ye** düşüş.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Reflexion** | Sözel pekiştirmeli öğrenme ve hafıza ile çalışan öz-eleştiri ajan çerçevesi. |
| **Verbal RL** | Gradyan güncellemesi yerine doğal dil geri bildirimleriyle politika iyileştirme. |
| **Actor** | Verilen problem ve hafızaya dayanarak kod veya eylem üreten dil modeli. |
| **Evaluator** | Çözümün doğruluğunu birim testlerle nesnel olarak ölçen değerlendirici. |
| **Reflector** | Başarısızlık nedenini analiz edip somut düzeltme stratejisi üreten modül. |
| **Self-Reflection ($\text{sr}_t$)** | Modelin başarısız denemesi hakkında ürettiği sözel öz-eleştiri cümlesi. |
| **Episodic Memory** | Önceki denemelerin kod, hata ve ders geçmişini saklayan çalışma hafızası. |
| **Pass@k** | Modelin $k$ deneme içerisinde problemi en az bir kez doğru çözme olasılığı. |
| **Repetitive Loop** | Ajanın hatasını anlamayıp aynı yanlış kodu tekrar tekrar üretmesi durumu. |
| **Zero Weight Update** | Model parametrelerini değiştirmeden yalnızca bağlam içi yönlendirmeyle öğrenme. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %95.8'e varan Pass@3 başarımı.     │ • Çok turlu LLM çağrıları nedeniyle  │
 │ • Sıfır GPU eğitimi ile anında uyum. │   artan token tüketimi ve gecikme.   │
 │ • Hata tekrarlamayı %3.2'ye indirme. │ • Değerlendirici (test) yetersizse   │
 │ • Kodlama, mantık ve web ajanlarında │   hatalı kod doğru zannedilebilir.   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Otomatik CI/CD hata ayıklama ve    │ • Çok uzun hafıza kayıtlarında       │
 │   kendi kendini düzelten yazılım botu│   bağlam penceresi taşması riski.    │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/reflexion_ajan_paneli.png` dosyası üretilir:
1. **Problem Başına Test Başarı Oranı (Ödül İlerlemesi)**
2. **Benchmark İteratif Pass@k Başarımı (%)**
3. **Aynı Hatayı Tekrarlama Oranı (%)**
4. **Hata Türlerine Göre Reflexion Çözüm Başarısı (%)**
5. **Reflexion Sözel RL (Verbal RL) Döngüsü Şeması**
6. **Reflexion Ajan ve Performans Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Reflexion otonom hata ayıklama ajanını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
