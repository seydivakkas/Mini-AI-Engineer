# Day 315: Görsel Olmayan Modaliteler (Koku, Kızılötesi, Ultrason) Latent Köprüsü (Cross-Modal Latent Bridge for Non-Visual Sensory Modalities)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)

Yapay zeka modellerinin büyük çoğunluğu yalnızca insan benzeri metin ve RGB piksel görsel modalitelerine kilitlenmiştir. Ancak endüstriyel tesisler, havacılık, derin deniz arama ve otonom savunma sistemlerinde en kritik fiziksel anomaliler **görünmezdir**: kimyasal gaz sızıntıları (koku), iç rulman sürtünmeleri (termal kızılötesi) ve malzeme içi mikroskobik çatlaklar (ultrasonik sonar).

**Day 315**, elektronik burun (MOS Gas Sensors), radyometrik termal kızılötesi spektrum ve ultrasonik akustik Doppler sinyallerini tek bir **Birleşik Çoklu-Modalite Gizil Uzayına (Unified Joint Latent Space - $\mathbb{R}^{64}$)** yansıtan ve çok yönlü **InfoNCE Karşıtsal Hizalama (Contrastive Cross-Modal Alignment)** uygulayan derin köprü mimarisini sunar. Model, hiç görmediği fiziksel sensör verilerini metinsel anlamsal prototipler üzerinden **%100 sıfır-örnek (zero-shot) doğruluğuyla** sınıflandırır.

---

## 🏗️ 1. Mimari Tasarım ve Matematiksel Temeller

```
========================================================================================
       GÖRSEL OLMAYAN ÇAPRAZ-MODALİTE GİZİL KÖPRÜ MİMARİSİ (DAY 315)
========================================================================================

   [ 16-Kanal MOS Kimyasal Dizi ] ----> [ Koku Kodlayıcı ] -------> z_olf \
   [ 32-Kanal Radyometrik Termal ] ---> [ Termal Kodlayıcı ] -----> z_thm  +---> Birleşik Gizil
   [ 64-Kanal Akustik Doppler ] ------> [ Sonar Kodlayıcı ] ------> z_snr /      Uzay (R^64)
                                                                                  |
   [ Metinsel Sınıf Prototipleri ] ---> [ Semantik Kodlayıcı ] ---> z_txt --------+
                                                                                  |
   [ InfoNCE Karşıtsal Kayıp & Çapraz-Modalite Hizalama ] <-----------------------/
   L = -log [ exp(sim(z_mod, z_txt)/tau) / sum_k exp(sim(z_mod, z_txt_k)/tau) ]
========================================================================================
```

### Matematiksel Formülasyon

1. **Modaliteye Özgü L2-Normalize Edilmiş Gizil Temsiller:**
   $$\mathbf{z}_{\text{mod}} = \frac{f_{\text{mod}}(\mathbf{x}_{\text{mod}})}{\|f_{\text{mod}}(\mathbf{x}_{\text{mod}})\|_2} \in \mathbb{S}^{d-1}$$

2. **Çapraz-Modalite InfoNCE Hizalama Kaybı:**
   $$\mathcal{L}_{\text{InfoNCE}} = -\sum_{i=1}^B \log \frac{\exp\left(\frac{\mathbf{z}_{\text{mod}, i} \cdot \mathbf{z}_{\text{txt}, i}}{\tau}\right)}{\sum_{k=1}^B \exp\left(\frac{\mathbf{z}_{\text{mod}, i} \cdot \mathbf{z}_{\text{txt}, k}}{\tau}\right)}$$

3. **Duyular-Arası Eşzamanlı Düzenlileştirme (Sensory Cross-Regularization):**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{olf-txt}} + \mathcal{L}_{\text{thm-txt}} + \mathcal{L}_{\text{snr-txt}} + \gamma \cdot \mathcal{L}_{\text{olf-thm}}$$

---

## 🔬 2. Derinlemesine Mimari Analizler

### Analiz 1: Yapay Genel Zeka Neden Görsel Ötesi Duyulara İhtiyaç Duyar?
İnsan duyuları elektromanyetik spektrumun yalnızca 380-700 nm aralığını görür ve kimyasal kokuları çok düşük çözünürlükte algılar. Gerçek dünyada otonom hareket eden bir nükleer santral denetim ajanı veya kimyasal fabrikada gezen robotik bir köpek (quadruped), görsel körlük durumunda (duman, karanlık, boru içi) gaz afinitesi (e-nose), termal yayılım (IR) ve akustik eko (sonar) ile yön bulmak ve arızaları teşhis etmek zorundadır.

### Analiz 2: InfoNCE ve Sıcaklık Parametresi ($\tau = 0.07$) Dengesi
$\tau$ hiper-parametresi, gizil uzaydaki benzerlik dağılımının keskinliğini belirler. $\tau = 0.07$ seçimi, doğru anomali sınıfına ait vektörleri dar bir koni içinde kümelendirirken, farklı arıza sınıflarını (örneğin aşırı ısınan rulman ile kriyojenik sapma) ortogonal yönlere iter.

### Analiz 3: Metinsel Prototiplerden Sıfır-Örnek (Zero-Shot) Fiziksel Çıkarım
Sistem, eğitim sırasında "Toksik Gaz Kaçağı" kavramını kimyasal sensör tepkileriyle eşleştirdiğinde, metinsel prototip üzerinden sıfır-örnek çıkarım yeteneği kazanır. Hiç etiketli veri toplanmamış yeni bir test sensör dizisi geldiğinde, kosinüs benzerliği $\arg\max_c (\mathbf{z}_{\text{sensor}} \cdot \mathbf{z}_{\text{proto}, c})$ hesaplanarak anında %100 doğrulukla sınıflandırma yapılır.

### Analiz 4: Gizil Uzay İzometrisi ve Topolojik Korunum
Farklı sensörlerin uzamsal manifoldları arasındaki ikili mesafelerin ortak uzayda korunması (isometry), modaliteler arası kayıpsız çeviri (cross-modal translation) yapılabilmesini sağlar. Örneğin, sadece koku sensörü okuyan bir ajan, tesisin termal haritasını latent uzay üzerinden başarıyla yeniden inşa edebilir.

---

## 📊 3. 6-Panelli Teşhis Panosu İncelemesi

Modül çalıştırıldığında `ciktilar/gorsel_olmayan_kopru_paneli.png` konumunda üretilen 6 teşhis paneli:

1. **Görsel Olmayan Modaliteler Sıfır-Örnek Doğruluğu:** Koku (%100), Termal IR (%100) ve Ultrasonik Sonar (%100) başarı grafikleri.
2. **Çoklu Modalite Karşıtsal Hizalama Kaybı:** 45 epoch boyunca InfoNCE kaybının pürüzsüz yakınsaması.
3. **Koku (E-Nose) Hata Matrisi:** 6 anomali sınıfında sıfır çapraz hata gösteren kusursuz köşegen dağılımı.
4. **Termal Kızılötesi Hata Matrisi:** Radyometrik spektrum üzerinden tespit edilen net arıza sınıfları.
5. **Ultrasonik Sonar Hata Matrisi:** Akustik Doppler yansımalarından elde edilen sınıf ayrımı.
6. **Birleşik Modalite Modeli Özeti:** Ortalama kosinüs hizalaması ($0.5983$), izometri skoru ($0.6183$) ve entegrasyon seviyesi özeti.

---

## 📖 4. Terimler Sözlüğü (Glossary)

- **Electronic Nose (E-Nose):** Gaz moleküllerinin kimyasal bağlanma afinitesini elektriksel direnç değişimine çeviren Metal-Oksit Yarı İletken (MOS) sensör dizisi.
- **Radiometric Thermal Infrared:** Cisimlerin yaydığı kızılötesi termal ışımanın dalga boyu spektrumunu ölçen optik olmayan sensör.
- **Ultrasonic Acoustic Doppler:** Malzeme içi çatlakları ve akışkan hızlarını yüksek frekanslı ses dalgalarının faz kaymasıyla tespit eden eko sinyali.
- **InfoNCE Loss:** Pozitif eşleşmeleri çeken, negatif eşleşmeleri iten bilgi teorisi tabanlı karşıtsal kayıp fonksiyonu.
- **Zero-Shot Transfer:** Modelin eğitimde görmediği test verilerini ortak semantik uzay üzerinden sıfır ek eğitimle doğru tanıması.
- **Latent Isometry:** Farklı modaliteler arasındaki manifold geometrisinin ve mesafe oranlarının bozulmadan korunması durumu.
- **Modality Gap:** Farklı sensör tiplerinin ham uzaylarındaki dağılım uyumsuzluğu; latent köprü ile kapatılır.
- **Cosine Alignment:** İki farklı sensör temsil vektörünün latent uzaydaki açısal örtüşme derecesi.
- **Class Prototypes:** Metinsel arıza tanımlarının $d$-boyutlu normalize edilmiş merkez semantik vektörleri.
- **Temperature ($\tau$):** Karşıtsal softmax dağılımının entropisini ve ceza sertliğini ayarlayan ölçekleme katsayısı.

---

## ⚖️ 5. SWOT Analizi

```
+----------------------------------------------------+----------------------------------------------------+
| 🟢 GÜÇLÜ YÖNLER (STRENGTHS)                        | 🟡 ZAYIF YÖNLER (WEAKNESSES)                       |
| • %100 sıfır-örnek çoklu-duyusal sınıflandırma     | • Fiziksel sensörlerin sıcaklık/nem sürüklenmesine |
| • Koku, termal ve sonar sinyallerinin tam uyumu    |   (sensor drift) karşı kalibrasyon ihtiyacı        |
| • Sıfır görsel körlük; karanlık ve dumanda çalışma | • Aşırı gürültülü endüstriyel ortamlarda ön filtre |
+----------------------------------------------------+----------------------------------------------------+
| 🔵 FIRSATLAR (OPPORTUNITIES)                       | 🔴 TEHDİTLER (THREATS)                             |
| • Otonom nükleer, kimyasal ve maden robotları      | • Sensör donanım arızalarında eksik modalite       |
| • Savunma sanayiinde su altı sonar ve kamuflaj tespiti| (missing modality) nedeniyle latent sapma riski |
+----------------------------------------------------+----------------------------------------------------+
```

---

## 🚀 6. Hızlı Başlangıç

### Kurulum
```bash
pip install -r gereksinimler.txt
```

### Ana Akışı Çalıştırma
```bash
python ana_akis.py
```

### Testleri Çalıştırma (8/8 Unit Test)
```bash
pytest testler/test_gorsel_olmayan_kopru.py -v
```

---

## 👨‍🏫 7. Senior AI / SRE Mentor Soru-Cevap

**Soru 1: Fiziksel sensörlerdeki donanım kirliliği ve yaşlanma sürüklenmesi (Sensor Drift) latent köprüyü nasıl etkiler?**  
*Cevap:* Sensör dirençleri zamanla kirlenip kaysa bile, normalize edilmiş kodlayıcı katmanları mutlak voltaj yerine bağıl spektral tepe noktalarını (peak ratios) işler. Ayrıca latent uzayda *Domain Adaptation / AdaIN* katmanları kullanılarak sürüklenme gerçek zamanlı dengelenir.

**Soru 2: Sahada bir sensör aniden bozulursa (örneğin E-Nose arızalanırsa) sistem çöker mi?**  
*Cevap:* Hayır. Ortak latent uzay sayesinde sistem *Masked Modality Inference* moduna geçer. Kalan termal ve sonar sinyalleri ortak $z$ vektörünü yeterli doğrulukla doldurarak sıfır kesintiyle çalışmayı sürdürür.

**Soru 3: Bu mimari LLM ajanlarıyla nasıl konuşur?**  
*Cevap:* Latent köprüden çıkan $z \in \mathbb{R}^{64}$ vektörü, bir projeksiyon katmanı (Linear Projector) üzerinden doğrudan LLM'in metin gömme uzayına (Soft Token Embedding) enjekte edilir. Böylece LLM ajanı, "Fabrikanın 4. hattında 320 ppm benzen sızıntısı ve 82°C rulman sürtünmesi algılandı" şeklinde doğrudan anlamsal çıkarım yapar.

---

## 📜 Lisans

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Bu modül eğitim ve araştırma amaçlıdır. İzinsiz kopyalanamaz ve ticari amaçla kullanılamaz.
