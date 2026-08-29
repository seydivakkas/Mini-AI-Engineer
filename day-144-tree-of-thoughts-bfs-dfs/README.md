# Day 144: Tree of Thoughts (ToT): BFS ve DFS Arama ile Düşünce Ağacı Gezintisi & Geri İzleme (Backtracking)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; Shunyu Yao ve ekibinin öncülüğünü yaptığı **Tree of Thoughts (ToT)** mimarisini, **Düşünce Durum Değerlendiricisi (Value Function $V(s)$)**, **Genişlik Öncelikli Arama (BFS / Beam Search)**, **Derinlik Öncelikli Arama (DFS)** ve **Çıkmaz Sokaklardan Geri İzleme (Backtracking)** mekanizmalarıyla **Game of 24 (24 Oyunu)** üzerinde sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Chain-of-Thought Doğrusal Çizgisini Aşıp Neden Ağaç (Tree) Aramasına Geçiyoruz?
- **Doğrusal CoT Nerede Tıkanır?**
  Standart CoT, kelimeleri soldan sağa doğru üretir ($A \to B \to C$). Eğer model $B$ adımında yanlış bir varsayım yaparsa, bunu fark edemez ve hata zincirleme devam eder (**Error Propagation**). Klasik Game of 24 bulmacasında standart CoT başarısı sadece **%7.3**'tür!
- **Tree of Thoughts (ToT) Nasıl Çalışır?**
  1. 🌲 **Düşünce Üretimi (Thought Generation):** Mevcut durumdan birden fazla alternatif sonraki adım türetilir.
  2. ⚖️ **Durum Değerlendirmesi (State Evaluation $V(s)$):** Her ara durum hedefe ulaşabilirlik açısından puanlanır (`kesin`, `olası`, `imkansız`).
  3. ✂️ **Budama (Pruning):** İmkansız çıkmaz sokaklar derhal kesilip atılır.
  4. 🔄 **Geri İzleme (Backtracking):** Model bir dalda çıkmaza girerse önceki çatallanma noktasına geri döner ve alternatif dalları keşfeder.
  5. 🏆 **Devasa Başarı:** Game of 24 başarısı **%7.3'ten %78.0'e fırlar!**

```
               [Kök Durum: 4 9 10 13]
                 ┌───────┴───────┐
                 ▼               ▼
          [13-9=4, 4, 10]   [10*4=40, 9, 13]
          Puan: 0.95        Puan: 0.10 (BUDANDI! ✂️)
                 │
                 ▼
          [10-4=6, 4] ──► Puan: 0.95
                 │
                 ▼
          [6 * 4 = 24] ──► HEDEF BULUNDU! 🎯 (Puan: 1.00)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Düşünce Durum Uzayı (Thought State Space Formulation)
- Problem bir durum uzayı grafı $\mathcal{G} = (\mathcal{S}, \mathcal{E})$ olarak modellenir. Başlangıç durumu $s_0$, her ara düşünce adımı $s \in \mathcal{S}$ ve hedef kümesi $\mathcal{S}^*$ ile tanımlanır.

### B. Sezgisel Değer Fonksiyonu ve Budama ($V(s)$ Heuristic Evaluation)
- Bir durumun geçerliliği sezgisel puanlayıcı ile derecelendirilir:
  $$V(s) = \begin{cases} 1.0 & \text{hedef bulundu (kesin)} \\ 0.7 - 0.95 & \text{umut verici ara adım (olası)} \\ 0.0 & \text{çıkmaz sokak / imkansız (budanır)} \end{cases}$$

### C. BFS (Genişlik Öncelikli Arama / Beam Search)
- Katman katman en yüksek değerli $k$ durumu saklayarak (Beam Width $k=3-5$) paralel genişletme yapar.

### D. DFS (Derinlik Öncelikli Arama ve Backtracking)
- En derin yaprağa kadar iner. Bir dal $V(s) \le 0.10$ olduğunda veya maksimum derinliğe ulaşıldığında yığından (Stack) bir önceki dallanma durumuna geri döner.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Tree of Thoughts (ToT)** | Akıl yürütme adımlarını doğrusal zincir yerine bir arama ağacı olarak keşfeden paradigma. |
| **Thought State (Düşünce Durumu)** | Arama ağacındaki her bir düğümün temsil ettiği ara mantıksal veya matematiksel veri durumu. |
| **Value Function ($V(s)$)** | Bir düşünce durumunun nihai çözüme ulaşma olasılığını tahmin eden değer fonksiyonu. |
| **Pruning (Budama)** | Düşük olasılıklı veya çıkmaz sokak olan düşünce dallarını arama uzayından çıkarma işlemi. |
| **Backtracking (Geri İzleme)** | Bir dalda başarıya ulaşılamadığında önceki geçerli ebeveyn düğüme geri dönme mekanizması. |
| **Beam Search (BFS)** | Her seviyede sadece en yüksek puanlı $k$ sayıda düşünce düğümünü genişleten arama tekniği. |
| **Depth-First Search (DFS)** | Bir düşünce zincirini sonuna kadar takip eden, başarısızlıkta geri izleyen arama yöntemi. |
| **Lookahead Search** | Bir adım atmadan önce sonraki olası durumları ve sonuçları simüle etme süreci. |
| **Game of 24** | Verilen 4 sayıyı temel işlemlerle 24'e ulaştırmayı hedefleyen klasik kombinatoryal bulmaca. |
| **Error Propagation** | Doğrusal CoT'ta ilk adımlardaki hatanın sonraki tüm adımları kaçınılmaz olarak bozması sorunu. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Game of 24'te CoT'un %7.3'lük başa-│ • Çoklu dal genişletme sebebiyle     │
 │   rısını %78.0'e çıkaran güç.        │   artan token ve hesaplama maliyeti. │
 │ • Çıkmazları erken fark edip budama  │ • Derin ağaçlarda durum değerlendirme│
 │   ve geri dönebilme esnekliği.       │   gecikmesi (Latency).               │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Satranç/Go, robotik planlama,      │ • Değer fonksiyonunun (V(s)) yetersiz│
 │   matematik teorem ispatı ve kodlama.│   kalması durumunda yanlış budama.   │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/tree_of_thoughts_bfs_dfs_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
