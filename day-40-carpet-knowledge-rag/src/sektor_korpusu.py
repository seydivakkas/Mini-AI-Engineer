"""
Tekstil ve Halı Üretim Teknik Doküman Korpusu (Domain-Specific Knowledge Base).
"""

from typing import List, Dict, Any

TEKSTIL_TEKNIK_KORPUS: List[Dict[str, Any]] = [
    {
        "dokuman_id": "DOC-YARN-01",
        "baslik": "TS EN ISO 2060 İplik Numara, Büküm ve Karışım Standartları",
        "kategori": "iplik_standardi",
        "kaynak_standart": "TS EN ISO 2060 / DIN 53835",
        "guven_derecesi": "RESMI_STANDART",
        "metin": """
1.1 İplik Numaralandırma ve Yoğunluk:
Halı dokumada kullanılan akrilik ipliklerde standart iplik numarası Nm 28/2 (yaklaşık 714 dtex) olarak belirlenmiştir. Yün karışımlı jakarlı halılarda ise Nm 18/2 veya Nm 20/3 iplikler tercih edilmelidir. İplik numarasındaki tolerans sınırı maksimum ±%2.5 olmalıdır.

1.2 İplik Büküm Katsayısı (Twist Factor):
Akrilik halı ipliklerinde tüylenme (pilling) ve hav deformasyonunu önlemek için büküm katsayısı Alpha_m = 75 - 85 aralığında tutulmalıdır. Büküm sayısı tek kat iplikte 420 tur/m, çift kat iplikte (katlama) ise 360 tur/m (Z/S büküm) olarak uygulanmalıdır.

1.3 Mukavemet ve Kopma Uzaması:
Dokuma tezgahı hızının 250 rpm üzerinde olduğu hatlarda iplik kopma dayanımı minimum 14.5 cN/tex, kopma uzaması ise %24 ± 3 olmalıdır. Mukavemeti 12 cN/tex altında kalan iplik partileri çözgü kopuşlarına yol açtığı için üretime sevk edilmez.
"""
    },
    {
        "dokuman_id": "DOC-WEAVE-02",
        "baslik": "Jakarlı Halı Dokuma Tezgahı Ayar ve Çözgü Gerginlik Reçetesi",
        "kategori": "dokuma_recetesi",
        "kaynak_standart": "İç Üretim Teknik Kılavuzu (Vandewiele / Schönherr Uyumlu)",
        "guven_derecesi": "URETIM_KILAVUZU",
        "metin": """
2.1 Çözgü Tansiyonu ve Gerginlik Limitleri:
Elektronik jakarlı halı dokuma tezgahlarında zemin çözgü gerginliği 45 ± 5 cN, hav çözgü gerginliği ise 25 ± 3 cN seviyesine ayarlanmalıdır. Gerginlik 55 cN üzerine çıktığında çözgü ipliği kopması, 20 cN altına düştüğünde ise hav boyu düzensizliği (looping) oluşur.

2.2 Atkı Sıklığı ve Tarak Ayarları:
Standart 500 tarak jakarlı makinelerde atkı sıklığı santimetrede 42 atkı (420 atkı/dm) olmalıdır. Çift raportlu madalyonlu desenlerde hav yüksekliği 11.0 mm, toplam halı sırt kalınlığı 13.5 mm olarak kalibre edilmelidir.

2.3 Hav Bıçakları ve Bileme Periyotları:
Hav kesme bıçaklarının bileme periyodu kesintisiz 48 çalışma saatidir. Kör bıçaklar tüylenmeye ve hav boyunda mikro-kırıklara neden olur. Bıçak baskı yayı basıncı 2.2 bar olarak sabitlenmelidir.
"""
    },
    {
        "dokuman_id": "DOC-FAST-03",
        "baslik": "ISO 105-X12 Tekstil Renk ve Sürtünme Haslığı Test Standartları",
        "kategori": "haslik_testi",
        "kaynak_standart": "ISO 105-X12 / AATCC 8",
        "guven_derecesi": "RESMI_STANDART",
        "metin": """
3.1 Sürtünme Haslığı Kabul Kriterleri (Crocking Fastness):
İç mekan halılarında ISO 105-X12 standardına göre kuru sürtünme haslığı minimum 4. Derece (Gri Skala), yaş sürtünme haslığı ise minimum 3-4 Derece olmalıdır. Bordo ve koyu lacivert gibi yoğun pigmentli renklerde yaş sürtünme haslığı 3. Derecenin altına inemez.

3.2 Işık Haslığı (Xenon Arc Testi - ISO 105-B02):
Doğrudan güneş ışığına maruz kalan salon ve otel halılarında ışık haslığı Mavi Yün Skalasında minimum 5-6. Derece olmalıdır. 40 saatlik UV ışınlama sonucunda Delta-E 2000 renk değişimi 2.0 sınırını aşmamalıdır.

3.3 Su ve Şampuan Haslığı (ISO 105-E01):
Halı yıkama ve profesyonel şampuanlama testlerinde lekeleme derecesi minimum 4. Derece olmalıdır. Liften liffe renk akması (bleeding) tespit edilen partiler yıkanmadan fiksaja alınamaz.
"""
    },
    {
        "dokuman_id": "DOC-FINISH-04",
        "baslik": "Tekstil Apre, Kurutma Sıcaklıkları ve Leke Önleyici Kimyasallar",
        "kategori": "apre_kimyasal",
        "kaynak_standart": "Tekstil Kimyası ve Terbiye Talimatı",
        "guven_derecesi": "URETIM_KILAVUZU",
        "metin": """
4.1 Kurutma ve Fiksaj Sıcaklıkları (Stenter Ram Ayarı):
Lateks sırt kaplama ve apre işleminde kurutma tüneli sıcaklığı 145°C - 155°C arasında kademeli olarak uygulanmalıdır. 160°C üzerindeki sıcaklıklar akrilik liflerinde sararmaya ve sertleşmeye, 135°C altı ise lateksin tam kürlenmemesine (tozuma yapmasına) neden olur.

4.2 Leke Önleyici Florokarbon Apre Reçetesi:
Su ve yağ iticilik (oleofobik/hidrofobik) apre banyosunda C6-florokarbon emülsiyonu konsantrasyonu %2.5 - 3.0 (25-30 g/L) oranında çözündürülmelidir. Banyo pH değeri 4.5 - 5.5 aralığında asetik asit ile tamponlanmalı ve sıkma silindiri pikap oranı %65 olarak ayarlanmalıdır.

4.3 Antistatik ve Antibakteriyel Koruma:
Kalıcı iletkenlik için kuaterner amonyum tuzu bazlı antistatik apre %1.0 konsantrasyonda eklenmelidir. Bu işlem statik elektrik birikimini 2.0 kV sınırının altına düşürür.
"""
    },
    {
        "dokuman_id": "DOC-MAINT-05",
        "baslik": "Dokuma Tezgahlarında Yağ Lekesi ve Mekanik Arıza Giderme",
        "kategori": "bakim_onarim",
        "kaynak_standart": "Mekanik Bakım & Kalite Güvence El Kitabı",
        "guven_derecesi": "BAKIM_KILAVUZU",
        "metin": """
5.1 Tezgah Yağlama ve Leke Önleme Prosedürü:
Jakar modül ve tahrik zincirlerinde yalnızca sentetik renksiz PTFE bazlı tekstil yağları kullanılmalıdır. Mineral bazlı koyu yağ kullanımı yasaktır. İğne tablası her 8 saatlik vardiya sonunda kuru basınçlı hava (6 bar) ile üflenerek temizlenmelidir.

5.2 Yağ Lekesi Temizleme Protokolü:
Kumaş yüzeyine damlayan taze yağ lekelerine anında ultrasonik tabanca ile trikloretilen içermeyen ekolojik leke çözücü uygulanmalı ve vakumla çekilmelidir. 24 saat bekleyen yağ lekelerinde elyaf içine işleme gerçekleştiğinden kumaş 2. Kaliteye ayrılır.
"""
    }
]
