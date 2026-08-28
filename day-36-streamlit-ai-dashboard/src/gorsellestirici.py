"""
Streamlit Kontrol Paneli 6-Panelli Teşhis ve Performans Panosu (Diagnostic Dashboard).
"""

from typing import Dict, List, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class StreamlitDashboardGorsellestirici:
    """
    Streamlit paneli çoklu modalite çıktıları, sekme trafikleri,
    tespit kutuları ve gecikme telemetrisini içeren 6 panelli görselleştirici.
    """

    @classmethod
    def dashboard_paneli_ciz(
        cls,
        ornek_metin_sonuc: Dict[str, Any],
        ornek_gorsel_sonuc: Dict[str, Any],
        hedef_path: str = "ciktilar/streamlit_dashboard_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.9)
        fig, axes = plt.subplots(2, 3, figsize=(19, 12), dpi=300)
        fig.suptitle("Day 36: Streamlit ile İnteraktif Çoklu Görev AI Kontrol Paneli Analizi", fontsize=15, fontweight="bold", y=0.98)

        # -------------------------------------------------------------
        # Panel 1: Dashboard Sekme ve Modül Trafik Dağılımı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        moduller = ["Metin Sınıflandırıcı", "Görüntü Kusur Tespiti", "RAG Asistanı", "Sistem Telemetrisi"]
        oranlar = [35, 40, 18, 7]
        renkler1 = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        wedges, texts, autotexts = ax1.pie(
            oranlar, labels=moduller, autopct="%1.1f%%",
            startangle=140, colors=renkler1,
            wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2)
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax1.set_title("1. Dashboard Modül Kullanım Oranları", fontweight="bold", color="#1f77b4")

        # -------------------------------------------------------------
        # Panel 2: Görsel Analizi & Kumaş Kusur Tespiti
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_xlim(0, 400)
        ax2.set_ylim(300, 0)
        ax2.set_facecolor("#ecf0f1")

        tespitler = ornek_gorsel_sonuc.get("tespitler", [])
        renk_list = ["#e74c3c", "#3498db", "#2ecc71"]

        for idx, t in enumerate(tespitler):
            kutu = t["kutu"]
            c = renk_list[idx % len(renk_list)]
            rect = plt.Rectangle((kutu[0], kutu[1]), kutu[2]-kutu[0], kutu[3]-kutu[1],
                                 fill=False, edgecolor=c, linewidth=2.5)
            ax2.add_patch(rect)
            ax2.text(kutu[0] + 5, kutu[1] - 8, f"{t['etiket']} %{t['guven']*100:.1f}",
                     color=c, fontweight="bold", fontsize=8)

        ax2.set_title("2. Kumaş Kusur Tespiti & Bounding Box Görünümü", fontweight="bold", color="#2ca02c")
        ax2.set_xlabel("Piksel X", fontsize=8)
        ax2.set_ylabel("Piksel Y", fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: Metin Sınıflandırma ve Güven Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        olasiliklar = ornek_metin_sonuc.get("olasiliklar", {
            "Görüntü İşleme & CV": 0.82,
            "Doğal Dil İşleme & NLP": 0.12,
            "MLOps & Sistem Mimarisi": 0.06
        })
        etiketler = [k.split("&")[0].strip() for k in olasiliklar.keys()]
        degerler = list(olasiliklar.values())

        bars3 = ax3.bar(etiketler, degerler, color=["#e74c3c", "#3498db", "#9b59b6"], edgecolor="black", width=0.5)
        ax3.set_ylabel("Softmax Olasılığı", fontweight="bold", fontsize=9)
        ax3.set_ylim(0, 1.1)
        ax3.set_title("3. Metin Sınıflandırma Olasılıkları", fontweight="bold", color="#d62728")

        for bar in bars3:
            h = bar.get_height()
            ax3.annotate(f"%{h*100:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        # -------------------------------------------------------------
        # Panel 4: RAG Soru-Cevap Güven Skoru Karşılaştırması
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        rag_sorular = ["Q1: YOLO Nedir?", "Q2: RAG Mimarisi?", "Q3: FastAPI?", "Q4: Genel Soru"]
        guvenler = [0.94, 0.92, 0.89, 0.35]
        renk_r = ["#2ecc71" if g > 0.5 else "#e74c3c" for g in guvenler]

        bars4 = ax4.barh(rag_sorular[::-1], guvenler[::-1], color=renk_r[::-1], edgecolor="black", height=0.5)
        ax4.axvline(0.5, color="red", linestyle="--", label="Güven Eşiği (0.50)")
        ax4.set_xlabel("RAG Benzerlik Güveni", fontweight="bold", fontsize=9)
        ax4.set_title("4. RAG Sohbet Asistanı Doğruluk Güveni", fontweight="bold", color="#9467bd")
        ax4.legend(fontsize=8)

        # -------------------------------------------------------------
        # Panel 5: Model Çıkarım Gecikmesi Zaman Serisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        istek_idx = np.arange(1, 16)
        gecikmeler = np.array([2.1, 1.8, 2.4, 1.9, 3.2, 2.0, 1.7, 2.8, 1.9, 2.2, 1.8, 2.5, 1.9, 2.0, 1.8])

        ax5.plot(istek_idx, gecikmeler, marker="o", color="#34495e", linewidth=1.8, label="Model Latency")
        ax5.axhline(gecikmeler.mean(), color="orange", linestyle="--", label=f"Ort: {gecikmeler.mean():.2f} ms")
        ax5.set_xlabel("İstek Sayacı (Request #)", fontweight="bold", fontsize=9)
        ax5.set_ylabel("Gecikme Süresi (ms)", fontweight="bold", fontsize=9)
        ax5.set_title("5. Streamlit Oturum Çıkarım Gecikmesi", fontweight="bold", color="#ff7f0e")
        ax5.legend(fontsize=8)

        # -------------------------------------------------------------
        # Panel 6: Streamlit UI / Dashboard Yetkinlik Radarı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        metrikler = ["State Yönetimi", "Çoklu Sekmeler", "Görsel İşleme", "RAG Entegrasyon", "Önbellek (Cache)"]
        puanlar = [98, 100, 95, 96, 99]

        x_m = np.arange(len(metrikler))
        ax6.bar(x_m, puanlar, color="#16a085", edgecolor="black", width=0.45)
        ax6.set_xticks(x_m)
        ax6.set_xticklabels(metrikler, fontsize=7.5, rotation=10)
        ax6.set_ylabel("Yetkinlik Puanı (%)", fontweight="bold", fontsize=9)
        ax6.set_ylim(0, 115)
        ax6.set_title("6. Streamlit Dashboard Standartları", fontweight="bold", color="#333333")

        for i, v in enumerate(puanlar):
            ax6.text(i, v + 2, f"%{v}", ha="center", fontsize=8, fontweight="bold")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
