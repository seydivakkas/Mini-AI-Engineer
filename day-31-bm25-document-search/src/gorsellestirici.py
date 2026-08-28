"""
BM25 Leksikal Arama Teşhis ve Parametre Analiz Panosu.
"""

from typing import Dict, List, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BM25Gorsellestirici:
    """
    BM25 arama sonuçları, IDF dağılımı, TF doygunluk eğrileri ve
    parametre duyarlılık analizini görselleştiren 6 panelli teşhis panosu.
    """

    @classmethod
    def arama_panosu_ciz(
        cls,
        arama_sonuclari: List[Dict[str, Any]],
        sorgu: str,
        indeks_istatistikleri: Dict[str, Any],
        hedef_path: str = "ciktilar/bm25_arama_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.9)
        fig, axes = plt.subplots(2, 3, figsize=(19, 12), dpi=300)
        fig.suptitle(f"Day 31: BM25 Leksikal Belge Arama Motoru Analizi (Sorgu: '{sorgu}')", fontsize=15, fontweight="bold", y=0.98)

        # -------------------------------------------------------------
        # Panel 1: Top-k Belge BM25 Uygunluk Skorları
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        if arama_sonuclari:
            doc_adlari = [f"{r['doc_id']}\n({r['baslik'][:18]}...)" for r in arama_sonuclari]
            skorlar = [r["skor"] for r in arama_sonuclari]
            renkler = sns.color_palette("Blues_r", len(skorlar))

            bars1 = ax1.barh(doc_adlari[::-1], skorlar[::-1], color=renkler, edgecolor="black", linewidth=1.1)
            ax1.set_xlabel("BM25 Uygunluk Skoru (Score)", fontweight="bold", fontsize=9)
            ax1.set_title("1. En Uygun Belgeler (Top-k Retrieval)", fontweight="bold", color="#1f77b4")

            for bar in bars1:
                w = bar.get_width()
                ax1.annotate(f"{w:.2f}", (w, bar.get_y() + bar.get_height() / 2),
                             xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, fontweight="bold")
        else:
            ax1.text(0.5, 0.5, "Eşleşen Belge Bulunamadı", ha="center", va="center")

        # -------------------------------------------------------------
        # Panel 2: En İyi Belge İçin Terim Bazlı Skor Katkısı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        if arama_sonuclari and arama_sonuclari[0].get("terim_katkilari"):
            en_iyi = arama_sonuclari[0]
            terimler = list(en_iyi["terim_katkilari"].keys())
            katkilar = list(en_iyi["terim_katkilari"].values())
            renkler2 = sns.color_palette("Greens_r", len(terimler))

            bars2 = ax2.bar(terimler, katkilar, color=renkler2, edgecolor="black", linewidth=1.1, width=0.5)
            ax2.set_ylabel("Terim Katkı Skoru", fontweight="bold", fontsize=9)
            ax2.set_title(f"2. Terim Bazlı Skor Katkısı ({en_iyi['doc_id']})", fontweight="bold", color="#2ca02c")
            for bar in bars2:
                h = bar.get_height()
                ax2.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "Katkı Verisi Yok", ha="center", va="center")

        # -------------------------------------------------------------
        # Panel 3: IDF (Ters Belge Frekansı) Teorik Eğrisi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        N = indeks_istatistikleri.get("belge_sayisi", 20)
        n_q_vals = np.arange(1, N + 1)
        idf_vals = [np.log((N - n + 0.5) / (n + 0.5) + 1.0) for n in n_q_vals]

        ax3.plot(n_q_vals, idf_vals, marker="o", markersize=4, color="#d62728", linewidth=2.0)
        ax3.set_xlabel("Terimi İçeren Belge Sayısı n(q)", fontsize=9, fontweight="bold")
        ax3.set_ylabel("Okapi IDF Skoru", fontsize=9, fontweight="bold")
        ax3.set_title(f"3. IDF Azalma Eğrisi (Toplam Belge N = {N})", fontweight="bold", color="#d62728")
        ax3.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Panel 4: k1 (TF Doygunluğu) Parametre Duyarlılığı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        tf_vals = np.arange(0, 25)
        k1_list = [0.5, 1.2, 1.5, 2.0, 3.0]
        renk_k1 = sns.color_palette("plasma", len(k1_list))

        for idx, k1 in enumerate(k1_list):
            # doc_len == avgdl durumu (K = k1)
            tf_comp = (tf_vals * (k1 + 1.0)) / (tf_vals + k1)
            ax4.plot(tf_vals, tf_comp, label=f"k1 = {k1:.1f}", color=renk_k1[idx], linewidth=1.8)

        ax4.set_xlabel("Terim Frekansı f(q, D)", fontsize=9, fontweight="bold")
        ax4.set_ylabel("TF Terimi Çarpanı", fontsize=9, fontweight="bold")
        ax4.set_title("4. k1 Terim Frekansı Doygunluk Etkisi", fontweight="bold", color="#9467bd")
        ax4.legend(fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Panel 5: b (Belge Uzunluğu Normalizasyonu) Etkisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        len_ratios = np.linspace(0.2, 3.0, 50)  # |D| / avgdl
        b_list = [0.0, 0.25, 0.5, 0.75, 1.0]
        renk_b = sns.color_palette("coolwarm", len(b_list))
        f_q = 3  # Sabit TF = 3

        for idx, b_val in enumerate(b_list):
            k1_def = 1.5
            K = k1_def * ((1.0 - b_val) + b_val * len_ratios)
            tf_comp = (f_q * (k1_def + 1.0)) / (f_q + K)
            ax5.plot(len_ratios, tf_comp, label=f"b = {b_val:.2f}", color=renk_b[idx], linewidth=1.8)

        ax5.set_xlabel(r"Belge Uzunluk Oranı ($|D| / \text{avgdl}$)", fontsize=9, fontweight="bold")
        ax5.set_ylabel("TF Terimi Çarpanı", fontsize=9, fontweight="bold")
        ax5.set_title("5. b Belge Uzunluğu Ceza Etkisi (TF=3)", fontweight="bold", color="#ff7f0e")
        ax5.legend(fontsize=8)
        ax5.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Panel 6: BM25 vs Klasik Doğrusal TF-IDF Karşılaştırması
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        tf_range = np.arange(1, 20)
        # Klasik TF-IDF (Doğrusal TF ve Logaritmik TF)
        tf_linear = tf_range * 1.5
        tf_sublinear = (1 + np.log(tf_range)) * 1.5
        # BM25 TF (k1=1.5)
        bm25_tf = ((tf_range * 2.5) / (tf_range + 1.5)) * 1.5

        ax6.plot(tf_range, tf_linear, "--", label="Doğrusal TF-IDF (f_q * IDF)", color="#e377c2", linewidth=1.8)
        ax6.plot(tf_range, tf_sublinear, "-.", label="Sublinear TF ( (1+ln(f_q)) * IDF )", color="#7f7f7f", linewidth=1.8)
        ax6.plot(tf_range, bm25_tf, label="Okapi BM25 (Asimptotik Doygunluk)", color="#17becf", linewidth=2.2)

        ax6.set_xlabel("Terim Frekansı f(q, D)", fontsize=9, fontweight="bold")
        ax6.set_ylabel("Puan Katkısı", fontsize=9, fontweight="bold")
        ax6.set_title("6. BM25 Asimptotik Doygunluk vs TF-IDF", fontweight="bold", color="#17becf")
        ax6.legend(fontsize=8)
        ax6.grid(True, linestyle=":", alpha=0.6)

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.28, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
