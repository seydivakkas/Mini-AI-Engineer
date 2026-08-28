"""
Otomatik Deney ve Model Değerlendirme Raporlayıcısı (HTML & Markdown Generator).
"""

from typing import Dict, Any
import os
import json


class OtomatikDeneyRaporlayici:
    """Deney telemetrisini konsolide edip şık ve bağımsız bir HTML/Markdown raporu üretir."""

    @classmethod
    def html_raporu_olustur(
        cls,
        egitim_analizi: Dict[str, Any],
        cm_analizi: Dict[str, Any],
        roc_analizi: Dict[str, Any],
        pr_analizi: Dict[str, Any],
        hiperparametreler: Dict[str, Any],
        hedef_path: str = "ciktilar/deney_raporu.html"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        auc = roc_analizi["roc_auc"]
        f1 = cm_analizi["f1_skoru"]
        overfitting = egitim_analizi["overfitting_gap"]

        if auc >= 0.85 and f1 >= 0.80 and overfitting < 0.10:
            karar = "URETIME_HAZIR_ONAYLANDI (PROD_READY)"
            renk = "#2ecc71"
        elif overfitting >= 0.15:
            karar = "ASIRI_OGRENME_UYARISI (OVERFITTING)"
            renk = "#e67e22"
        else:
            karar = "GELISTIRME_GEREKLI (NEEDS_TUNING)"
            renk = "#e74c3c"

        html_icerik = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>AI Deney Raporu - {egitim_analizi.get('model_adi', 'Model')}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #2c3e50; margin: 0; padding: 25px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); padding: 30px; }}
        .header {{ border-bottom: 2px solid #eaeded; padding-bottom: 15px; margin-bottom: 25px; }}
        .badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; background-color: {renk}; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e1e8ed; }}
        th {{ background-color: #f4f6f7; font-weight: 600; color: #34495e; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .card {{ background: #fdfefe; border: 1px solid #d5dbdb; border-radius: 8px; padding: 18px; }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #2980b9; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>🧪 Yapay Zeka Model Deney & Performans Raporu</h2>
            <p><strong>Model Adı:</strong> {egitim_analizi.get('model_adi')} | <strong>Durum:</strong> <span class="badge">{karar}</span></p>
        </div>

        <h3>1. Temel Performans Metrikleri</h3>
        <div class="grid">
            <div class="card">
                <div>ROC-AUC Skoru</div>
                <div class="metric-val">%{auc * 100:.2f}</div>
            </div>
            <div class="card">
                <div>F1-Skoru</div>
                <div class="metric-val">%{f1 * 100:.2f}</div>
            </div>
            <div class="card">
                <div>Doğruluk (Accuracy)</div>
                <div class="metric-val">%{cm_analizi['dogruluk_acc']:.2f}</div>
            </div>
            <div class="card">
                <div>Average Precision (AP)</div>
                <div class="metric-val">%{pr_analizi['average_precision_ap'] * 100:.2f}</div>
            </div>
        </div>

        <h3>2. Eğitim ve Yakınsama Özeti</h3>
        <table>
            <tr><th>Metrik</th><th>Değer</th></tr>
            <tr><td>Toplam Epoch</td><td>{egitim_analizi['toplam_epoch']}</td></tr>
            <tr><td>En İyi Epoch</td><td>{egitim_analizi['en_iyi_epoch']} (Val Loss: {egitim_analizi['en_iyi_val_loss']})</td></tr>
            <tr><td>En İyi Val Accuracy</td><td>%{egitim_analizi['en_iyi_val_acc']}</td></tr>
            <tr><td>Overfitting Farkı (Gap)</td><td>{overfitting} ({egitim_analizi['overfitting_riski']})</td></tr>
            <tr><td>Erken Durdurma Önerisi</td><td>{'Tetiklendi' if egitim_analizi['erken_durdurma_tetiklendi'] else 'Gerekmedi'}</td></tr>
        </table>

        <h3>3. Hiperparametre Yapılandırması</h3>
        <table>
            <tr><th>Parametre</th><th>Değer</th></tr>
            {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in hiperparametreler.items()])}
        </table>

        <p style="text-align: right; color: #7f8c8d; font-size: 12px; margin-top: 30px;">
            Rapor Day 46 Otomatik Deney Motoru Tarafından Üretilmiştir. Telif Hakkı (c) 2026 Seydi Eryılmaz.
        </p>
    </div>
</body>
</html>"""

        with open(hedef_path, "w", encoding="utf-8") as f:
            f.write(html_icerik)

        return hedef_path
