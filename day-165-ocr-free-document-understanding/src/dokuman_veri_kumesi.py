"""
Doküman Veri Kümesi Modülü (Day 165 - FAZ 9).
Akademik LaTeX formülleri, Markdown tabloları ve Finansal Fatura veri senaryoları.
"""

from typing import List, Dict, Any


class DokumanVeriKumesi:
    """Nougat ve Donut için Doküman Senaryoları Bankası."""

    @classmethod
    def senaryolari_getir(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "doc_latex_01",
                "dokuman_tipi": "Akademik Formül (LaTeX)",
                "baslik": "Gauss İntegrali ve Euler Özdeşliği",
                "hedef_cikti": r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \quad \text{ve} \quad e^{i\pi} + 1 = 0",
                "tahmin_cikti": r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \quad \text{ve} \quad e^{i\pi} + 1 = 0",
            },
            {
                "id": "doc_table_02",
                "dokuman_tipi": "Markdown Tablosu",
                "baslik": "Finansal Gelir Tablosu (Q1-Q2)",
                "hedef_cikti": "| Çeyrek | Gelir ($M) | Kar Marjı |\n| :--- | :--- | :--- |\n| Q1 | 120.5 | %18.4 |\n| Q2 | 145.2 | %21.0 |",
                "tahmin_cikti": "| Çeyrek | Gelir ($M) | Kar Marjı |\n| :--- | :--- | :--- |\n| Q1 | 120.5 | %18.4 |\n| Q2 | 145.2 | %21.0 |",
            },
            {
                "id": "doc_invoice_03",
                "dokuman_tipi": "Yapılandırılmış Fatura (JSON)",
                "baslik": "Kurumsal Tedarik Faturası",
                "hedef_cikti": '{"fatura_no": "INV-2026-88", "tutar": 4500.0, "para_birimi": "USD", "durum": "ODENDI"}',
                "tahmin_cikti": '{"fatura_no": "INV-2026-88", "tutar": 4500.0, "para_birimi": "USD", "durum": "ODENDI"}',
            },
            {
                "id": "doc_matrix_04",
                "dokuman_tipi": "Matris ve Denklem Sistemi (LaTeX)",
                "baslik": "Özdeğer Karakteristik Denklemi",
                "hedef_cikti": r"\det(A - \lambda I) = 0 \implies \begin{pmatrix} a-\lambda & b \\ c & d-\lambda \end{pmatrix}",
                "tahmin_cikti": r"\det(A - \lambda I) = 0 \implies \begin{pmatrix} a-\lambda & b \\ c & d-\lambda \end{pmatrix}",
            },
        ]
