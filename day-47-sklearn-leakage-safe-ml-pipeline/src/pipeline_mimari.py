"""
Veri Sızıntısına (Data Leakage) Karşı Güvenli Scikit-Learn Pipeline Mimarisi.
"""

from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


class GuvenliPipelineUretici:
    """Veri sızıntısını önleyen kapsüllenmiş ColumnTransformer ve Pipeline mimarisi üretir."""

    @classmethod
    def guvenli_pipeline_olustur(
        cls,
        sayisal_kolonlar: List[str],
        kategorik_kolonlar: List[str],
        model_turu: str = "logistic",
        c_param: float = 1.0,
        random_state: int = 42
    ) -> Pipeline:
        """Sayısal ve kategorik dönüşümleri dış dünyadan izole eden güvenli pipeline üretir."""

        # 1. Sayısal Dönüşüm Hattı: Median İmpütasyon + RobustScaler
        sayisal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        # 2. Kategorik Dönüşüm Hattı: Sabit İmpütasyon + OneHotEncoder (handle_unknown='ignore')
        kategorik_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="BILINMIYOR")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        # 3. Kolon Dönüştürücü (ColumnTransformer)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", sayisal_transformer, sayisal_kolonlar),
                ("cat", kategorik_transformer, kategorik_kolonlar)
            ],
            remainder="drop"
        )

        # 4. Model Seçimi
        if model_turu == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        elif model_turu == "hist_gradient":
            model = HistGradientBoostingClassifier(random_state=random_state)
        else:
            model = LogisticRegression(C=c_param, max_iter=1000, random_state=random_state)

        # 5. Uçtan Uca Güvenli Pipeline
        tam_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        return tam_pipeline
