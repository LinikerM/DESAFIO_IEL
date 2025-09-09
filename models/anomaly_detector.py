import pandas as pd  # manipulação tabular
import numpy as np   # numérico/vetorização
from typing import List, Tuple, Optional  # tipos
from sklearn.ensemble import IsolationForest  # detector ML
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # pré-processamento
from sklearn.compose import ColumnTransformer  # colunas mistas
from sklearn.pipeline import Pipeline  # pipeline ML


class AnomalyDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.pipeline: Optional[Pipeline] = None  # guarda pipeline ML
        self.contamination = contamination  # fração anômala estimada
        self.random_state = random_state  # reprodutibilidade

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # evita side-effects
        if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
            df.loc[:, "date"] = pd.to_datetime(df["date"], errors="coerce")  # força datetime
        return df  # retorna dataframe coerente

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        df = self._ensure_datetime(df)  # garante datetime
        f = df.copy()  # trabalha em cópia

        f.loc[:, "fee_difference"] = f["charged_fee"] - f["expected_fee"]  # delta em R$
        f.loc[:, "fee_ratio"] = f["charged_fee"] / (f["expected_fee"] + 1e-6)  # razão segura
        f.loc[:, "day_of_week"] = f["date"].dt.dayofweek  # dia da semana
        f.loc[:, "month"] = f["date"].dt.month  # mês numérico

        numeric_cols: List[str] = ["amount", "fee_difference", "fee_ratio", "day_of_week", "month"]  # numéricas
        categorical_cols: List[str] = []  # categóricas

        for col in ["merchant_type", "channel"]:  # adiciona categóricas se existirem
            if col in f.columns:
                categorical_cols.append(col)

        X = f.loc[:, numeric_cols + categorical_cols].copy()  # fatia final
        bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()  # booleans
        if bool_cols:
            X.loc[:, bool_cols] = X[bool_cols].astype(int)  # bool->int

        for col in numeric_cols:
            X.loc[:, col] = pd.to_numeric(X[col], errors="coerce")  # numéricos robustos

        for col in X.columns:
            if X[col].dtype.name == 'category':
                X[col] = X[col].astype(str)
        X = X.fillna(0)  # nulos viram 0
        return X, numeric_cols, categorical_cols  # retorna matriz e listas

    def _build_pipeline(self, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
        preprocess = ColumnTransformer(  # pré-processamento misto
            transformers=[
                ("num", StandardScaler(), numeric_cols),  # escala numéricos
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),  # codifica categóricas
            ],
            remainder="drop",  # descarta extras
        )
        model = IsolationForest(contamination=self.contamination, random_state=self.random_state)  # detector
        return Pipeline(steps=[("prep", preprocess), ("model", model)])  # pipeline completo

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        mode: str = "hybrid",  # "stat" (rápido), "ml" (pipeline), "hybrid" (combina)
        chunk_size: Optional[int] = None,  # chunk opcional para pontuar em partes
        robust_thresh: float = 3.5  # limiar Z robusto
    ) -> pd.DataFrame:
        base = self._ensure_datetime(df)  # normaliza datas
        out = base.copy()  # cópia para saída

        # --- modo estatístico robusto (escala-friendly) ---
        X, _, _ = self.prepare_features(base)  # features prontas
        group_keys = []  # agrupa por segmento
        if "merchant_type" in base.columns:
            group_keys.append("merchant_type")
        if "channel" in base.columns:
            group_keys.append("channel")
        if not group_keys:
            group_keys = ["month"]  # fallback por mês se não houver segmentação

        tmp = pd.concat([base[group_keys + ["date"]], X[["charged_fee"] if "charged_fee" in X.columns else []]], axis=1)  # prepara join
        tmp = base[group_keys + ["date", "charged_fee", "charged_fee_rate"]].copy() if "charged_fee_rate" in base.columns else base[group_keys + ["date", "charged_fee"]].copy()  # garante colunas

        # calcula z-score robusto por grupo/data (mediana e MAD)
        grp = base.groupby(group_keys, dropna=False)  # agrupa por segmento
        med = grp["charged_fee"].transform("median")  # mediana por grupo
        mad = grp["charged_fee"].transform(lambda s: (s - s.median()).abs().median() + 1e-6)  # MAD por grupo
        out.loc[:, "robust_z_fee"] = (base["charged_fee"] - med) / mad  # z robusto

        stat_anom = out["robust_z_fee"].abs() > robust_thresh  # flag estatística

        # --- modo ML com pipeline sklearn ---
        if mode in ("ml", "hybrid"):
            X_ml, num_cols, cat_cols = self.prepare_features(base)  # features ML
            self.pipeline = self._build_pipeline(num_cols, cat_cols)  # constrói pipeline

            if chunk_size and chunk_size > 0:  # pontuação em chunks (memória constante)
                preds = np.empty(len(X_ml), dtype=int)  # aloca vetor
                scores = np.empty(len(X_ml), dtype=float)  # aloca scores
                # fit em amostra inicial (até chunk_size)
                head_end = min(chunk_size, len(X_ml))  # limite head
                self.pipeline.fit(X_ml.iloc[:head_end])  # ajusta no head
                # pontua head
                preds[:head_end] = self.pipeline.predict(X_ml.iloc[:head_end])  # -1/1
                scores[:head_end] = self.pipeline.decision_function(X_ml.iloc[:head_end])  # score
                # pontua o restante por fatias
                start = head_end
                while start < len(X_ml):
                    end = min(start + chunk_size, len(X_ml))  # define janela
                    preds[start:end] = self.pipeline.predict(X_ml.iloc[start:end])  # pred diz -1/1
                    scores[start:end] = self.pipeline.decision_function(X_ml.iloc[start:end])  # score numérico
                    start = end  # avança janela
            else:
                self.pipeline.fit(X_ml)  # ajusta em memória
                preds = self.pipeline.predict(X_ml)  # -1/1
                scores = self.pipeline.decision_function(X_ml)  # score

            ml_anom = preds == -1  # booleans ML
            out.loc[:, "ml_anomaly"] = ml_anom  # marca ML
            out.loc[:, "ml_score"] = scores  # score ML
        else:
            out.loc[:, "ml_anomaly"] = False  # sem ML
            out.loc[:, "ml_score"] = 0.0  # score neutro

        # --- combinação dos sinais (híbrido) ---
        if mode == "stat":
            out.loc[:, "is_anomaly"] = stat_anom  # só estatística
            out.loc[:, "anomaly_score"] = out["robust_z_fee"].abs()  # usa |z| como score
        elif mode == "ml":
            out.loc[:, "is_anomaly"] = out["ml_anomaly"]  # só ML
            out.loc[:, "anomaly_score"] = out["ml_score"]  # usa score do modelo
        else:
            both = stat_anom | out["ml_anomaly"]  # união de flags
            out.loc[:, "is_anomaly"] = both  # anomalia se qualquer um disparar
            # score combinado: max(|z| normalizado, score ML)
            z_norm = (out["robust_z_fee"].abs() / (out["robust_z_fee"].abs().max() + 1e-6)).clip(0, 1)  # normaliza |z|
            ml_norm = (out["ml_score"] - out["ml_score"].min()) / (out["ml_score"].max() - out["ml_score"].min() + 1e-6)  # normaliza ML
            out.loc[:, "anomaly_score"] = np.maximum(z_norm, ml_norm)  # score final

        return out  # retorna dataframe anotado

    def classify_billing_errors(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        out = df.copy()  # cópia segura
        charged = pd.to_numeric(out["charged_fee"], errors="coerce").fillna(0.0)  # cobrado numérico
        expected = pd.to_numeric(out["expected_fee"], errors="coerce").fillna(0.0)  # esperado numérico

        conds = [  # condições de classificação
            (charged.sub(expected).abs() < (expected.abs() * threshold)),  # correto
            (charged > expected * (1 + threshold)),  # excesso
            (charged < expected * (1 - threshold)),  # insuficiente
        ]
        choices = ["CORRETO", "COBRANCA_EXCESSIVA", "COBRANCA_INSUFICIENTE"]  # rótulos
        out.loc[:, "billing_status"] = np.select(conds, choices, default="INDEFINIDO")  # aplica rótulos
        return out  # retorna classificado
