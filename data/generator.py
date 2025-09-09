import os
import pandas as pd
import numpy as np
import random


# ---------- Helpers de Parquet ----------
def _parquet_engine() -> str | None:
    """Retorna 'pyarrow' ou 'fastparquet' se disponível; senão None."""
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            return None


def parquet_available() -> bool:
    """True se alguma engine de Parquet estiver disponível."""
    return _parquet_engine() is not None
# ----------------------------------------


class DataGenerator:
    def __init__(
        self,
        n_transactions: int = 1_000_000,   # total de linhas
        start_date: str | None = None,     # data inicial (default: hoje - months)
        months: int = 12,                  # duração do período (default: 12 meses)
        anomaly_rate: float = 0.10,        # % de anomalias
        seed: int | None = None            # semente opcional p/ reprodutibilidade
    ):
        self.n_transactions = int(n_transactions)

        # Se não passar start_date → começa 'months' meses atrás a partir de hoje
        if start_date is None:
            today = pd.Timestamp.now(tz="America/Sao_Paulo").tz_localize(None).normalize()
            start_date = (today - pd.DateOffset(months=months)).strftime("%Y-%m-%d")

        self.start_date = pd.to_datetime(start_date)
        self.months = int(months)
        self.anomaly_rate = float(anomaly_rate)
        self.seed = seed

    def _make_calendar(self) -> np.ndarray:
        """Cria calendário diário do período configurado."""
        end_date = self.start_date + pd.DateOffset(months=self.months) - pd.Timedelta(days=1)
        dates = pd.date_range(self.start_date, end_date, freq="D")
        return dates.values

    def generate_transactions(self) -> pd.DataFrame:
        """Gera a base sintética de transações com regras + anomalias."""
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        merchant_types = np.array([
            "SUPERMERCADO", "POSTO_COMBUSTIVEL", "RESTAURANTE",
            "FARMACIA", "LOJA_ROUPAS", "ELETRONICOS",
            "ONLINE_MARKETPLACE", "SERVICOS_DIGITAIS"
        ])

        base_fee_map = {
            "SUPERMERCADO": 0.015, "POSTO_COMBUSTIVEL": 0.012, "RESTAURANTE": 0.018,
            "FARMACIA": 0.014, "LOJA_ROUPAS": 0.020, "ELETRONICOS": 0.025,
            "ONLINE_MARKETPLACE": 0.022, "SERVICOS_DIGITAIS": 0.030
        }

        channels = np.array(["PRESENCIAL", "ONLINE", "APP", "TELEFONE"])
        dates_vec = self._make_calendar()

        n = self.n_transactions
        mt_idx = np.random.randint(0, len(merchant_types), size=n)
        ch_idx = np.random.randint(0, len(channels), size=n)
        dt_idx = np.random.randint(0, len(dates_vec), size=n)

        merchant_arr = merchant_types[mt_idx]
        channel_arr = channels[ch_idx]
        date_arr = dates_vec[dt_idx].astype("datetime64[ns]")

        amount = np.random.lognormal(mean=4.0, sigma=1.5, size=n)  # valores com cauda longa
        base_fee_rate = np.vectorize(base_fee_map.get)(merchant_arr)

        expected_fee_rate = base_fee_rate.copy()
        is_exempt = np.zeros(n, dtype=bool)
        exemption_reason = np.empty(n, dtype=object); exemption_reason[:] = None

        # regra: alto valor → -20% na taxa (marcamos como isenção para proxy de avaliação)
        high_val = amount > 5000
        expected_fee_rate = np.where(high_val, expected_fee_rate * 0.8, expected_fee_rate)
        is_exempt |= high_val
        exemption_reason = np.where(high_val, "ALTO_VALOR", exemption_reason)

        # regra: supermercado no domingo → isento
        dt = pd.to_datetime(date_arr)
        is_sunday = (dt.weekday == 6)
        sup_dom = (merchant_arr == "SUPERMERCADO") & is_sunday
        expected_fee_rate = np.where(sup_dom, 0.0, expected_fee_rate)
        is_exempt |= sup_dom
        exemption_reason = np.where(sup_dom, "DOMINGO_SUPERMERCADO", exemption_reason)

        # regra: serviços digitais em janeiro → -50% na taxa
        jan = (dt.month == 1)
        serv_jan = (merchant_arr == "SERVICOS_DIGITAIS") & jan
        expected_fee_rate = np.where(serv_jan, expected_fee_rate * 0.5, expected_fee_rate)
        is_exempt |= serv_jan
        exemption_reason = np.where(serv_jan, "PROMO_JANEIRO_DIGITAL", exemption_reason)

        # anomalias na taxa cobrada (desvio aleatório multiplicativo sobre a taxa-base)
        anom_mask = (np.random.rand(n) < self.anomaly_rate)
        rand_factor = np.random.uniform(0.1, 2.0, size=n)
        charged_fee_rate = np.where(anom_mask, base_fee_rate * rand_factor, expected_fee_rate)

        expected_fee = amount * expected_fee_rate
        charged_fee = amount * charged_fee_rate

        df = pd.DataFrame({
            "transaction_id": np.char.add("TXN_", np.char.zfill(np.arange(n).astype(str), 7)),
            "date": pd.to_datetime(date_arr),
            "merchant_type": pd.Categorical(merchant_arr, categories=list(base_fee_map.keys())),
            "channel": pd.Categorical(channel_arr, categories=list(channels)),
            "amount": np.round(amount, 2),
            "base_fee_rate": base_fee_rate,
            "expected_fee_rate": expected_fee_rate,
            "charged_fee_rate": charged_fee_rate,
            "expected_fee": np.round(expected_fee, 2),
            "charged_fee": np.round(charged_fee, 2),
            "is_exempt": is_exempt,
            "exemption_reason": exemption_reason
        })

        return df

    def save_data(
        self,
        df: pd.DataFrame,
        filename: str = "transactions.csv",
        parquet: bool | str = "auto",    # True/False/"auto" → detecta engine
        partition_size: int = 250_000,
        csv_chunked: bool = True         # grava CSV em chunks no fallback
    ) -> dict:
        """
        Salva Parquet particionado (se engine disponível) ou faz fallback para CSV.
        Retorna dict com caminhos gravados (ex.: {"parquet_parts": [...], "csv_sample": "..."} ou {"csv": "..."}).
        """
        os.makedirs("data", exist_ok=True)
        out: dict = {}

        # Decide Parquet
        use_parquet = parquet_available() if parquet == "auto" else bool(parquet)
        engine = _parquet_engine() if use_parquet else None

        # Tentativa de Parquet
        if engine is not None:
            try:
                parts = []
                for i in range(0, len(df), partition_size):
                    part = df.iloc[i:i + partition_size]
                    path = f"data/transactions_part_{i // partition_size:03}.parquet"
                    part.to_parquet(path, index=False, engine=engine)
                    parts.append(path)
                # salva amostra CSV (inspeção)
                sample_csv = "data/" + filename
                df.head(min(100_000, len(df))).to_csv(sample_csv, index=False)
                out["parquet_parts"] = parts
                out["csv_sample"] = sample_csv
                return out
            except Exception as e:
                print(f"⚠ Falha ao salvar Parquet ({e}). Salvando CSV como fallback...")

        # Fallback: CSV (único ou em chunks)
        csv_path = "data/" + filename
        if csv_chunked and len(df) > partition_size:
            header = True
            for i in range(0, len(df), partition_size):
                part = df.iloc[i:i + partition_size]
                part.to_csv(csv_path, mode=("w" if header else "a"), index=False, header=header)
                header = False
        else:
            df.to_csv(csv_path, index=False)

        out["csv"] = csv_path
        return out