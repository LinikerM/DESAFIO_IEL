# data/enhanced_generator.py
import os
import pandas as pd
import numpy as np
import random
from typing import Optional, Dict, Any, Tuple

from discovery.rule_discovery import RuleDiscovery


class EnhancedDataGenerator:
    """
    Gerador de dados aprimorado que descobre regras dos dados existentes
    e aplica essas regras descobertas para gerar novos dados consistentes.
    """

    def __init__(
        self,
        n_transactions: int = 1_000_000,
        start_date: Optional[str] = None,
        months: int = 12,
        anomaly_rate: float = 0.10,
        seed: Optional[int] = None,
        use_discovered_rules: bool = True,
        # hiperparâmetros (opcionais) para a árvore de regras
        rd_min_samples_leaf: int = 200,
        rd_max_depth: int = 6,
    ):
        self.n_transactions = int(n_transactions)
        self.months = int(months)
        self.anomaly_rate = float(anomaly_rate)
        self.seed = seed
        self.use_discovered_rules = use_discovered_rules

        # Configurar datas
        if start_date is None:
            today = pd.Timestamp.now(tz="America/Sao_Paulo").tz_localize(None).normalize()
            start_date = (today - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
        self.start_date = pd.to_datetime(start_date)

        # Sistema de descoberta de regras
        self.rule_discovery = RuleDiscovery(
            min_samples_leaf=rd_min_samples_leaf,
            max_depth=rd_max_depth
        )
        self.discovered_rules: Dict[str, Any] = {}
        self.base_patterns: Dict[str, Any] = {}

        # Categorias padronizadas (devem bater entre amostra e dataset final)
        self.merchant_types = [
            "SUPERMERCADO", "SERVICOS_DIGITAIS", "RESTAURANTE", "POSTO_COMBUSTIVEL",
            "ELETRONICOS", "FARMACIA", "ONLINE_MARKETPLACE", "LOJA_ROUPAS"
        ]
        # Alinhado com a amostra e com o restante do pipeline
        self.channels = ["APP", "POS", "SITE", "TELEFONE"]

    # =========================
    # 1) AMOSTRA PARA DESCOBERTA
    # =========================
    def _generate_base_sample(self, n: int = 20_000, seed: int = 42) -> pd.DataFrame:
        """
        Gera uma amostra com PADRÕES CLAROS + pouco ruído
        (facilita a descoberta de regras e aumenta o R²).
        """
        rng = np.random.default_rng(seed)

        # Datas últimos 12 meses
        end = pd.Timestamp.today().normalize()
        start = end - pd.DateOffset(months=12)
        dates = pd.to_datetime(
            rng.integers(int(start.value / 1e9), int(end.value / 1e9), size=n),
            unit="s"
        )

        df = pd.DataFrame({
            "transaction_id": np.arange(1, n + 1),
            "date": dates,
            "merchant_type": rng.choice(self.merchant_types, size=n),
            "channel": rng.choice(self.channels, size=n),
        })

        # Valores (distribuição com cauda)
        df["amount"] = np.exp(rng.normal(5.5, 0.9, size=n)).clip(20, 50_000)

        # Features temporais
        dow = df["date"].dt.dayofweek   # 0=seg ... 6=dom
        month = df["date"].dt.month

        # 1) Taxa base por tipo (sinal forte por merchant)
        base_fee_by_merchant = {
            "SUPERMERCADO":       0.031,
            "SERVICOS_DIGITAIS":  0.025,
            "RESTAURANTE":        0.023,
            "POSTO_COMBUSTIVEL":  0.029,
            "ELETRONICOS":        0.016,
            "FARMACIA":           0.023,
            "ONLINE_MARKETPLACE": 0.023,
            "LOJA_ROUPAS":        0.034,
        }

        # 2) Ajuste por canal (sinal moderado)
        channel_adj = {
            "APP":      -0.0020,
            "POS":       0.0000,
            "SITE":     -0.0010,
            "TELEFONE":  0.0015,
        }

        # Base: merchant + canal
        base = df["merchant_type"].map(base_fee_by_merchant).astype(float)
        base += df["channel"].map(channel_adj).astype(float)

        # 3) Alto valor com thresholds por merchant (mais realista)
        hv_threshold = np.where(df["merchant_type"].astype(str).values == "ELETRONICOS", 3000, 5000)
        is_high_value = df["amount"].values > hv_threshold
        base = np.where(is_high_value, base * 0.85, base)  # ~ -15% na taxa

        # 4) Regras de negócio claras (já existentes)
        # SUPERMERCADO aos domingos → isento (taxa 0)
        is_super_domingo = (df["merchant_type"] == "SUPERMERCADO") & (dow == 6)
        base = np.where(is_super_domingo, 0.0, base)
        # SERVICOS_DIGITAIS em janeiro → -50%
        is_sd_jan = (df["merchant_type"] == "SERVICOS_DIGITAIS") & (month == 1)
        base = np.where(is_sd_jan, base * 0.50, base)

        # ----- EFEITOS ADICIONAIS PARA SIMULAR "SINAL REAL" -----
        # 5) Efeito por dia da semana (poucos p.p. para árvore “ver”)
        dow_offsets = np.array([
            0.0010,  # seg  +0,10 pp
            0.0005,  # ter  +0,05 pp
            0.0000,  # qua   0
            -0.0003, # qui  -0,03 pp
            0.0006,  # sex  +0,06 pp
            0.0008,  # sáb  +0,08 pp
            -0.0012  # dom  -0,12 pp (mais barato)
        ])
        base = base + dow_offsets[dow]

        # 6) Sazonalidade (BF/Natal/férias)
        month_offsets = np.zeros(len(df))
        month_offsets[(month == 11)] += 0.0007  # novembro (Black Friday) +0,07 pp
        month_offsets[(month == 12)] += 0.0009  # dezembro (Natal)      +0,09 pp
        month_offsets[(month == 1)]  += 0.0003  # janeiro (férias)      +0,03 pp
        base = base + month_offsets

        # 7) Efeito por FAIXA DE VALOR (bins)
        amount = df["amount"].values
        bins = np.array([0, 150, 300, 1000, 1e12])  # faixas coerentes com o mix
        # efeitos por faixa (ordem: [0-150), [150-300), [300-1000), [>=1000])
        bin_effects = np.array([0.0010, 0.0005, -0.0002, -0.0008])  # +/- em pp
        bin_idx = np.digitize(amount, bins) - 1
        base = base + bin_effects[bin_idx]

        # 8) Interações leves (opcionais, mas úteis)
        # 8a) Domingo + compra baixa (<=150) => ainda mais barato
        low_val = amount <= 150
        base = np.where((dow == 6) & low_val, base - 0.0005, base)

        # 8b) Black Friday (nov) + canal SITE => leve queda (promo online)
        is_nov = (month == 11)
        is_site = (df["channel"].astype(str).values == "SITE")
        base = np.where(is_nov & is_site, base - 0.0004, base)

        # Clipping de segurança (antes do ruído)
        base = np.clip(base, 0.0, 0.06)
        # ---------------------------------------------------------

        # 9) Pouco ruído (não matar o sinal)
        noise = rng.normal(0.0, 0.0015, size=n)           # ~ ±0,15 pp
        expected_rate = (base + noise).clip(0.0, 0.06)

        # 10) Taxa cobrada = esperada + pequena variação + anomalias raras/fortes
        rng2 = np.random.default_rng(seed + 1)
        anomaly_mask = rng2.random(n) < 0.05               # 5% anomalias
        anomaly_factor = rng2.choice([0.0, 0.5, 1.5, 2.0],
                                     p=[0.10, 0.15, 0.45, 0.30], size=n)
        charged_rate = expected_rate * (1.0 + rng2.normal(0.0, 0.0020, size=n))
        charged_rate = np.where(anomaly_mask, expected_rate * anomaly_factor, charged_rate)
        charged_rate = charged_rate.clip(0.0, 0.08)

        df["expected_fee_rate"] = expected_rate
        df["charged_fee_rate"] = charged_rate
        df["expected_fee"] = (df["amount"] * df["expected_fee_rate"]).round(2)
        df["charged_fee"] = (df["amount"] * df["charged_fee_rate"]).round(2)
        df["is_anomaly"] = anomaly_mask.astype(bool)
        df["anomalies"] = df["is_anomaly"].astype(int)

        return df

    # =========================
    # 2) DESCOBERTA DE REGRAS
    # =========================
    def discover_rules_from_sample(self, sample_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Descobre regras a partir de uma amostra de dados.
        """
        print("🔍 Descobrindo regras de negócio a partir dos dados...")

        if sample_df is None:
            print("Gerando amostra base para descoberta de regras...")
            sample_df = self._generate_base_sample()

        # Executar descoberta de regras
        rules = self.rule_discovery.discover_fee_rules(sample_df)
        self.discovered_rules = rules

        # Salvar regras descobertas
        os.makedirs("reports", exist_ok=True)
        self.rule_discovery.save_discovered_rules()

        print("✅ Regras descobertas com sucesso!")
        print("\n📋 Resumo das regras principais:")
        business_rules = self.rule_discovery.generate_business_rules()
        for rule in business_rules[:10]:  # Mostrar apenas as primeiras 10
            print(rule)

        return rules

    # =========================
    # 3) GERAÇÃO COM REGRAS
    # =========================
    def generate_transactions_with_discovered_rules(self) -> pd.DataFrame:
        """
        Gera transações completas usando regras descobertas.
        """
        print("🏭 Gerando transações com regras descobertas...")

        # Se não temos regras descobertas, descobrir primeiro
        if not self.discovered_rules:
            sample = self._generate_base_sample(100_000)  # Amostra maior para melhor descoberta
            self.discover_rules_from_sample(sample)

        # Gerar dataset completo
        full_df = self._generate_full_dataset()

        # Aplicar regras descobertas
        if self.use_discovered_rules and self.rule_discovery.rule_tree is not None:
            print("📐 Aplicando regras descobertas...")

            # Predizer taxas esperadas com base nas regras descobertas
            predicted_rates = self.rule_discovery.predict_expected_fee_rate(full_df).clip(0.0, 0.06)
            full_df['expected_fee_rate'] = predicted_rates
            full_df['expected_fee'] = (full_df['amount'] * predicted_rates).round(2)

            # Simular taxas cobradas com base nas esperadas + ruído/anomalias
            rng = np.random.default_rng(self.seed or 42)
            charged_rates = predicted_rates * (1.0 + rng.normal(0.0, 0.0020, size=len(full_df)))

            # Anomalias (5–8% recomendado)
            anomaly_mask = rng.random(len(full_df)) < self.anomaly_rate
            anomaly_factor = rng.choice([0.0, 0.5, 1.5, 2.0],
                                        p=[0.10, 0.15, 0.45, 0.30],
                                        size=len(full_df))
            charged_rates = np.where(anomaly_mask,
                                     predicted_rates * anomaly_factor,
                                     charged_rates)
            charged_rates = np.clip(charged_rates, 0.0, 0.08)

            full_df['charged_fee_rate'] = charged_rates
            full_df['charged_fee'] = (full_df['amount'] * charged_rates).round(2)

            # marcar anomalias para uso posterior (PDF/relatórios)
            full_df["is_anomaly"] = anomaly_mask.astype(bool)
            full_df["anomalies"] = full_df["is_anomaly"].astype(int)

        return full_df

    # =========================
    # 4) DATASET FINAL (estrutura)
    # =========================
    def _generate_full_dataset(self) -> pd.DataFrame:
        """
        Gera o dataset completo com a estrutura básica (datas, merchants, canais, valores).
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        merchant_types = self.merchant_types
        channels = self.channels
        dates_vec = self._make_calendar()

        n = self.n_transactions

        # Distribuições de escolha
        mt_weights = [0.20, 0.05, 0.18, 0.15, 0.10, 0.08, 0.12, 0.12]  # soma 1.0
        mt_idx = np.random.choice(len(merchant_types), size=n, p=mt_weights)

        # Canais alinhados com a amostra
        ch_weights = [0.30, 0.30, 0.35, 0.05]  # APP cresce, SITE/POS relevantes
        ch_idx = np.random.choice(len(channels), size=n, p=ch_weights)

        # Datas com sazonalidade
        dt_idx = self._generate_seasonal_dates(n, dates_vec)

        merchant_arr = np.array(merchant_types)[mt_idx]
        channel_arr = np.array(channels)[ch_idx]
        date_arr = dates_vec[dt_idx].astype("datetime64[ns]")

        # Valores com distribuições específicas por estabelecimento
        amount = self._generate_realistic_amounts(merchant_arr, n)

        # DataFrame inicial
        df = pd.DataFrame({
            "transaction_id": np.char.add("TXN_", np.char.zfill(np.arange(n).astype(str), 8)),
            "date": pd.to_datetime(date_arr),
            "merchant_type": pd.Categorical(merchant_arr, categories=merchant_types),
            "channel": pd.Categorical(channel_arr, categories=channels),
            "amount": np.round(amount, 2),
            "expected_fee_rate": 0.0,  # será preenchido pelas regras descobertas
            "charged_fee_rate": 0.0,   # será preenchido pelas regras descobertas
            "expected_fee": 0.0,
            "charged_fee": 0.0,
            "is_exempt": False,
            "exemption_reason": None
        })

        return df

    def _generate_seasonal_dates(self, n: int, dates_vec: np.ndarray) -> np.ndarray:
        """
        Gera índices de datas com sazonalidade (mais transações em certas épocas).
        """
        # Criar pesos sazonais
        date_weights = np.ones(len(dates_vec), dtype=float)

        for i, date in enumerate(dates_vec):
            dt = pd.to_datetime(date)

            # Black Friday (novembro)
            if dt.month == 11:
                date_weights[i] *= 1.5

            # Natal (dezembro)
            elif dt.month == 12:
                date_weights[i] *= 1.8

            # Férias (janeiro)
            elif dt.month == 1:
                date_weights[i] *= 1.2

            # Dia das mães (maio)
            elif dt.month == 5:
                date_weights[i] *= 1.3

            # Fim de semana (menos transações comerciais)
            if dt.weekday() >= 5:  # Sábado e domingo
                date_weights[i] *= 0.7

        # Normalizar pesos
        date_weights = date_weights / date_weights.sum()

        # Gerar índices com base nos pesos
        return np.random.choice(len(dates_vec), size=n, p=date_weights)

    def _generate_realistic_amounts(self, merchant_arr: np.ndarray, n: int) -> np.ndarray:
        """
        Gera valores de transação realistas por tipo de estabelecimento.
        """
        amounts = np.zeros(n, dtype=float)

        # Parâmetros por tipo de estabelecimento (média log, desvio)
        amount_params = {
            "SUPERMERCADO": (3.5, 1.2),        # R$ 30-300 típico
            "POSTO_COMBUSTIVEL": (4.2, 0.8),   # R$ 60-200 típico
            "RESTAURANTE": (3.8, 1.0),         # R$ 40-150 típico
            "FARMACIA": (3.0, 1.3),            # R$ 20-200 típico
            "LOJA_ROUPAS": (4.5, 1.4),         # R$ 80-500 típico
            "ELETRONICOS": (6.0, 1.8),         # R$ 400-5000 típico
            "ONLINE_MARKETPLACE": (4.8, 1.6),  # R$ 100-1000 típico
            "SERVICOS_DIGITAIS": (2.5, 1.0)    # R$ 10-50 típico
        }

        for merchant in np.unique(merchant_arr):
            mask = merchant_arr == merchant
            count = int(mask.sum())

            if merchant in amount_params:
                mean_log, std_log = amount_params[merchant]
                amounts[mask] = np.random.lognormal(mean_log, std_log, count)
            else:
                # Fallback padrão
                amounts[mask] = np.random.lognormal(4.0, 1.5, count)

        return amounts

    def _make_calendar(self) -> np.ndarray:
        """Cria calendário do período configurado."""
        end_date = self.start_date + pd.DateOffset(months=self.months) - pd.Timedelta(days=1)
        dates = pd.date_range(self.start_date, end_date, freq="D")
        return dates.values

    # =========================
    # 5) SAÍDA E RELATÓRIOS
    # =========================
    def save_data_with_rules(
        self,
        df: pd.DataFrame,
        filename: str = "transactions_discovered_rules.csv",
        include_rule_info: bool = True
    ) -> Dict[str, Any]:
        """
        Salva dados com informações sobre regras descobertas.
        """
        os.makedirs("data", exist_ok=True)

        # Adicionar colunas sobre as regras se solicitado
        if include_rule_info and self.rule_discovery.rule_tree is not None:
            # Predizer taxas com regras descobertas
            rule_predicted_rates = self.rule_discovery.predict_expected_fee_rate(df).clip(0.0, 0.06)
            df_with_rules = df.copy()
            df_with_rules['rule_predicted_rate'] = rule_predicted_rates
            df_with_rules['rule_vs_expected_diff'] = (
                df_with_rules['expected_fee_rate'] - rule_predicted_rates
            )
            df_with_rules['is_rule_compliant'] = (
                df_with_rules['rule_vs_expected_diff'].abs() < 0.005  # 0.5% tolerância
            )
        else:
            df_with_rules = df

        # Salvar CSV
        csv_path = f"data/{filename}"
        df_with_rules.to_csv(csv_path, index=False)

        # Salvar também em chunks se muito grande
        out = {"csv": csv_path}

        if len(df_with_rules) > 500_000:
            chunk_size = 250_000
            parts = []
            for i in range(0, len(df_with_rules), chunk_size):
                part = df_with_rules.iloc[i:i + chunk_size]
                part_path = f"data/{filename.replace('.csv', f'_part_{i//chunk_size:03}.csv')}"
                part.to_csv(part_path, index=False)
                parts.append(part_path)
            out["csv_parts"] = parts

        return out

    def generate_comparison_report(self, original_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Gera relatório comparando abordagem de regras fixas vs regras descobertas.
        """
        print("📊 Gerando relatório de comparação...")

        # Se não temos DataFrame original, gerar um com regras fixas
        if original_df is None:
            from data.generator import DataGenerator  # Import do sistema original
            original_gen = DataGenerator(
                n_transactions=min(100_000, self.n_transactions),
                seed=self.seed
            )
            original_df = original_gen.generate_transactions()

        # Gerar com regras descobertas
        discovered_df = self.generate_transactions_with_discovered_rules()

        # Comparar distribuições
        comparison = {
            "original_approach": {
                "total_transactions": len(original_df),
                "mean_fee_rate": original_df['charged_fee_rate'].mean(),
                "std_fee_rate": original_df['charged_fee_rate'].std(),
                "exemption_rate": original_df['is_exempt'].mean() if 'is_exempt' in original_df.columns else 0,
                "merchant_distribution": original_df['merchant_type'].value_counts().to_dict(),
                "fee_rate_by_merchant": original_df.groupby('merchant_type')['charged_fee_rate'].mean().to_dict()
            },
            "discovered_rules_approach": {
                "total_transactions": len(discovered_df),
                "mean_fee_rate": discovered_df['charged_fee_rate'].mean(),
                "std_fee_rate": discovered_df['charged_fee_rate'].std(),
                "exemption_rate": discovered_df['is_exempt'].mean() if 'is_exempt' in discovered_df.columns else 0,
                "merchant_distribution": discovered_df['merchant_type'].value_counts().to_dict(),
                "fee_rate_by_merchant": discovered_df.groupby('merchant_type')['charged_fee_rate'].mean().to_dict()
            },
            "discovered_rules_quality": self.discovered_rules.get('model_performance', {}),
            "business_rules": self.rule_discovery.generate_business_rules()
        }

        # Salvar relatório
        os.makedirs("reports", exist_ok=True)

        with open("reports/rules_comparison_report.txt", "w", encoding="utf-8") as f:
            f.write("RELATÓRIO DE COMPARAÇÃO: REGRAS FIXAS vs REGRAS DESCOBERTAS\n")
            f.write("=" * 60 + "\n\n")

            f.write("ABORDAGEM ORIGINAL (regras fixas):\n")
            f.write(f"- Taxa média: {comparison['original_approach']['mean_fee_rate']:.1%}\n")
            f.write(f"- Desvio padrão: {comparison['original_approach']['std_fee_rate']:.1%}\n")
            f.write(f"- Taxa de isenção: {comparison['original_approach']['exemption_rate']:.1%}\n\n")

            f.write("ABORDAGEM DESCOBERTA (regras dos dados):\n")
            f.write(f"- Taxa média: {comparison['discovered_rules_approach']['mean_fee_rate']:.1%}\n")
            f.write(f"- Desvio padrão: {comparison['discovered_rules_approach']['std_fee_rate']:.1%}\n")
            f.write(f"- Taxa de isenção: {comparison['discovered_rules_approach']['exemption_rate']:.1%}\n\n")

            if 'model_performance' in self.discovered_rules:
                perf = self.discovered_rules['model_performance']
                f.write("QUALIDADE DAS REGRAS DESCOBERTAS:\n")
                f.write(f"- R² Score: {perf.get('r2_score', 0):.1%}\n")
                f.write(f"- Erro médio (MAE): {perf.get('mae', 0):.1%}\n")
                f.write(f"- MAPE: {perf.get('mean_absolute_percentage_error', 0):.1f}%\n\n")

            f.write("REGRAS DE NEGÓCIO DESCOBERTAS:\n")
            for rule in comparison['business_rules']:
                f.write(f"{rule}\n")

        return comparison


# =========================
# Função de conveniência
# =========================
def generate_with_discovered_rules(
    n_transactions: int = 1_000_000,
    months: int = 12,
    anomaly_rate: float = 0.10,
    seed: Optional[int] = None,
    save_files: bool = True
) -> Tuple[pd.DataFrame, RuleDiscovery, Dict[str, Any]]:
    """
    Função principal para gerar dados com regras descobertas

    Returns:
        - DataFrame com transações
        - Objeto RuleDiscovery com regras descobertas
        - Dicionário com informações sobre arquivos salvos
    """
    print("🚀 Iniciando geração de dados com descoberta automática de regras...")

    # Criar gerador aprimorado
    generator = EnhancedDataGenerator(
        n_transactions=n_transactions,
        months=months,
        anomaly_rate=anomaly_rate,
        seed=seed,
        use_discovered_rules=True
    )

    # Gerar transações com regras descobertas
    df = generator.generate_transactions_with_discovered_rules()

    # Salvar dados se solicitado
    files_info: Dict[str, Any] = {}
    if save_files:
        files_info = generator.save_data_with_rules(df, include_rule_info=True)

        # Gerar relatório de comparação
        generator.generate_comparison_report()
        files_info['comparison_report'] = "reports/rules_comparison_report.txt"
        files_info['discovered_rules'] = "reports/discovered_rules.txt"

    print("✅ Geração concluída com sucesso!")
    print(f"📈 {len(df):,} transações geradas")
    print(f"🎯 Regras descobertas com R² = {generator.discovered_rules.get('model_performance', {}).get('r2_score', 0):.1%}")

    return df, generator.rule_discovery, files_info
