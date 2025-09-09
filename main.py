#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaySmart Solutions - Sistema de Análise de Cobranças
"""

import os
import glob
import math
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend offscreen para salvar imagens
import matplotlib.pyplot as plt

# Dependências opcionais para UI do terminal (com fallback)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
    from rich.theme import Theme
    RICH_OK = True
except Exception:
    RICH_OK = False
    Console = None

try:
    from tqdm import tqdm
    TQDM_OK = True
except Exception:
    TQDM_OK = False

# --- módulos do projeto
from data.generator import DataGenerator
from models.anomaly_detector import AnomalyDetector
from reports.pdf_generator import PDFReportGenerator


# ================== Configurações ==================
TZ = "America/Sao_Paulo"
DETECTION_MODE = "hybrid"
CSV_PATH = "data/transactions.csv"
PARQUET_GLOB = "data/transactions_part_*.parquet"
CSV_CHUNKSIZE = 200_000
ML_CHUNK_SCORING = 100_000
FORCE_REGENERATE = True
SHOW_PLOTS = False  # manter False (execução automática)
N_TRANSACTIONS_GEN = 1_000_000
GEN_MONTHS = 12
HIST_MIN, HIST_MAX, HIST_BINS = -100.0, 100.0, 80

# ======= Identidade visual dos gráficos (paleta & estilo) =======
PALETTE = {
    "primary":  "#184C7A",  # navy
    "accent":   "#05A6AB",  # teal
    "warning":  "#FFB300",  # amarelo
    "muted":    "#6B7A90",
    "red":      "#D14D57",
    "green":    "#2E8B57",
    "ink":      "#0F172A",
}
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 110,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": "#E5E7EB",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.alpha": 0.45,
    "font.family": "DejaVu Sans",
    "text.color": PALETTE["ink"],
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"],
    "axes.prop_cycle": plt.cycler(color=[PALETTE["primary"], PALETTE["accent"], PALETTE["warning"]]),
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ================== Helpers de UI (console) ==================
if RICH_OK:
    console = Console(theme=Theme({
        "ok": "bold " + PALETTE["accent"],
        "warn": "bold " + PALETTE["warning"],
        "err": "bold " + PALETTE["red"],
        "muted": PALETTE["muted"],
        "title": "bold " + PALETTE["primary"]
    }))
else:
    class _Dummy:
        def print(self, *a, **k):  # fallback simples
            print(*a)
    console = _Dummy()

def banner(title: str, subtitle: str = ""):
    if RICH_OK:
        console.print(Panel.fit(
            f"[title]{title}[/title]\n[muted]{subtitle}[/muted]",
            border_style=PALETTE["primary"]
        ))
    else:
        print("=" * 60)
        print(title)
        if subtitle:
            print(subtitle)
        print("=" * 60)

def step(msg: str):
    if RICH_OK:
        console.print(f"[ok]• {msg}[/ok]")
    else:
        print(f"• {msg}")

def warn(msg: str):
    if RICH_OK:
        console.print(f"[warn]⚠ {msg}[/warn]")
    else:
        print(f"⚠ {msg}")

def err(msg: str):
    if RICH_OK:
        console.print(f"[err]✖ {msg}[/err]")
    else:
        print(f"✖ {msg}")

def success(msg: str):
    if RICH_OK:
        console.print(f"[ok]✅ {msg}[/ok]")
    else:
        print(f"✅ {msg}")

# ================== Funções utilitárias ==================
def create_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

def prune_old_base():
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    for p in glob.glob(PARQUET_GLOB):
        os.remove(p)

def ensure_fresh_base():
    have_parquet = bool(sorted(glob.glob(PARQUET_GLOB)))
    have_csv = os.path.exists(CSV_PATH)
    if FORCE_REGENERATE or not (have_parquet or have_csv):
        if FORCE_REGENERATE:
            step("Forçando regeneração: apagando base antiga…")
            prune_old_base()
        step("Gerando base sintética (últimos 12 meses)…")
        gen = DataGenerator(n_transactions=N_TRANSACTIONS_GEN, start_date=None, months=GEN_MONTHS)
        df = gen.generate_transactions()
        gen.save_data(df, parquet="auto", partition_size=250_000)
        success("Base sintética gerada e particionada.")

def iter_dataframes():
    """Itera dataframes sem carregar tudo na memória (com barra de progresso se possível)."""
    ensure_fresh_base()
    parquet_parts = sorted(glob.glob(PARQUET_GLOB))
    if parquet_parts:
        iterator = parquet_parts
        if RICH_OK:
            with Progress(
                SpinnerColumn(style=PALETTE["accent"]),
                TextColumn("[muted]Lendo partições Parquet[/muted]"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("read_parquet", total=len(parquet_parts))
                for path in iterator:
                    df = pd.read_parquet(path)
                    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    progress.advance(task, 1)
                    yield df
        else:
            bar = tqdm(iterator, desc="Lendo Parquet", unit="parte") if TQDM_OK else iterator
            for path in bar:
                df = pd.read_parquet(path)
                if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                yield df
        return

    if os.path.exists(CSV_PATH):
        # Sem como estimar total de chunks; exibir spinner barra “infinita” com rich
        if RICH_OK:
            with Progress(
                SpinnerColumn(style=PALETTE["accent"]),
                TextColumn("[muted]Lendo CSV em chunks[/muted]"),
                transient=True,
            ) as progress:
                task = progress.add_task("read_csv", total=None)
                for df in pd.read_csv(CSV_PATH, chunksize=CSV_CHUNKSIZE, parse_dates=["date"]):
                    progress.advance(task, 1)
                    yield df
        else:
            iterator = pd.read_csv(CSV_PATH, chunksize=CSV_CHUNKSIZE, parse_dates=["date"])
            for df in (tqdm(iterator, desc="Lendo CSV", unit="chunk") if TQDM_OK else iterator):
                yield df
        return

    raise FileNotFoundError("Nenhuma base encontrada após tentativa de geração.")

def running_agg_init():
    return {"n_rows": 0, "sum_diff": 0.0, "exempt_total": 0, "exempt_correct": 0,
            "anom_count": 0, "merchant_summary": {}}

def update_merchant_summary(agg, df):
    by_merchant = df.groupby("merchant_type", dropna=False).agg(
        transactions=("transaction_id", "count"),
        amount_sum=("amount", "sum"),
        expected_fee_sum=("expected_fee", "sum"),
        charged_fee_sum=("charged_fee", "sum"),
        anomalies=("is_anomaly", "sum"),
    ).reset_index()
    for _, r in by_merchant.iterrows():
        k = r["merchant_type"]
        cur = agg["merchant_summary"].get(k, {"transactions": 0, "amount_sum": 0.0,
                                              "expected_fee_sum": 0.0, "charged_fee_sum": 0.0, "anomalies": 0})
        cur["transactions"] += int(r["transactions"])
        cur["amount_sum"] += float(r["amount_sum"])
        cur["expected_fee_sum"] += float(r["expected_fee_sum"])
        cur["charged_fee_sum"] += float(r["charged_fee_sum"])
        cur["anomalies"] += int(r["anomalies"])
        agg["merchant_summary"][k] = cur

def save_problematic_append(df_anom_only):
    out_path = "reports/problematic_transactions.csv"
    header = not os.path.exists(out_path)
    df_anom_only.to_csv(out_path, mode="a", index=False, header=header)

def finalize_merchant_summary(agg):
    rows = []
    for k, v in agg["merchant_summary"].items():
        rows.append({
            "merchant_type": k,
            "transactions": v["transactions"],
            "amount_sum": v["amount_sum"],
            "expected_fee_sum": v["expected_fee_sum"],
            "charged_fee_sum": v["charged_fee_sum"],
            "anomalies": v["anomalies"],
            "fee_diff_sum": v["charged_fee_sum"] - v["expected_fee_sum"],
        })
    df = pd.DataFrame(rows).sort_values("transactions", ascending=False)
    df.to_csv("reports/merchant_summary.csv", index=False)
    return df

def init_window_aggs():
    return {
        "trend": defaultdict(lambda: {"total": 0, "anom": 0, "diff": 0.0, "ex_total": 0, "ex_ok": 0}),
        "merchant": defaultdict(lambda: {"total": 0, "anom": 0, "diff": 0.0}),
    }

def update_window_aggs(window_aggs, df):
    g_day = df.groupby(df["date"].dt.floor("D")).agg(
        total=("transaction_id", "count"),
        anom=("is_anomaly", "sum"),
        charged_sum=("charged_fee", "sum"),
        expected_sum=("expected_fee", "sum"),
        ex_total=("expected_fee", lambda s: (s == 0).sum()),
        ex_ok=("charged_fee", lambda s: (s.abs() < 0.01).sum()),
    ).reset_index()
    g_day["diff"] = g_day["charged_sum"] - g_day["expected_sum"]

    for _, r in g_day.iterrows():
        k = r["date"].date()
        window_aggs["trend"][k]["total"] += int(r["total"])
        window_aggs["trend"][k]["anom"] += int(r["anom"])
        window_aggs["trend"][k]["diff"] += float(r["diff"])
        window_aggs["trend"][k]["ex_total"] += int(r["ex_total"])
        window_aggs["trend"][k]["ex_ok"] += int(r["ex_ok"])

    g_merch = df.groupby("merchant_type", dropna=False).agg(
        total=("transaction_id", "count"),
        anom=("is_anomaly", "sum"),
        charged_sum=("charged_fee", "sum"),
        expected_sum=("expected_fee", "sum"),
    ).reset_index()
    g_merch["diff"] = g_merch["charged_sum"] - g_merch["expected_sum"]

    for _, r in g_merch.iterrows():
        k = r["merchant_type"]
        window_aggs["merchant"][k]["total"] += int(r["total"])
        window_aggs["merchant"][k]["anom"] += int(r["anom"])
        window_aggs["merchant"][k]["diff"] += float(r["diff"])

def make_df_from_aggs(trend_dict, merchant_dict):
    if trend_dict:
        trend_rows = [{"date": k, **v} for k, v in sorted(trend_dict.items())]
        df_trend = pd.DataFrame(trend_rows)
        if not df_trend.empty:
            df_trend["anomaly_rate"] = df_trend["anom"] / df_trend["total"].replace(0, np.nan)
            df_trend["exemption_accuracy"] = df_trend["ex_ok"] / df_trend["ex_total"].replace(0, np.nan)
        else:
            df_trend = pd.DataFrame(columns=["date", "total", "anom", "diff", "anomaly_rate", "exemption_accuracy"])
    else:
        df_trend = pd.DataFrame(columns=["date", "total", "anom", "diff", "anomaly_rate", "exemption_accuracy"])

    if merchant_dict:
        merch_rows = [{"merchant_type": k, **v} for k, v in merchant_dict.items()]
        df_merch = pd.DataFrame(merch_rows)
        if not df_merch.empty and "total" in df_merch.columns:
            df_merch = df_merch.sort_values("total", ascending=False)
            df_merch["anomaly_rate"] = df_merch["anom"] / df_merch["total"].replace(0, np.nan)
        else:
            df_merch = pd.DataFrame(columns=["merchant_type", "total", "anom", "diff", "anomaly_rate"])
    else:
        df_merch = pd.DataFrame(columns=["merchant_type", "total", "anom", "diff", "anomaly_rate"])

    return df_trend, df_merch

def init_global_aggs():
    bin_edges = np.linspace(HIST_MIN, HIST_MAX, HIST_BINS + 1)
    return {
        "hist_edges": bin_edges,
        "hist_counts": np.zeros(HIST_BINS, dtype=np.int64),
        "daily": defaultdict(lambda: {"total": 0, "anom": 0, "diff": 0.0}),
        "monthly": defaultdict(lambda: {"total": 0, "anom": 0, "ex_total": 0, "ex_ok": 0}),
    }

def update_global_aggs(g, df):
    diff = (df["charged_fee"] - df["expected_fee"]).to_numpy(dtype=float)
    diff = np.clip(diff, HIST_MIN, HIST_MAX)
    counts, _ = np.histogram(diff, bins=g["hist_edges"])
    g["hist_counts"] += counts

    gd = df.groupby(df["date"].dt.floor("D")).agg(
        total=("transaction_id", "count"),
        anom=("is_anomaly", "sum"),
        charged_sum=("charged_fee", "sum"),
        expected_sum=("expected_fee", "sum"),
    ).reset_index()
    gd["diff"] = gd["charged_sum"] - gd["expected_sum"]
    for _, r in gd.iterrows():
        k = r["date"].date()
        g["daily"][k]["total"] += int(r["total"])
        g["daily"][k]["anom"] += int(r["anom"])
        g["daily"][k]["diff"] += float(r["diff"])

    gm = df.groupby(df["date"].dt.to_period("M")).agg(
        total=("transaction_id", "count"),
        anom=("is_anomaly", "sum"),
        ex_total=("expected_fee", lambda s: (s == 0).sum()),
        ex_ok=("charged_fee", lambda s: (s.abs() < 0.01).sum()),
    ).reset_index()
    gm["month"] = gm["date"].astype(str)
    for _, r in gm.iterrows():
        k = r["month"]
        g["monthly"][k]["total"] += int(r["total"])
        g["monthly"][k]["anom"] += int(r["anom"])
        g["monthly"][k]["ex_total"] += int(r["ex_total"])
        g["monthly"][k]["ex_ok"] += int(r["ex_ok"])

def make_global_dfs(g):
    if g["daily"]:
        rows = [{"date": k, **v} for k, v in sorted(g["daily"].items())]
        df_daily = pd.DataFrame(rows)
        df_daily["anomaly_rate"] = df_daily["anom"] / df_daily["total"].replace(0, np.nan)
    else:
        df_daily = pd.DataFrame(columns=["date", "total", "anom", "diff", "anomaly_rate"])

    if g["monthly"]:
        rows = [{"month": k, **v} for k, v in sorted(g["monthly"].items())]
        df_month = pd.DataFrame(rows)
        df_month["exemption_accuracy"] = df_month["ex_ok"] / df_month["ex_total"].replace(0, np.nan)
    else:
        df_month = pd.DataFrame(columns=["month", "total", "anom", "ex_total", "ex_ok", "exemption_accuracy"])

    edges = g["hist_edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = g["hist_counts"].astype(int)
    return df_daily, df_month, centers, counts

# ================== Visualização (PDF-friendly) ==================
def _apply_common_chart_decor():
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_chart_with_style(filename, title, xlabel, ylabel, subtitle=None):
    plt.title(title, pad=16, color=PALETTE["primary"])
    if subtitle:
        plt.suptitle(subtitle, y=0.97, fontsize=10, color=PALETTE["muted"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _apply_common_chart_decor()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def plot_core_diagnostics(df_daily, df_month, hist_x, hist_counts, merchant_summary_df):
    # A) Histograma
    if hist_counts.sum() > 0:
        plt.figure(figsize=(10, 6))
        binw = hist_x[1] - hist_x[0] if len(hist_x) > 1 else 1.0
        plt.bar(hist_x, hist_counts, width=binw, color=PALETTE["accent"], alpha=0.85, edgecolor="white")
        plt.axvline(x=0, color=PALETTE["red"], linestyle="--", linewidth=2, label="Zero (Cobrança Correta)")
        plt.legend(frameon=False, loc="upper right")
        save_chart_with_style(
            "reports/fee_diff_hist.png",
            "Distribuição das Diferenças de Cobrança",
            "Diferença (R$ cobrado - R$ esperado)",
            "Frequência",
            subtitle="Faixa limitada para visualização: [{:.0f}, {:.0f}]".format(HIST_MIN, HIST_MAX)
        )

    # B) Acurácia mensal de isenções
    if not df_month.empty and "exemption_accuracy" in df_month.columns:
        plt.figure(figsize=(12, 6))
        dfm = df_month.copy()
        dfm["month_dt"] = pd.to_datetime(dfm["month"] + "-01")
        dfm = dfm.sort_values("month_dt")
        y = dfm["exemption_accuracy"].fillna(0.0).clip(0, 1)
        plt.plot(dfm["month_dt"], y, marker="o", linewidth=2, markersize=6, color=PALETTE["green"])
        plt.ylim(0, 1.05)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
        save_chart_with_style(
            "reports/exemption_accuracy_monthly.png",
            "Acurácia de Isenções por Mês",
            "Mês",
            "Acurácia (%)"
        )

    # C) Timeline de anomalias
    if not df_daily.empty:
        plt.figure(figsize=(14, 6))
        x = pd.to_datetime(df_daily["date"])
        y = df_daily["anom"]
        plt.plot(x, y, color=PALETTE["red"], linewidth=1.8)
        plt.fill_between(x, y, alpha=0.15, color=PALETTE["red"])
        save_chart_with_style(
            "reports/anomalies_timeline_12m.png",
            "Anomalias Detectadas por Dia (Últimos 12 Meses)",
            "Data",
            "Qtde de Anomalias"
        )

    # D) Top merchants por anomalias
    if merchant_summary_df is not None and not merchant_summary_df.empty:
        top = merchant_summary_df.nlargest(10, "anomalies")
        if not top.empty:
            plt.figure(figsize=(12, 7.5))
            bars = plt.bar(range(len(top)), top["anomalies"], color=PALETTE["warning"], edgecolor="#AA7A00", linewidth=0.8)
            plt.xticks(range(len(top)), [str(x) for x in top["merchant_type"]], rotation=35, ha="right")
            # labels nas barras
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., h, f"{int(h):,}".replace(",", "."), ha="center", va="bottom", fontweight="bold", color=PALETTE["ink"])
            save_chart_with_style(
                "reports/top_merchants_anomalies.png",
                "Top 10 Estabelecimentos por Número de Anomalias",
                "Tipo de Estabelecimento",
                "Número de Anomalias"
            )

def write_executive_report(agg, merchant_summary_df, df_month):
    total = agg["n_rows"]
    total_diff = agg["sum_diff"]
    anom = agg["anom_count"]
    exempt_acc = (agg["exempt_correct"] / agg["exempt_total"]) if agg["exempt_total"] > 0 else float("nan")
    anom_rate = (anom / total) if total else 0.0

    lines = [
        "PAYSMART SOLUTIONS - RELATÓRIO EXECUTIVO",
        "=" * 50,
        f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}",
        "",
        "RESUMO EXECUTIVO",
        f"Total de transações: {total:,}".replace(",", "."),
        f"Impacto financeiro: R$ {total_diff:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        f"Taxa de anomalias: {anom_rate:.2%}",
        f"Acurácia de isenções: {exempt_acc:.2%}" if not math.isnan(exempt_acc) else "Acurácia de isenções: N/D",
        "",
        "ARQUIVOS GERADOS:",
        "• paysmart_traditional_report.pdf - Relatório completo com gráficos",
        "• problematic_transactions.csv - Transações problemáticas",
        "• merchant_summary.csv - Resumo por estabelecimento",
        "• Gráficos PNG salvos em reports/"
    ]
    with open("reports/executive_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def print_final_summary(agg, merchant_summary_df, pdf_ok: bool):
    total = agg["n_rows"]
    total_diff = agg["sum_diff"]
    anom = agg["anom_count"]
    exempt_acc = (agg["exempt_correct"] / agg["exempt_total"]) if agg["exempt_total"] > 0 else float("nan")
    anom_rate = (anom / total) if total else 0.0

    if RICH_OK:
        table = Table(title="Resumo Executivo", title_style="title", header_style="bold")
        table.add_column("Indicador", justify="left")
        table.add_column("Valor", justify="right")
        table.add_row("Transações", f"{total:,}".replace(",", "."))
        table.add_row("Impacto financeiro", f"R$ {total_diff:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        table.add_row("Taxa de anomalias", f"{anom_rate:.2%}")
        table.add_row("Acurácia de isenções", "N/D" if math.isnan(exempt_acc) else f"{exempt_acc:.2%}")
        console.print(table)

        if merchant_summary_df is not None and not merchant_summary_df.empty:
            t2 = Table(title="Top 5 Estabelecimentos por Anomalias", title_style="title", show_lines=False, header_style="bold")
            t2.add_column("Estabelecimento")
            t2.add_column("Anomalias", justify="right")
            t2.add_column("Transações", justify="right")
            for _, r in merchant_summary_df.nlargest(5, "anomalies").iterrows():
                t2.add_row(str(r["merchant_type"]), f"{int(r['anomalies']):,}".replace(",", "."), f"{int(r['transactions']):,}".replace(",", "."))
            console.print(t2)

        console.print(Panel.fit(
            "[ok]Arquivos gerados:[/ok]\n"
            "• reports/paysmart_executive_report.pdf\n"
            "• reports/executive_report.txt\n"
            "• reports/problematic_transactions.csv\n"
            "• reports/merchant_summary.csv\n"
            "• reports/*.png",
            border_style=PALETTE["accent"]
        ))
        success("RELATÓRIO PDF GERADO COM SUCESSO!" if pdf_ok else "Relatório PDF não pôde ser gerado (verifique dependências).")
    else:
        print("\n=== Resumo Executivo ===")
        print(f"Transações: {total:,}".replace(",", "."))
        print(f"Impacto financeiro: R$ {total_diff:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        print(f"Taxa de anomalias: {anom_rate:.2%}")
        print("Acurácia de isenções:", "N/D" if math.isnan(exempt_acc) else f"{exempt_acc:.2%}")
        print("\nArquivos gerados:")
        print("• reports/paysmart_traditional_report.pdf")
        print("• reports/executive_report.txt")
        print("• reports/problematic_transactions.csv")
        print("• reports/merchant_summary.csv")
        print("• reports/*.png")
        if pdf_ok: print("✅ RELATÓRIO PDF GERADO COM SUCESSO!")
        else: print("⚠ Relatório PDF não pôde ser gerado (verifique dependências).")

# ================== MAIN ==================
def main():
    banner("PAYSMART SOLUTIONS — ANÁLISE DE COBRANÇAS")
    create_directories()

    # Janela temporal
    now_tz = pd.Timestamp.now(tz=TZ)
    today = now_tz.tz_localize(None).normalize()
    first_day_this_month = today.replace(day=1)
    last_day_prev_month = first_day_this_month - pd.Timedelta(days=1)
    first_day_prev_month = last_day_prev_month.replace(day=1)
    first_day_prev_3months = first_day_this_month - pd.DateOffset(months=3)
    last_day_prev_3months = last_day_prev_month

    detector = AnomalyDetector()
    agg = running_agg_init()
    chunks_processed = 0
    win3 = init_window_aggs()
    win1 = init_window_aggs()
    g12 = init_global_aggs()

    step("Processando dados em streaming…")

    for df in iter_dataframes():
        chunks_processed += 1

        needed = {"transaction_id", "date", "merchant_type", "amount",
                  "expected_fee_rate", "charged_fee_rate", "expected_fee", "charged_fee"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Colunas faltantes: {missing}")

        # Agregações globais
        agg["n_rows"] += len(df)
        agg["sum_diff"] += float((df["charged_fee"] - df["expected_fee"]).sum())

        ex_mask = (df["expected_fee"] == 0)
        agg["exempt_total"] += int(ex_mask.sum())
        agg["exempt_correct"] += int((ex_mask & (df["charged_fee"].abs() < 0.01)).sum())

        # Detecção de anomalias
        df_detected = detector.detect_anomalies(df, mode=DETECTION_MODE, chunk_size=ML_CHUNK_SCORING)
        df_classified = detector.classify_billing_errors(df_detected)

        chunk_anom = int(df_classified["is_anomaly"].sum())
        agg["anom_count"] += chunk_anom

        if chunk_anom > 0:
            save_problematic_append(df_classified[df_classified["is_anomaly"]])

        update_merchant_summary(agg, df_classified)

        # Filtros temporais
        df_dates = df_classified.copy()
        if not np.issubdtype(df_dates["date"].dtype, np.datetime64):
            df_dates["date"] = pd.to_datetime(df_dates["date"], errors="coerce")

        mask_1m = (df_dates["date"] >= first_day_prev_month) & (df_dates["date"] <= last_day_prev_month)
        mask_3m = (df_dates["date"] >= first_day_prev_3months) & (df_dates["date"] <= last_day_prev_3months)

        df_3m = df_dates.loc[mask_3m, ["date", "merchant_type", "transaction_id",
                                       "is_anomaly", "expected_fee", "charged_fee"]].copy()
        df_1m = df_dates.loc[mask_1m, ["date", "merchant_type", "transaction_id",
                                       "is_anomaly", "expected_fee", "charged_fee"]].copy()

        if not df_3m.empty:
            update_window_aggs(win3, df_3m)
        if not df_1m.empty:
            update_window_aggs(win1, df_1m)

        # 12m fechados (exclui mês corrente)
        start_12m_closed = first_day_this_month - pd.DateOffset(months=12)
        df_12m_closed = df_dates.loc[
            (df_dates["date"] >= start_12m_closed) & (df_dates["date"] < first_day_this_month),
            ["date", "transaction_id", "is_anomaly", "expected_fee", "charged_fee"]
        ]
        if not df_12m_closed.empty:
            update_global_aggs(g12, df_12m_closed)

        step(f"Chunk {chunks_processed}: {len(df):,} linhas, {chunk_anom:,} anomalias"
             .replace(",", "."))

    step("Consolidando resultados e gerando visualizações…")

    # Consolidação
    merchant_summary_df = finalize_merchant_summary(agg)
    df_trend3, df_merch3 = make_df_from_aggs(win3["trend"], win3["merchant"])  # (mantido se quiser usar depois)
    df_trend1, df_merch1 = make_df_from_aggs(win1["trend"], win1["merchant"])
    df_daily, df_month, hist_x, hist_counts = make_global_dfs(g12)

    # Gráficos
    plot_core_diagnostics(df_daily, df_month, hist_x, hist_counts, merchant_summary_df)
    success("Gráficos principais salvos em reports/")

    step("Gerando relatórios…")

    # Texto (compatibilidade)
    write_executive_report(agg, merchant_summary_df, df_month)

    # PDF completo
    pdf_gen = PDFReportGenerator("reports/paysmart_traditional_report.pdf")
    success_pdf = pdf_gen.generate_complete_report(agg, merchant_summary_df, df_month)

    banner("ANÁLISE CONCLUÍDA", "Arquivos finais disponíveis em reports/")
    print_final_summary(agg, merchant_summary_df, success_pdf)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err(f"Falha na execução: {e}")
        raise
