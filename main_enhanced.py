#!/usr/bin/env python3
"""
Sistema Aprimorado de Análise de Cobrança PaySmart
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos do sistema
from data.enhanced_generator import EnhancedDataGenerator, generate_with_discovered_rules
from discovery.rule_discovery import RuleDiscovery
from models.anomaly_detector import AnomalyDetector
from analysis.transaction_analyzer import TransactionAnalyzer
from reports.report_generator import ReportGenerator
from reports.pdf_generator import PDFReportGenerator

# Configurações globais
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def compare_approaches():
    """
    Compara abordagem tradicional vs descoberta de regras
    """
    print("\n" + "="*80)
    print("COMPARAÇÃO: REGRAS FIXAS vs REGRAS DESCOBERTAS")
    print("="*80)
    
    # 1. Abordagem tradicional (regras fixas)
    print("\n1️⃣ Gerando dados com REGRAS FIXAS (abordagem tradicional)...")
    from data.generator import DataGenerator
    
    traditional_gen = DataGenerator(
        n_transactions=100_000,
        months=12,
        anomaly_rate=0.08,
        seed=42
    )
    traditional_df = traditional_gen.generate_transactions()
    traditional_files = traditional_gen.save_data(traditional_df, "transactions_traditional.csv")
    
    # 2. Abordagem com descoberta de regras
    print("\n2️⃣ Gerando dados com REGRAS DESCOBERTAS (abordagem aprimorada)...")
    
    enhanced_df, rule_discovery, enhanced_files = generate_with_discovered_rules(
        n_transactions=100_000,
        months=12,
        anomaly_rate=0.08,
        seed=42,
        save_files=True
    )
    
    # 3. Análise comparativa
    print("\n3️⃣ Executando análise comparativa...")
    
    # Estatísticas básicas
    print(f"\nESTATÍSTICAS BÁSICAS:")
    print(f"{'Métrica':<30} {'Tradicional':<15} {'Descoberta':<15} {'Diferença':<15}")
    print("-" * 75)
    
    metrics = [
        ("Transações", len(traditional_df), len(enhanced_df)),
        ("Taxa média", traditional_df['charged_fee_rate'].mean(), enhanced_df['charged_fee_rate'].mean()),
        ("Desvio padrão taxa", traditional_df['charged_fee_rate'].std(), enhanced_df['charged_fee_rate'].std()),
        ("Receita total", traditional_df['charged_fee'].sum(), enhanced_df['charged_fee'].sum()),
    ]
    
    for metric_name, trad_val, enh_val in metrics:
        if metric_name == "Transações":
            diff = abs(trad_val - enh_val)
            print(f"{metric_name:<30} {trad_val:<15,} {enh_val:<15,} {diff:<15,}")
        else:
            diff = ((enh_val - trad_val) / trad_val * 100) if trad_val != 0 else 0
            print(f"{metric_name:<30} {trad_val:<15.4f} {enh_val:<15.4f} {diff:<15.2f}%")
    
    return traditional_df, enhanced_df, rule_discovery


def run_complete_analysis(df, rule_discovery_obj, approach_name="Enhanced"):
    """
    Executa análise completa em um DataFrame
    """
    print(f"\n📊 Executando análise completa - Abordagem {approach_name}...")
    
    # 1. Detectar anomalias
    detector = AnomalyDetector(contamination=0.08)
    df_with_anomalies = detector.detect_anomalies(df, mode="hybrid", chunk_size=50_000)
    df_classified = detector.classify_billing_errors(df_with_anomalies)
    
    # 2. Análise de transações
    analyzer = TransactionAnalyzer(df_classified)
    impact = analyzer.calculate_impact()
    exemptions = analyzer.analyze_exemptions()
    summary = analyzer.generate_summary_stats()

    import numpy as np

    # --- FIX: garantir colunas de anomalia ---
    if 'is_anomaly' not in df_classified.columns:
        diff = (df_classified["charged_fee_rate"] - df_classified["expected_fee_rate"]).abs()
        base = df_classified["expected_fee_rate"].clip(lower=1e-6)
        df_classified["is_anomaly"] = (diff > 0.5 * base)
    if 'anomalies' not in df_classified.columns:
        df_classified["anomalies"] = df_classified["is_anomaly"].astype(int)

    anomaly_count = int(df_classified["is_anomaly"].sum())
    
    # 3. Gerar relatórios
    reporter = ReportGenerator(df_classified, impact, exemptions, summary)
    
    # Salvar dados problemáticos
    problems, merchant_summary = reporter.save_detailed_results()
    
    # Renomear arquivos para evitar conflito
    suffix = approach_name.lower().replace(" ", "_")
    
    if os.path.exists('reports/problematic_transactions.csv'):
        os.rename('reports/problematic_transactions.csv', 
                 f'reports/problematic_transactions_{suffix}.csv')
    
    if os.path.exists('reports/merchant_summary.csv'):
        os.rename('reports/merchant_summary.csv', 
                 f'reports/merchant_summary_{suffix}.csv')
    
    # 4. Gerar visualizações específicas
    generate_analysis_charts(df_classified, suffix)
    
    # 5. Métricas específicas da descoberta de regras
    rule_metrics = {}
    if rule_discovery_obj and hasattr(rule_discovery_obj, 'discovered_rules'):
        if rule_discovery_obj.discovered_rules:
            rule_metrics = rule_discovery_obj.discovered_rules.get('model_performance', {})
    
    print(f"✅ Análise {approach_name} concluída")
    print(f"   - Anomalias detectadas: {df_classified['is_anomaly'].sum():,}")
    print(f"   - Taxa de anomalias: {df_classified['is_anomaly'].mean():.1%}")
    print(f"   - Impacto financeiro: R$ {impact['total_difference']:,.2f}")
    
    if rule_metrics:
        print(f"   - Qualidade das regras (R²): {rule_metrics.get('r2_score', 0):.1%}")
    
    return {
        'df': df_classified,
        'impact': impact,
        'exemptions': exemptions,
        'summary': summary,
        'problems': problems,
        'merchant_summary': merchant_summary,     # <- já com anomalies e fee_diff_sum
        'rule_metrics': rule_metrics,
        # extras úteis:
        'anomaly_count': int(df_classified['is_anomaly'].sum()),
        'anomaly_rate': float(df_classified['is_anomaly'].mean())
    }



def generate_analysis_charts(df, suffix=""):
    """
    Gera gráficos de análise
    """
    print(f"📈 Gerando gráficos de análise...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Análise de Cobrança PaySmart - {suffix.replace("_", " ").title()}', fontsize=16)
    
    # 1. Distribuição de diferenças
    fee_diff = df['charged_fee'] - df['expected_fee']
    axes[0, 0].hist(fee_diff, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Diferença Zero')
    axes[0, 0].set_title('Distribuição de Diferenças na Cobrança')
    axes[0, 0].set_xlabel('Diferença (R$)')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].legend()
    
    # 2. Taxa por estabelecimento
    merchant_rates = df.groupby('merchant_type')['charged_fee_rate'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(merchant_rates)), merchant_rates.values, color='lightcoral')
    axes[0, 1].set_title('Taxa Média por Estabelecimento')
    axes[0, 1].set_ylabel('Taxa Média')
    axes[0, 1].set_xticks(range(len(merchant_rates)))
    axes[0, 1].set_xticklabels(merchant_rates.index, rotation=45, ha='right')
    
    # 3. Anomalias por dia
    if 'is_anomaly' in df.columns:
        daily_anomalies = df.groupby(df['date'].dt.date)['is_anomaly'].sum()
        axes[0, 2].plot(daily_anomalies.index, daily_anomalies.values, color='orange')
        axes[0, 2].set_title('Anomalias por Dia')
        axes[0, 2].set_ylabel('Número de Anomalias')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Scatter: Valor vs Taxa
    sample_idx = np.random.choice(len(df), size=min(10000, len(df)), replace=False)
    sample_df = df.iloc[sample_idx]
    
    scatter = axes[1, 0].scatter(sample_df['amount'], sample_df['charged_fee_rate'], 
                               c=sample_df['is_anomaly'] if 'is_anomaly' in sample_df.columns else 'blue', 
                               alpha=0.6, cmap='RdYlBu_r')
    axes[1, 0].set_title('Valor vs Taxa de Cobrança')
    axes[1, 0].set_xlabel('Valor da Transação (R$)')
    axes[1, 0].set_ylabel('Taxa de Cobrança')
    axes[1, 0].set_xscale('log')
    
    # 5. Distribuição por canal
    channel_counts = df['channel'].value_counts()
    axes[1, 1].pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Distribuição por Canal')
    
    # 6. Timeline mensal de receita
    monthly_revenue = df.groupby(df['date'].dt.to_period('M'))['charged_fee'].sum()
    axes[1, 2].plot(monthly_revenue.index.astype(str), monthly_revenue.values, marker='o', color='green')
    axes[1, 2].set_title('Receita Mensal')
    axes[1, 2].set_ylabel('Receita (R$)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Salvar gráfico
    chart_filename = f'reports/analysis_comprehensive_{suffix}.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Gráficos salvos: {chart_filename}")


def generate_consolidated_report(traditional_results, enhanced_results):
    """
    Gera relatório consolidado comparando ambas as abordagens
    """
    print("\n📋 Gerando relatório consolidado...")
    
    report_lines = []
    report_lines.append("RELATÓRIO CONSOLIDADO - PAYSMART SOLUTIONS")
    report_lines.append("=" * 60)
    report_lines.append(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
    
    # Comparação de métricas principais
    report_lines.append("COMPARAÇÃO DE ABORDAGENS")
    report_lines.append("-" * 30)
    
    trad_impact = traditional_results['impact']['total_difference']
    enh_impact = enhanced_results['impact']['total_difference']
    trad_anomalies = traditional_results['df']['is_anomaly'].sum()
    enh_anomalies = enhanced_results['df']['is_anomaly'].sum()
    
    report_lines.append(f"IMPACTO FINANCEIRO:")
    report_lines.append(f"  Tradicional: R$ {trad_impact:,.2f}")
    report_lines.append(f"  Descoberta:  R$ {enh_impact:,.2f}")
    report_lines.append(f"  Diferença:   R$ {abs(enh_impact - trad_impact):,.2f}\n")
    
    report_lines.append(f"DETECÇÃO DE ANOMALIAS:")
    report_lines.append(f"  Tradicional: {trad_anomalies:,} anomalias")
    report_lines.append(f"  Descoberta:  {enh_anomalies:,} anomalias")
    report_lines.append(f"  Diferença:   {abs(enh_anomalies - trad_anomalies):,} anomalias\n")
    
    # Qualidade das regras descobertas
    rule_metrics = enhanced_results.get('rule_metrics', {})
    if rule_metrics:
        report_lines.append("QUALIDADE DAS REGRAS DESCOBERTAS")
        report_lines.append("-" * 35)
        report_lines.append(f"R² Score: {rule_metrics.get('r2_score', 0):.1%}")
        report_lines.append(f"Erro Médio Absoluto: {rule_metrics.get('mae', 0):.1%}")
        report_lines.append(f"MAPE: {rule_metrics.get('mean_absolute_percentage_error', 0):.1f}%\n")
    
    # Recomendações
    report_lines.append("RECOMENDAÇÕES FINAIS")
    report_lines.append("-" * 20)
    
    if rule_metrics.get('r2_score', 0) > 0.8:
        report_lines.append("✅ Regras descobertas têm alta qualidade - recomenda-se adoção")
    else:
        report_lines.append("⚠️ Regras descobertas precisam de refinamento")
    
    if abs(enh_impact) < abs(trad_impact):
        report_lines.append("✅ Abordagem descoberta resultou em menor impacto financeiro")
    else:
        report_lines.append("⚠️ Abordagem tradicional teve melhor resultado financeiro")
    
    report_lines.append("📊 Implementar monitoramento contínuo com regras adaptativas")
    report_lines.append("🔄 Reavaliar regras mensalmente com base em novos dados")
    report_lines.append("🎯 Focar correções nos estabelecimentos com mais anomalias")
    
    # Salvar relatório
    consolidated_report = "\n".join(report_lines)
    
    with open("reports/consolidated_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write(consolidated_report)
    
    print(consolidated_report)
    print(f"\n✅ Relatório salvo: reports/consolidated_comparison_report.txt")
    
    return consolidated_report


def main():
    """
    Função principal - executa análise completa com comparação de abordagens
    """
    print("🚀 SISTEMA PAYSMART - ANÁLISE DE COBRANÇA COM DESCOBERTA DE REGRAS")
    print("=" * 80)
    print("Este sistema demonstra duas abordagens para análise de cobrança:")
    print("1. Tradicional: Regras fixas predefinidas")
    print("2. Aprimorada: Descoberta automática de regras dos dados")
    print("=" * 80)
    
    try:
        # Etapa 1: Comparar abordagens de geração
        traditional_df, enhanced_df, rule_discovery = compare_approaches()
        
        # Etapa 2: Análise completa de cada abordagem
        print("\n" + "="*80)
        print("ANÁLISE DETALHADA DE CADA ABORDAGEM")
        print("="*80)
        
        traditional_results = run_complete_analysis(
            traditional_df, None, "Tradicional"
        )
        
        enhanced_results = run_complete_analysis(
            enhanced_df, rule_discovery, "Descoberta"
        )
        
        # Etapa 3: Relatório consolidado
        consolidated_report = generate_consolidated_report(
            traditional_results, enhanced_results
        )
        
        # Etapa 4: Gerar PDF executivo (versão aprimorada)
        print("\n📑 Gerando relatório PDF executivo...")
        try:
            # 3.1 Aggregates: use tipos nativos e chaves que o template costuma acessar
            agg_data = {
                'n_rows': int(len(enhanced_results['df'])),
                'sum_diff': float(enhanced_results['impact']['total_difference']),
                'anom_count': int(enhanced_results['anomaly_count']),
                'anomaly_rate': float(enhanced_results['anomaly_rate']),
                'exempt_total': int(enhanced_results['exemptions']['total_exemptions']),
                'exempt_correct': int(enhanced_results['exemptions']['correct_exemptions']),
                # muitos templates esperam exatamente 'anomalies'
                'anomalies': int(enhanced_results['anomaly_count']),
            }

            # 3.2 Merchant summary: garantir índice e tipos já foram feitos no report_generator
            merchant_summary_df = enhanced_results['merchant_summary'].copy()

            # 3.3 Série mensal: entregue como DataFrame com colunas explícitas
            monthly_counts = (
                enhanced_results['df']
                .groupby(enhanced_results['df']['date'].dt.to_period('M'))
                .size()
                .rename('count')
                .reset_index()
                .rename(columns={'date': 'month'})
            )
            # converter Period -> str para PDF
            monthly_counts['month'] = monthly_counts['month'].astype(str)

            pdf_gen = PDFReportGenerator()
            pdf_success = pdf_gen.generate_complete_report(
                agg_data,
                merchant_summary_df,
                monthly_counts
            )
            if pdf_success:
                print("✅ Relatório PDF gerado com sucesso")
            else:
                print("⚠️ PDFReportGenerator retornou False")
        except Exception as e:
            print(f"⚠️ Erro ao gerar PDF: {e}")
            print("Continuando com relatórios em texto...")
        
        # Resumo final
        print("\n" + "="*80)
        print("RESUMO FINAL")
        print("="*80)
        print("📁 Arquivos gerados:")
        print("   - data/transactions_traditional.csv")
        print("   - data/transactions_discovered_rules.csv")
        print("   - reports/discovered_rules.txt")
        print("   - reports/rules_comparison_report.txt")
        print("   - reports/consolidated_comparison_report.txt")
        print("   - reports/analysis_comprehensive_*.png")
        print("   - reports/paysmart_executive_report.pdf (se disponível)")
        
        print(f"\n🎯 RESULTADO PRINCIPAL:")
        
        print("\n🎯 RESULTADO PRINCIPAL:")

        r2 = 0.0
        dr = getattr(rule_discovery, "discovered_rules", {}) or {}

        # formato A (direto)
        if "model_performance" in dr:
            r2 = dr["model_performance"].get("r2_score", 0.0)
        # formato B (aninhado em 'fee_rules')
        elif "fee_rules" in dr and isinstance(dr["fee_rules"], dict):
            r2 = dr["fee_rules"].get("model_performance", {}).get("r2_score", 0.0)

        print(f"   Regras descobertas explicam {r2:.1%} dos padrões de cobrança")
        
        print(f"   Diferença de impacto financeiro: R$ {abs(enhanced_results['impact']['total_difference'] - traditional_results['impact']['total_difference']):,.2f}")
        
        print("\n✅ Análise completa finalizada com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()