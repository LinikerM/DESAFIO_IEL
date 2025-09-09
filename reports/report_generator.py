import pandas as pd
from datetime import datetime

class ReportGenerator:
    def __init__(self, df, impact_analysis, exemption_analysis, summary_stats):
        self.df = df
        self.impact_analysis = impact_analysis
        self.exemption_analysis = exemption_analysis
        self.summary_stats = summary_stats
        
    def generate_executive_report(self):
        """Gera relatório executivo"""
        report = []
        
        report.append("=" * 80)
        report.append("RELATÓRIO DE ANÁLISE DE COBRANÇAS - PAYSMART SOLUTIONS")
        report.append("=" * 80)
        report.append(f"Data de Geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        report.append("")
        
        # Resumo Executivo
        report.append("RESUMO EXECUTIVO")
        report.append("-" * 20)
        report.append(f"Total de Transações Analisadas: {self.summary_stats['total_transactions']:,}")
        report.append(f"Volume Total Processado: R$ {self.summary_stats['total_amount']:,.2f}")
        report.append(f"Taxas Cobradas: R$ {self.summary_stats['total_fees_charged']:,.2f}")
        report.append(f"Taxas Esperadas: R$ {self.summary_stats['total_fees_expected']:,.2f}")
        report.append("")
        
        # Impacto Financeiro
        report.append("IMPACTO FINANCEIRO DAS COBRANÇAS INCORRETAS")
        report.append("-" * 45)
        report.append(f"Diferença Total: R$ {self.impact_analysis['total_difference']:,.2f}")
        
        if self.impact_analysis['total_difference'] > 0:
            report.append("  ↗ Resultado: COBRANÇA EXCESSIVA")
        elif self.impact_analysis['total_difference'] < 0:
            report.append("  ↘ Resultado: COBRANÇA INSUFICIENTE")
        else:
            report.append("  → Resultado: EQUILIBRADO")
            
        report.append(f"Cobrança Excessiva: R$ {self.impact_analysis['total_overcharge']:,.2f} ({self.impact_analysis['overcharge_count']} casos)")
        report.append(f"Cobrança Insuficiente: R$ {self.impact_analysis['total_undercharge']:,.2f} ({self.impact_analysis['undercharge_count']} casos)")
        report.append("")
        
        # Análise de Isenções
        report.append("ANÁLISE DE ISENÇÕES")
        report.append("-" * 20)
        report.append(f"Total de Isenções Concedidas: {self.exemption_analysis['total_exemptions']}")
        report.append(f"Isenções Aplicadas Corretamente: {self.exemption_analysis['correct_exemptions']}")
        report.append(f"Isenções com Problemas: {self.exemption_analysis['incorrect_exemptions']}")
        report.append(f"Taxa de Acerto: {self.exemption_analysis['exemption_accuracy']:.1%}")
        report.append("")
        
        # Anomalias
        if 'anomaly_rate' in self.summary_stats:
            report.append("DETECÇÃO DE ANOMALIAS")
            report.append("-" * 22)
            report.append(f"Taxa de Anomalias: {self.summary_stats['anomaly_rate']:.1%}")
            report.append(f"Casos Anômalos Detectados: {self.summary_stats['anomaly_count']}")
            report.append("")
        
        # Recomendações
        report.append("RECOMENDAÇÕES")
        report.append("-" * 14)
        
        if self.exemption_analysis['exemption_accuracy'] < 0.95:
            report.append("• Revisar processo de aplicação de isenções")
            
        if abs(self.impact_analysis['total_difference']) > 1000:
            report.append("• Implementar controles mais rigorosos de validação")
            
        if self.summary_stats.get('anomaly_rate', 0) > 0.05:
            report.append("• Investigar causas das anomalias detectadas")
            
        report.append("• Implementar monitoramento automatizado")
        report.append("• Criar dashboards em tempo real")
        report.append("• Treinar equipe sobre regras de isenção")
        
        return "\n".join(report)
    
    def save_detailed_results(self):
        """Salva resultados detalhados em CSV e retorna (problems_df, merchant_summary_df)."""
        # Transações com problemas
        fee_problems = abs(self.df['charged_fee'] - self.df['expected_fee']) > 0.01
        anomaly_problems = self.df.get('is_anomaly', pd.Series([False] * len(self.df), index=self.df.index))
        problems = self.df[fee_problems | anomaly_problems].copy()

        if len(problems) > 0:
            problems.to_csv('reports/problematic_transactions.csv', index=False)

        # Resumo por estabelecimento
        agg = self.df.groupby('merchant_type', dropna=False).agg({
            'amount': ['count', 'sum', 'mean'],
            'charged_fee': 'sum',
            'expected_fee': 'sum',
            'is_anomaly': 'sum' if 'is_anomaly' in self.df.columns else (lambda s: 0)
        }).round(2)

        # Normaliza nomes
        agg.columns = [
            'transactions', 'amount_sum', 'amount_mean',
            'charged_fee_sum', 'expected_fee_sum', 'anomalies'
        ] if 'is_anomaly' in self.df.columns else [
            'transactions', 'amount_sum', 'amount_mean',
            'charged_fee_sum', 'expected_fee_sum'
        ]
        merchant_summary = agg.reset_index()

        # Garante colunas obrigatórias que o PDF pode acessar
        if 'anomalies' not in merchant_summary.columns:
            merchant_summary['anomalies'] = 0
        merchant_summary['fee_diff_sum'] = (
            merchant_summary['charged_fee_sum'] - merchant_summary['expected_fee_sum']
        )

        # Converte para tipos Python nativos (evita np.int64/np.float64 em serializers)
        for col in ['transactions', 'anomalies']:
            merchant_summary[col] = merchant_summary[col].astype(int)
        for col in ['amount_sum', 'amount_mean', 'charged_fee_sum', 'expected_fee_sum', 'fee_diff_sum']:
            merchant_summary[col] = merchant_summary[col].astype(float)

        merchant_summary.to_csv('reports/merchant_summary.csv', index=False)
        return problems, merchant_summary

    
    def print_report(self):
        """Imprime o relatório no console"""
        report = self.generate_executive_report()
        print(report)
        
        # Salvar relatório em arquivo
        with open('reports/executive_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)