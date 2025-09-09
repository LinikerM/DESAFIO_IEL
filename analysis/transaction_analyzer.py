import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TransactionAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def calculate_impact(self):
        """Calcula impacto financeiro das cobranças incorretas"""
        impact = {}
        
        # Diferença total entre cobrado e esperado
        total_difference = (self.df['charged_fee'] - self.df['expected_fee']).sum()
        
        # Separar perdas e ganhos
        overcharge = self.df[self.df['charged_fee'] > self.df['expected_fee']]
        undercharge = self.df[self.df['charged_fee'] < self.df['expected_fee']]
        
        impact['total_difference'] = total_difference
        impact['total_overcharge'] = (overcharge['charged_fee'] - overcharge['expected_fee']).sum()
        impact['total_undercharge'] = (undercharge['expected_fee'] - undercharge['charged_fee']).sum()
        impact['overcharge_count'] = len(overcharge)
        impact['undercharge_count'] = len(undercharge)
        
        return impact
    
    def analyze_exemptions(self):
        """Analisa a aplicação correta de isenções"""
        exempt_df = self.df[self.df['is_exempt'] == True]
        
        # Verificar se isenções foram aplicadas corretamente
        correct_exemptions = exempt_df[exempt_df['charged_fee_rate'] == exempt_df['expected_fee_rate']]
        incorrect_exemptions = exempt_df[exempt_df['charged_fee_rate'] != exempt_df['expected_fee_rate']]
        
        exemption_analysis = {
            'total_exemptions': len(exempt_df),
            'correct_exemptions': len(correct_exemptions),
            'incorrect_exemptions': len(incorrect_exemptions),
            'exemption_accuracy': len(correct_exemptions) / len(exempt_df) if len(exempt_df) > 0 else 0
        }
        
        return exemption_analysis
    
    def generate_summary_stats(self):
        """Gera estatísticas resumidas"""
        stats = {}
        
        # Estatísticas básicas
        stats['total_transactions'] = len(self.df)
        stats['total_amount'] = self.df['amount'].sum()
        stats['total_fees_charged'] = self.df['charged_fee'].sum()
        stats['total_fees_expected'] = self.df['expected_fee'].sum()
        
        # Análise por tipo de estabelecimento
        merchant_stats = self.df.groupby('merchant_type').agg({
            'amount': ['count', 'sum', 'mean'],
            'charged_fee': 'sum',
            'expected_fee': 'sum'
        }).round(2)
        
        stats['merchant_breakdown'] = merchant_stats
        
        # Análise de anomalias se disponível
        if 'is_anomaly' in self.df.columns:
            stats['anomaly_rate'] = self.df['is_anomaly'].mean()
            stats['anomaly_count'] = self.df['is_anomaly'].sum()
        
        return stats
    
    def plot_analysis(self):
        """Gera gráficos de análise"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribuição de diferenças de cobrança
        fee_diff = self.df['charged_fee'] - self.df['expected_fee']
        axes[0, 0].hist(fee_diff, bins=50, alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Diferença Zero')
        axes[0, 0].set_title('Distribuição de Diferenças na Cobrança')
        axes[0, 0].set_xlabel('Diferença (R$)')
        axes[0, 0].legend()
        
        # Cobrança por tipo de estabelecimento
        merchant_fees = self.df.groupby('merchant_type')['charged_fee'].sum()
        axes[0, 1].bar(merchant_fees.index, merchant_fees.values)
        axes[0, 1].set_title('Cobrança Total por Tipo de Estabelecimento')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Scatter: Valor vs Diferença de cobrança
        axes[1, 0].scatter(self.df['amount'], fee_diff, alpha=0.5)
        axes[1, 0].set_title('Valor da Transação vs Diferença na Cobrança')
        axes[1, 0].set_xlabel('Valor da Transação (R$)')
        axes[1, 0].set_ylabel('Diferença na Cobrança (R$)')
        
        # Timeline de anomalias se disponível
        if 'is_anomaly' in self.df.columns:
            daily_anomalies = self.df.groupby(self.df['date'].dt.date)['is_anomaly'].sum()
            axes[1, 1].plot(daily_anomalies.index, daily_anomalies.values)
            axes[1, 1].set_title('Anomalias por Dia')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/analysis_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig