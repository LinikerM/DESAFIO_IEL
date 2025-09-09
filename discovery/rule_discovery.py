import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class RuleDiscovery:
    """
    Descoberta automática de regras de negócio a partir dos dados
    usando técnicas de machine learning interpretáveis
    """
    
    def __init__(self, min_samples_leaf: int = 50, max_depth: int = 8, min_rule_support: float = 0.05):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_rule_support = min_rule_support
        self.discovered_rules = {}
        self.encoders = {}
        self.rule_tree = None
        self.feature_importance = {}
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para descoberta de regras"""
        features = df.copy()
        
        # Features temporais
        if 'date' in features.columns:
            features['date'] = pd.to_datetime(features['date'])
            features['day_of_week'] = features['date'].dt.dayofweek
            features['month'] = features['date'].dt.month
            features['quarter'] = features['date'].dt.quarter
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            features['is_month_end'] = (features['date'].dt.day > 25).astype(int)
        
        # Features de valor
        if 'amount' in features.columns:
            features['amount_log'] = np.log1p(features['amount'])
            features['is_high_value'] = (features['amount'] > features['amount'].quantile(0.9)).astype(int)
            features['is_low_value'] = (features['amount'] < features['amount'].quantile(0.1)).astype(int)
            features['amount_decile'] = pd.qcut(features['amount'], 10, labels=False, duplicates='drop')
        
        # Encoding de variáveis categóricas
        categorical_cols = ['merchant_type', 'channel']
        for col in categorical_cols:
            if col in features.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.encoders[col].fit_transform(features[col].astype(str))
                else:
                    # Para dados novos, usar encoder já treinado
                    try:
                        features[f'{col}_encoded'] = self.encoders[col].transform(features[col].astype(str))
                    except ValueError:
                        # Lidar com valores não vistos
                        features[f'{col}_encoded'] = -1
        
        return features
    
    def discover_fee_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Descobre regras para cálculo de taxas usando árvores de decisão
        """
        print("🔍 Descobrindo regras de taxas a partir dos dados...")
        
        # Preparar dados
        features = self._prepare_features(df)
        
        # Calcular taxa efetiva observada
        features['observed_fee_rate'] = features['charged_fee'] / (features['amount'] + 1e-6)
        
        # Features para o modelo
        feature_cols = [
            'merchant_type_encoded', 'channel_encoded',
            'day_of_week', 'month', 'quarter', 'is_weekend', 'is_month_end',
            'amount_log', 'is_high_value', 'is_low_value', 'amount_decile'
        ]
        
        # Filtrar apenas features que existem
        available_features = [col for col in feature_cols if col in features.columns]
        
        X = features[available_features].fillna(-1)
        y = features['observed_fee_rate']
        
        # Treinar árvore de decisão interpretável
        self.rule_tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        
        self.rule_tree.fit(X, y)
        
        # Feature importance
        self.feature_importance = dict(zip(available_features, self.rule_tree.feature_importances_))
        
        # Extrair regras em texto
        tree_rules = export_text(self.rule_tree, feature_names=available_features)
        
        # Descobrir padrões específicos
        discovered_patterns = self._extract_patterns(features, available_features)
        
        results = {
            'tree_rules_text': tree_rules,
            'feature_importance': self.feature_importance,
            'discovered_patterns': discovered_patterns,
            'model_performance': self._evaluate_rule_performance(X, y),
            'available_features': available_features
        }
        
        self.discovered_rules['fee_rules'] = results
        return results
    
    def _extract_patterns(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """
        Extrai padrões específicos dos dados
        """
        patterns = {}
        
        # 1. Padrões por tipo de estabelecimento
        if 'merchant_type' in df.columns:
            merchant_patterns = {}
            for merchant in df['merchant_type'].unique():
                merchant_data = df[df['merchant_type'] == merchant]
                if len(merchant_data) >= self.min_samples_leaf:
                    median_rate = merchant_data['observed_fee_rate'].median()
                    std_rate = merchant_data['observed_fee_rate'].std()
                    
                    merchant_patterns[merchant] = {
                        'median_fee_rate': median_rate,
                        'std_fee_rate': std_rate,
                        'transaction_count': len(merchant_data),
                        'typical_range': (
                            merchant_data['observed_fee_rate'].quantile(0.25),
                            merchant_data['observed_fee_rate'].quantile(0.75)
                        )
                    }
            patterns['by_merchant'] = merchant_patterns
        
        # 2. Padrões temporais
        if 'day_of_week' in df.columns:
            temporal_patterns = {}
            for dow in range(7):
                dow_data = df[df['day_of_week'] == dow]
                if len(dow_data) >= self.min_samples_leaf:
                    temporal_patterns[f'day_{dow}'] = {
                        'median_fee_rate': dow_data['observed_fee_rate'].median(),
                        'transaction_count': len(dow_data)
                    }
            patterns['temporal'] = temporal_patterns
        
        # 3. Padrões por valor
        if 'amount' in df.columns:
            value_patterns = {}
            # Quartis de valor
            quartiles = df['amount'].quantile([0.25, 0.5, 0.75, 0.95])
            
            for i, (q_name, q_value) in enumerate(zip(['low', 'medium', 'high', 'premium'], quartiles.values)):
                if i == 0:
                    mask = df['amount'] <= q_value
                elif i == len(quartiles) - 1:
                    mask = df['amount'] > quartiles.iloc[i-1]
                else:
                    mask = (df['amount'] > quartiles.iloc[i-1]) & (df['amount'] <= q_value)
                
                segment_data = df[mask]
                if len(segment_data) >= self.min_samples_leaf:
                    value_patterns[q_name] = {
                        'median_fee_rate': segment_data['observed_fee_rate'].median(),
                        'amount_range': (segment_data['amount'].min(), segment_data['amount'].max()),
                        'transaction_count': len(segment_data)
                    }
            
            patterns['by_value'] = value_patterns
        
        # 4. Detectar isenções automáticas
        zero_fee_data = df[df['observed_fee_rate'] <= 0.001]  # Praticamente zero
        if len(zero_fee_data) > 0:
            exemption_patterns = {}
            
            # Por estabelecimento
            exemption_by_merchant = zero_fee_data['merchant_type'].value_counts()
            exemption_patterns['by_merchant'] = exemption_by_merchant.to_dict()
            
            # Por dia da semana
            if 'day_of_week' in zero_fee_data.columns:
                exemption_by_dow = zero_fee_data['day_of_week'].value_counts()
                exemption_patterns['by_day_of_week'] = exemption_by_dow.to_dict()
            
            # Por valor
            if 'amount' in zero_fee_data.columns:
                exemption_patterns['value_stats'] = {
                    'min_amount': zero_fee_data['amount'].min(),
                    'max_amount': zero_fee_data['amount'].max(),
                    'median_amount': zero_fee_data['amount'].median()
                }
            
            patterns['exemptions'] = exemption_patterns
        
        return patterns
    
    def _evaluate_rule_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Avalia performance do modelo de regras"""
        y_pred = self.rule_tree.predict(X)
        
        return {
            'mae': mean_absolute_error(y, y_pred),
            'r2_score': r2_score(y, y_pred),
            'mean_absolute_percentage_error': np.mean(np.abs((y - y_pred) / (y + 1e-6))) * 100
        }
    
    def generate_business_rules(self) -> List[str]:
        """
        Converte padrões descobertos em regras de negócio legíveis
        """
        if 'fee_rules' not in self.discovered_rules:
            return ["Nenhuma regra foi descoberta ainda. Execute discover_fee_rules() primeiro."]
        
        rules = []
        patterns = self.discovered_rules['fee_rules']['discovered_patterns']
        
        # Regras por estabelecimento
        if 'by_merchant' in patterns:
            rules.append("=== REGRAS POR TIPO DE ESTABELECIMENTO ===")
            for merchant, data in patterns['by_merchant'].items():
                median_rate = data['median_fee_rate']
                range_low, range_high = data['typical_range']
                rules.append(f"• {merchant}: taxa típica {median_rate:.1%} (faixa normal: {range_low:.1%} - {range_high:.1%})")
        
        # Regras temporais
        if 'temporal' in patterns:
            rules.append("\n=== REGRAS TEMPORAIS ===")
            day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            for day_key, data in patterns['temporal'].items():
                day_num = int(day_key.split('_')[1])
                day_name = day_names[day_num]
                rules.append(f"• {day_name}: taxa típica {data['median_fee_rate']:.1%}")
        
        # Regras por valor
        if 'by_value' in patterns:
            rules.append("\n=== REGRAS POR VALOR DE TRANSAÇÃO ===")
            for segment, data in patterns['by_value'].items():
                amount_min, amount_max = data['amount_range']
                rules.append(f"• {segment.title()}: R$ {amount_min:,.0f} - R$ {amount_max:,.0f} → taxa {data['median_fee_rate']:.1%}")
        
        # Regras de isenção
        if 'exemptions' in patterns:
            rules.append("\n=== REGRAS DE ISENÇÃO DESCOBERTAS ===")
            exemptions = patterns['exemptions']
            
            if 'by_merchant' in exemptions:
                rules.append("• Isenções por estabelecimento:")
                for merchant, count in exemptions['by_merchant'].items():
                    rules.append(f"  - {merchant}: {count} transações isentas")
            
            if 'by_day_of_week' in exemptions:
                rules.append("• Isenções por dia da semana:")
                day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                for day_num, count in exemptions['by_day_of_week'].items():
                    day_name = day_names[day_num]
                    rules.append(f"  - {day_name}: {count} transações isentas")
        
        # Feature importance
        rules.append("\n=== FATORES MAIS IMPORTANTES ===")
        sorted_importance = sorted(self.discovered_rules['fee_rules']['feature_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_importance[:5]:
            rules.append(f"• {feature}: {importance:.1%} de importância")
        
        return rules
    
    def predict_expected_fee_rate(self, df: pd.DataFrame) -> pd.Series:
        """
        Usa as regras descobertas para predizer a taxa esperada
        """
        if self.rule_tree is None:
            raise ValueError("Modelo não foi treinado. Execute discover_fee_rules() primeiro.")
        
        features = self._prepare_features(df)
        available_features = self.discovered_rules['fee_rules']['available_features']
        
        X = features[available_features].fillna(-1)
        predicted_rates = self.rule_tree.predict(X)
        
        return pd.Series(predicted_rates, index=df.index)
    
    def save_discovered_rules(self, filename: str = 'reports/discovered_rules.txt'):
        """Salva regras descobertas em arquivo"""
        rules = self.generate_business_rules()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("REGRAS DE NEGÓCIO DESCOBERTAS AUTOMATICAMENTE\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Data de geração: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n\n")
            
            for rule in rules:
                f.write(rule + "\n")
            
            # Adicionar performance do modelo
            if 'fee_rules' in self.discovered_rules:
                perf = self.discovered_rules['fee_rules']['model_performance']
                f.write(f"\n=== QUALIDADE DAS REGRAS DESCOBERTAS ===\n")
                f.write(f"• R² Score: {perf['r2_score']:.1%} (quão bem as regras explicam os dados)\n")
                f.write(f"• Erro Médio Absoluto: {perf['mae']:.1%}\n")
                f.write(f"• MAPE: {perf['mean_absolute_percentage_error']:.1f}%\n")
        
        print(f"✓ Regras salvas em: {filename}")
    
    def validate_discovered_rules(self, validation_df: pd.DataFrame) -> Dict[str, float]:
        """
        Valida as regras descobertas em um conjunto de dados diferente
        """
        if self.rule_tree is None:
            raise ValueError("Modelo não foi treinado. Execute discover_fee_rules() primeiro.")
        
        # Preparar dados de validação
        features = self._prepare_features(validation_df)
        available_features = self.discovered_rules['fee_rules']['available_features']
        
        X_val = features[available_features].fillna(-1)
        y_val = features['charged_fee'] / (features['amount'] + 1e-6)
        
        # Predizer com regras descobertas
        y_pred = self.rule_tree.predict(X_val)
        
        # Métricas de validação
        validation_metrics = {
            'validation_mae': mean_absolute_error(y_val, y_pred),
            'validation_r2': r2_score(y_val, y_pred),
            'validation_mape': np.mean(np.abs((y_val - y_pred) / (y_val + 1e-6))) * 100
        }
        
        return validation_metrics