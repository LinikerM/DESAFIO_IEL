import pandas as pd

# Ler o arquivo e corrigir
with open('models/anomaly_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Substituir o método problemático
old_method = """    def prepare_features(self, df):
        \"\"\"Prepara features para detecção de anomalias\"\"\"
        features = df.copy()
        
        # Diferença entre cobrado e esperado
        features['fee_difference'] = features['charged_fee'] - features['expected_fee']
        features['fee_ratio'] = features['charged_fee'] / (features['expected_fee'] + 1e-6)
        
        # Features temporais
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        
        # Encoding para merchant_type
        merchant_dummies = pd.get_dummies(features['merchant_type'], prefix='merchant')
        features = pd.concat([features, merchant_dummies], axis=1)
        
        # Selecionar features numéricas para o modelo
        numeric_features = ['amount', 'fee_difference', 'fee_ratio', 'day_of_week', 'month']
        merchant_cols = [col for col in features.columns if col.startswith('merchant_')]
        
        return features[numeric_features + merchant_cols]"""

new_method = """    def prepare_features(self, df):
        \"\"\"Prepara features para detecção de anomalias\"\"\"
        features = df.copy()
        
        # Diferença entre cobrado e esperado
        features['fee_difference'] = features['charged_fee'] - features['expected_fee']
        features['fee_ratio'] = features['charged_fee'] / (features['expected_fee'] + 1e-6)
        
        # Features temporais
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        
        # Encoding para merchant_type
        merchant_dummies = pd.get_dummies(features['merchant_type'], prefix='merchant')
        features = pd.concat([features, merchant_dummies], axis=1)
        
        # Selecionar APENAS features numéricas para o modelo
        numeric_features = ['amount', 'fee_difference', 'fee_ratio', 'day_of_week', 'month']
        merchant_cols = [col for col in features.columns if col.startswith('merchant_')]
        
        # Garantir que todas as colunas são numéricas
        selected_features = features[numeric_features + merchant_cols]
        
        # Converter colunas booleanas para int (se houver)
        for col in selected_features.columns:
            if selected_features[col].dtype == 'bool':
                selected_features[col] = selected_features[col].astype(int)
        
        return selected_features"""

content = content.replace(old_method, new_method)

with open('models/anomaly_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Arquivo corrigido!")