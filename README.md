# Desafio IEL — PaySmart Solutions

Este repositório contém a solução desenvolvida para o **Desafio IEL (Inova Talentos - Instituto Itaú de Ciência, Tecnologia e Inovação)**.  
O projeto simula e analisa transações financeiras de uma adquirente fictícia (**PaySmart Solutions**) com foco em **detecção de anomalias, validação de isenções e descoberta de regras ocultas**.

---

## 📌 Objetivo do Desafio
A PaySmart precisa garantir que as **taxas de transação**, seu principal produto, sejam cobradas corretamente, respeitando as isenções aplicáveis.  
A empresa enfrenta o desafio da **falta de documentação completa** sobre essas regras, o que pode gerar cobranças incorretas (a mais ou a menos).  

**Principais metas da solução:**
- Identificar **anomalias e outliers** em cobranças.
- Verificar se **isenções foram aplicadas corretamente**.
- Implementar um pipeline de **monitoramento contínuo**.
- Comparar duas abordagens:
  1. **Tradicional (regras fixas pré-definidas)**.
  2. **Descoberta automática de regras (Machine Learning)**.

---

## 🛠️ Tecnologias Utilizadas
- **Python 3.10+**
- **Pandas / NumPy** → manipulação de dados.
- **Matplotlib / Seaborn** → visualizações e gráficos.
- **Scikit-learn** → modelo Isolation Forest para detecção de anomalias.
- **ReportLab** → geração de relatórios PDF executivos.
- **Rich / TQDM** → interface interativa em terminal e barra de progresso.

---

## 📂 Estrutura do Projeto
paysmart_analysis/

│
├── main.py # Pipeline tradicional (1M transações, regras fixas)

├── main_enhanced.py # Pipeline aprimorado (comparação com regras descobertas

│

├── data/

│ ├── generator.py # Geração de base sintética com regras fixas

│ ├── enhanced_generator.py # Geração de base sintética com variabilidade

│

├── discovery/

│ └── rule_discovery.py # Descoberta automática de regras (árvore de decisão)

│

├── models/

│ └── anomaly_detector.py # Detecção de anomalias (estatístico, ML, híbrido)

│

├── analysis/

│ └── transaction_analyzer.py # Análise de impacto, isenções e estatísticas

│
├── reports/

│ ├── pdf_generator.py # Relatórios PDF executivos

│ ├── report_generator.py # Relatórios texto/CSV

│ └── outputs gerados (PDF, TXT, CSV, PNG)

│

└── README.md # Este arquivo


---

## ⚙️ Como Executar

### 1. Instalar dependências
Recomenda-se usar ambiente virtual (`venv` ou `conda`):

⚙️ Executar versão tradicional
python main.py

Processa 1 milhão de transações.
Aplica regras fixas pré-definidas.
Gera relatórios executivos e gráficos em reports/.

⚙️ Executar versão ML
python main_enhanced.py

Processa 100 mil transações (limitado por performance).
Descobre regras automaticamente e compara com as regras fixas.
Gera relatórios comparativos (TXT, PDF, CSVs).


📊 Relatórios Gerados

reports/paysmart_traditional_report.pdf → Relatório tradicional (controle fixo).
reports/paysmart_executive_report.pdf → Relatório executivo completo (UI++).
reports/consolidated_comparison_report.txt → Comparativo entre abordagens.
reports/discovered_rules.txt → Regras descobertas automaticamente.
reports/rules_comparison_report.txt → Comparação entre regras fixas e descobertas.
reports/problematic_transactions.csv → Lista de transações com problemas.
reports/merchant_summary.csv → Resumo por tipo de estabelecimento.
reports/*.png → Gráficos de distribuição, timeline de anomalias, top estabelecimentos.

reports/*.png → Gráficos de distribuição, timeline de anomalias, top estabelecimentos.

🔎 Principais Resultados
Abordagem Tradicional (1M transações)
Impacto financeiro: R$ 26.426,40.
Taxa de anomalias: 24,35% (243.455 casos).
Acurácia de isenções: 90,09%.
Abordagem Descoberta de Regras (100k transações)
Taxa típica em supermercados: 3,1% (faixa 2,9–3,3%).
Fatores explicativos: tipo de estabelecimento (86%), dia da semana (9,6%).
Qualidade do modelo: R² = 62,3%, MAE = 0,3%.
Necessita refinamento antes de adoção plena.

🚀 Conclusão e Recomendações

O sistema detecta falhas de cobrança e aplicação de isenções de forma automatizada.
Tradicional → garante consistência e serve como baseline confiável.
Descoberta → mostra potencial de adaptação e aprendizado de padrões ocultos.
Recomendação: adotar modelo híbrido (controle + aprendizado), com monitoramento recorrente e revisão mensal das regras.
Valor para o negócio: redução de riscos financeiros, conformidade regulatória e maior transparência com clientes.
