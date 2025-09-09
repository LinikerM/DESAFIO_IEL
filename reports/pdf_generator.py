# -*- coding: utf-8 -*-
import os
import math
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader

from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, Flowable
)

from reportlab.pdfgen import canvas as pdfcanvas


class NumberedCanvas(pdfcanvas.Canvas):
    """
    Canvas que escreve 'Página X de Y' no rodapé em dois passes.
    """
    def __init__(self, *args, **kwargs):
        self._saved_page_states = []
        # Permite configurar margens para posicionar paginação
        self._page_conf = kwargs.pop("page_conf", {
            "pagesize": A4,
            "margin_right": 20 * mm,
            "margin_bottom": 18 * mm,
            "muted_color": colors.HexColor("#6B7A90"),
            "font_name": "Helvetica",
            "font_size": 8.5,
        })
        super().__init__(*args, **kwargs)

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(total_pages)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _draw_page_number(self, page_count):
        w, h = self._page_conf["pagesize"]
        margin_right = self._page_conf["margin_right"]
        margin_bottom = self._page_conf["margin_bottom"]
        self.setFont(self._page_conf["font_name"], self._page_conf["font_size"])
        self.setFillColor(self._page_conf["muted_color"])
        text = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(w - margin_right, margin_bottom - 16, text)


# =========================
# PDFReportGenerator (UI++)
# =========================
class PDFReportGenerator:
    """
    Gerador de PDF com:
    - capa com chips
    - header/footer (linhas e data)
    - paginação 'Página X de Y' (NumberedCanvas)
    - paleta coesa, KPIs, tabelas em zebra
    - imagens auto-escaladas e legendadas
    """

    PALETTE = {
        "primary": colors.HexColor("#184C7A"),   # navy
        "accent":  colors.HexColor("#05A6AB"),   # teal
        "warn":    colors.HexColor("#FFB300"),   # amarelo
        "ink":     colors.HexColor("#0F172A"),   # quase-preto
        "muted":   colors.HexColor("#6B7A90"),   # cinza-azulado
        "bg":      colors.white,
        "line":    colors.HexColor("#E5E7EB"),
        "rose":    colors.HexColor("#D14D57"),
        "green":   colors.HexColor("#2E8B57"),
    }

    def __init__(self, filename="reports/paysmart_executive_report.pdf", logo_path="reports/logo.png"):
        self.filename = filename
        self.logo_path = logo_path if (logo_path and os.path.exists(logo_path)) else None

        # Margens e frames
        self.pagesize = A4
        self.margin_left = 20 * mm
        self.margin_right = 20 * mm
        self.margin_top = 18 * mm
        self.margin_bottom = 18 * mm

        # Documento
        self.doc = BaseDocTemplate(
            filename,
            pagesize=self.pagesize,
            leftMargin=self.margin_left,
            rightMargin=self.margin_right,
            topMargin=self.margin_top,
            bottomMargin=self.margin_bottom,
            title="PaySmart - Relatório Executivo"
        )

        frame = Frame(
            self.margin_left,
            self.margin_bottom,
            self.pagesize[0] - self.margin_left - self.margin_right,
            self.pagesize[1] - self.margin_top - self.margin_bottom,
            id='normal',
            showBoundary=0,
        )

        self.story = []

        # Estilos base + custom
        base = getSampleStyleSheet()
        self.styles = base

        self.styles.add(ParagraphStyle(
            name='TitleBrand',
            parent=base['Heading1'],
            fontSize=22,
            leading=26,
            alignment=TA_LEFT,
            textColor=self.PALETTE["primary"],
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=base['Heading2'],
            fontSize=14,
            leading=18,
            textColor=self.PALETTE["muted"],
            spaceAfter=16,
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=base['Heading2'],
            fontSize=16,
            leading=19,
            textColor=self.PALETTE["primary"],
            spaceBefore=10,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=base['Heading3'],
            fontSize=12,
            leading=14,
            textColor=self.PALETTE["accent"],
            spaceBefore=6,
            spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name='Body',
            parent=base['BodyText'],
            fontSize=10.5,
            leading=14.5,
            textColor=self.PALETTE["ink"],
            alignment=TA_JUSTIFY
        ))
        # Nome único para não colidir com 'Bullet' do sample stylesheet
        self.styles.add(ParagraphStyle(
            name='BulletPlus',
            parent=base['BodyText'],
            fontSize=10.5,
            leading=14,
            leftIndent=8,
            bulletIndent=0,
            textColor=self.PALETTE["ink"],
        ))
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=base['BodyText'],
            fontSize=9.2,
            leading=12,
            textColor=self.PALETTE["muted"],
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ))
        self.styles.add(ParagraphStyle(
            name='KPI',
            parent=base['BodyText'],
            fontSize=11.5,
            leading=14,
            textColor=colors.white,
            alignment=TA_CENTER,
        ))

        # Template com header/footer
        self.doc.addPageTemplates([
            PageTemplate(
                id='withHeader',
                frames=[frame],
                onPage=self._draw_header_footer
            )
        ])

    # ---------- Header/Footer ----------
    def _draw_header_footer(self, canv, doc):
        w, h = self.pagesize
        # Linhas
        canv.saveState()
        canv.setStrokeColor(self.PALETTE["line"])
        canv.setLineWidth(0.5)
        # topo
        canv.line(self.margin_left, h - self.margin_top + 8, w - self.margin_right, h - self.margin_top + 8)
        # header
        canv.setFillColor(self.PALETTE["primary"])
        canv.setFont("Helvetica-Bold", 9.5)
        canv.drawString(self.margin_left, h - self.margin_top + 14, "PaySmart — Análise de Cobranças")
        canv.setFillColor(self.PALETTE["muted"])
        canv.setFont("Helvetica", 8.8)
        canv.drawRightString(
            w - self.margin_right,
            h - self.margin_top + 14,
            datetime.now().strftime("Gerado em %d/%m/%Y %H:%M")
        )
        # rodapé (linha — a paginação real é desenhada pelo NumberedCanvas no save)
        canv.setStrokeColor(self.PALETTE["line"])
        canv.setLineWidth(0.5)
        canv.line(self.margin_left, self.margin_bottom - 8, w - self.margin_right, self.margin_bottom - 8)
        canv.restoreState()

    # ---------- Helpers visuais ----------
    def _hr(self, height=0.8, color=None):
        class HR(Flowable):
            def __init__(self, width, height, color):
                Flowable.__init__(self)
                self.width = width
                self.height = height
                self.color = color

            def draw(self):
                self.canv.setFillColor(self.color)
                self.canv.rect(0, 0, self.width, self.height, stroke=0, fill=1)

            def wrap(self, availWidth, availHeight):
                self.width = availWidth
                return availWidth, self.height

        return HR(100, height, color or self.PALETTE["line"])

    def _chip(self, text, bg_color, txt_color=colors.white, pad_h=6, pad_v=2):
        tbl = Table([[Paragraph(text, ParagraphStyle(
            name="chip",
            fontSize=8.8, leading=10,
            textColor=txt_color,
            alignment=TA_CENTER
        ))]])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_color),
            ("TEXTCOLOR", (0, 0), (-1, -1), txt_color),
            ("LEFTPADDING", (0, 0), (-1, -1), pad_h),
            ("RIGHTPADDING", (0, 0), (-1, -1), pad_h),
            ("TOPPADDING", (0, 0), (-1, -1), pad_v),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad_v),
            ("BOX", (0, 0), (-1, -1), 0.5, bg_color),
        ]))
        return tbl

    def _kpi_card(self, title, value, bg_color):
        title_p = Paragraph(f"<b>{title}</b>", ParagraphStyle(
            name="kpi_title", fontSize=9.5, leading=12, textColor=colors.white, alignment=TA_CENTER
        ))
        value_p = Paragraph(f"<b>{value}</b>", ParagraphStyle(
            name="kpi_value", fontSize=16, leading=18, textColor=colors.white, alignment=TA_CENTER
        ))
        box = Table([[title_p], [value_p]], colWidths="*")
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_color),
            ("BOX", (0, 0), (-1, -1), 0.6, bg_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.0, bg_color),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return box

    # ---------- Formatação BR ----------
    @staticmethod
    def _fmt_int(n):
        return f"{int(n):,}".replace(",", ".")

    @staticmethod
    def _fmt_money(v):
        s = f"{v:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"

    @staticmethod
    def _fmt_pct(p):
        try:
            return f"{p:.2%}"
        except Exception:
            return "N/D"

    # ---------- Capa ----------
    def add_title_page(self):
        self.story.append(Spacer(1, 12))
        self.story.append(self._hr(2.5, self.PALETTE["primary"]))

        self.story.append(Spacer(1, 26))
        self.story.append(Paragraph("PAYSMART SOLUTIONS", self.styles['TitleBrand']))
        self.story.append(Paragraph("Relatório Executivo de Análise de Cobranças", self.styles['Subtitle']))

        chips = Table([[
            self._chip("12 meses analisados", self.PALETTE["accent"]),
            self._chip("Detecção híbrida (ML + regras)", self.PALETTE["primary"]),
            self._chip(datetime.now().strftime("Gerado em %d/%m/%Y %H:%M"), self.PALETTE["warn"], txt_color=self.PALETTE["ink"]),
        ]])
        chips.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        self.story.append(chips)
        self.story.append(Spacer(1, 18))
        self.story.append(self._hr())

        if self.logo_path:
            try:
                logo = Image(self.logo_path, width=120, height=120, kind='proportional')
                logo.hAlign = 'RIGHT'
                self.story.append(Spacer(1, 6))
                self.story.append(logo)
            except Exception:
                pass

        self.story.append(PageBreak())

    # ---------- Seções ----------
    def add_section(self, title, content_list=None, table_data=None, image_path=None, caption=None):
        self.story.append(Paragraph(title, self.styles['SectionHeader']))
        self.story.append(self._hr())

        if content_list:
            self.story.append(Spacer(1, 6))
            for content in content_list:
                if content.strip().startswith("•"):
                    self.story.append(Paragraph(content, self.styles['BulletPlus']))
                elif content.strip().startswith("- "):
                    self.story.append(Paragraph(content[2:], self.styles['BulletPlus']))
                else:
                    self.story.append(Paragraph(content, self.styles['Body']))
                self.story.append(Spacer(1, 4))

        if table_data:
            self.story.append(Spacer(1, 6))
            self.story.append(self._styled_table(table_data))

        if image_path:
            self._image_block(image_path, caption=caption)

        self.story.append(Spacer(1, 12))

    def _styled_table(self, data):
        tbl = Table(data, hAlign='LEFT', colWidths=[None] * len(data[0]))
        nrows = len(data)
        style = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), self.PALETTE["primary"]),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.25, self.PALETTE["line"]),
            ("LINEABOVE", (0, 1), (-1, 1), 0.4, self.PALETTE["primary"]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, self.PALETTE["primary"]),
        ]
        for r in range(1, nrows):
            bg = colors.whitesmoke if r % 2 == 1 else colors.HexColor("#FAFAFA")
            style.append(("BACKGROUND", (0, r), (-1, r), bg))
        tbl.setStyle(TableStyle(style))
        return tbl

    def _image_block(self, path, caption=None, max_w=None, max_h=None):
        """
        Insere imagem redimensionada para caber no frame atual.
        """
        if not os.path.exists(path):
            self.story.append(Paragraph(f"<i>Imagem não encontrada: {path}</i>", self.styles['Body']))
            return

        try:
            frame_w = self.pagesize[0] - self.margin_left - self.margin_right
            frame_h = self.pagesize[1] - self.margin_top - self.margin_bottom
            max_w = max_w or frame_w
            max_h = max_h or (frame_h - 80)  # folga p/ títulos/legendas

            ir = ImageReader(path)
            orig_w, orig_h = ir.getSize()
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            tgt_w = orig_w * scale
            tgt_h = orig_h * scale

            img = Image(path, width=tgt_w, height=tgt_h)
            img.hAlign = 'CENTER'

            elems = [Spacer(1, 6), img]
            cap = caption or os.path.basename(path)
            elems.append(Paragraph(cap, self.styles['Caption']))

            self.story.append(KeepTogether(elems))

        except Exception as e:
            self.story.append(Paragraph(f"<i>Não foi possível carregar a imagem ({e}).</i>", self.styles['Body']))

    # ---------- Conteúdo específico ----------
    def add_metrics_summary(self, agg, merchant_summary_df, df_month):
        total = agg["n_rows"]
        total_diff = agg["sum_diff"]
        anom = agg["anom_count"]
        exempt_acc = (agg["exempt_correct"] / agg["exempt_total"]) if agg["exempt_total"] > 0 else float("nan")
        anom_rate = (anom / total) if total else 0.0

        k1 = self._kpi_card("Transações", self._fmt_int(total), self.PALETTE["primary"])
        k2 = self._kpi_card("Impacto Financeiro", self._fmt_money(total_diff), self.PALETTE["accent"])
        k3 = self._kpi_card("Taxa de Anomalias", self._fmt_pct(anom_rate), self.PALETTE["rose"])
        k4 = self._kpi_card("Acurácia de Isenções", "N/D" if math.isnan(exempt_acc) else self._fmt_pct(exempt_acc), self.PALETTE["green"])

        grid = Table([[k1, k2, k3, k4]], colWidths=[(self.pagesize[0] - self.margin_left - self.margin_right) / 4.0] * 4, hAlign='LEFT')
        grid.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("INNERGRID", (0, 0), (-1, -1), 0, colors.white),
        ]))

        self.add_section(
            title="RESUMO EXECUTIVO — MÉTRICAS-CHAVE",
            content_list=[
                "• Análise da integridade das cobranças e aplicação de isenções no horizonte de 12 meses.",
                "• Detecção de anomalias por abordagem híbrida (regras + ML) com priorização por impacto.",
                "• Resultados fornecem base para auditoria contínua, priorização de correções e governança de receita."
            ]
        )
        self.story.append(grid)
        self.story.append(Spacer(1, 10))

        summary_data = [
            ['Métrica', 'Valor'],
            ['Total de Transações', self._fmt_int(total)],
            ['Impacto Financeiro Total', self._fmt_money(total_diff)],
            ['Taxa de Anomalias', self._fmt_pct(anom_rate)],
            ['Anomalias Detectadas', self._fmt_int(anom)],
            ['Taxa de Acerto em Isenções', "N/D" if math.isnan(exempt_acc) else self._fmt_pct(exempt_acc)]
        ]
        self.story.append(self._styled_table(summary_data))
        self.story.append(Spacer(1, 8))

    def add_top_merchants_table(self, merchant_summary_df):
        if merchant_summary_df is None or merchant_summary_df.empty:
            return
        top_10 = merchant_summary_df.nlargest(10, "anomalies")
        table_data = [['Estabelecimento', 'Transações', 'Anomalias', 'Impacto (R$)']]
        for _, row in top_10.iterrows():
            table_data.append([
                str(row['merchant_type']),
                self._fmt_int(row['transactions']),
                self._fmt_int(row['anomalies']),
                self._fmt_money(row['fee_diff_sum'])
            ])
        self.add_section(
            title="TOP 10 ESTABELECIMENTOS POR ANOMALIAS",
            content_list=[
                "• Ranking dos tipos de estabelecimento com maior incidência de anomalias detectadas.",
                "• Útil para focar ações corretivas e treinamentos onde o risco é mais elevado."
            ],
            table_data=table_data
        )

    def add_analysis_with_chart(self, title, description, chart_path):
        self.add_section(
            title=title,
            content_list=description
        )
        self._image_block(chart_path)

    def add_methodology(self):
        content = [
            "BASE DE DADOS",
            "• Transações sintéticas dos últimos 12 meses espelhando padrões reais.",
            "• Processamento em streaming (chunks) para eficiência de memória.",
            "• Regras de negócio para cálculo esperado de taxas e isenções.",
            "",
            "DETECÇÃO DE ANOMALIAS",
            "• Abordagem híbrida: regras estatísticas + Isolation Forest.",
            "• Feature engineering temporal e por estabelecimento.",
            "• Classificação de erros de cobrança para priorização.",
            "",
            "ANÁLISE TEMPORAL",
            "• Janelas 1M e 3M para tendências e sazonalidades.",
            "• Agregações diárias e mensais para padrões operacionais.",
            "• Métricas de acurácia de isenções e impacto financeiro."
        ]
        self.add_section("METODOLOGIA APLICADA", content_list=content)

    def add_recommendations(self, agg, exemption_analysis=None):
        rec = ["Baseado nos achados, recomendamos:"]
        if agg["anom_count"] > agg["n_rows"] * 0.05:
            rec.append("• [ALTA PRIORIDADE] Investigar causas-raiz das anomalias (fluxos, integrações, cadastros).")
            rec.append("• Habilitar alertas automáticos e workflow de tratativa para transações suspeitas.")
        if abs(agg["sum_diff"]) > 10_000:
            rec.append("• Reforçar revisões de validação de cobrança e dupla checagem para altos valores.")
        rec += [
            "• Monitoramento contínuo (dashboards executivos + auditoria mensal).",
            "• Atualização de POPs e treinamento das equipes sobre regras de isenção.",
            "• Trilha de auditoria consolidada para conformidade (SOX/COSO/COBIT)."
        ]
        self.add_section("RECOMENDAÇÕES E PRÓXIMOS PASSOS", content_list=rec)

    def add_compliance_section(self):
        content = [
            "ALINHAMENTO A FRAMEWORKS",
            "",
            "COSO",
            "• Monitoramento automatizado e documentação de controles.",
            "• Segregação de funções e evidências de efetividade.",
            "",
            "SOX",
            "• Controles automáticos sobre fluxos de receita.",
            "• Testes contínuos, logs e rastreabilidade por transação.",
            "",
            "COBIT",
            "• Governança de dados, papéis/responsabilidades e SLAs.",
            "• Métricas de performance e melhoria contínua."
        ]
        self.add_section("COMPLIANCE E GOVERNANÇA", content_list=content)

    # ---------- Build ----------
    def generate_complete_report(self, agg, merchant_summary_df, df_month):
        print("Gerando relatório PDF completo (versão UI++)...")

        # CAPA
        self.add_title_page()

        # RESUMO / MÉTRICAS
        self.add_metrics_summary(agg, merchant_summary_df, df_month)

        # METODOLOGIA
        self.add_methodology()

        # ANÁLISES COM GRÁFICOS
        analyses = [
            {
                'title': '1) DISTRIBUIÇÃO DE DIFERENÇAS DE COBRANÇA',
                'description': [
                    "• Histograma das diferenças entre cobrado e esperado.",
                    "• Zero indica cobrança correta; positivo: sobrecobrança; negativo: cobrança abaixo.",
                    "• Permite identificar viés sistemático e caudas de risco."
                ],
                'chart': 'reports/fee_diff_hist.png'
            },
            {
                'title': '2) ACURÁCIA NA APLICAÇÃO DE ISENÇÕES (MENSAL)',
                'description': [
                    "• Percentual de isenções aplicadas corretamente mês a mês.",
                    "• Quedas sugerem gargalos operacionais ou regras desatualizadas."
                ],
                'chart': 'reports/exemption_accuracy_monthly.png'
            },
            {
                'title': '3) TIMELINE DE ANOMALIAS (12 MESES)',
                'description': [
                    "• Evolução diária das anomalias detectadas.",
                    "• Picos e padrões sazonais ajudam a priorizar investigações."
                ],
                'chart': 'reports/anomalies_timeline_12m.png'
            },
            {
                'title': '4) TOP ESTABELECIMENTOS POR ANOMALIA',
                'description': [
                    "• Ranking por volume de anomalias.",
                    "• Base para plano de ação por tipo de negócio."
                ],
                'chart': 'reports/top_merchants_anomalies.png'
            }
        ]
        for a in analyses:
            self.add_analysis_with_chart(a['title'], a['description'], a['chart'])

        # TABELA TOP MERCHANTS
        self.add_top_merchants_table(merchant_summary_df)

        # COMPLIANCE
        self.add_compliance_section()

        # ARQUIVOS GERADOS
        files_content = [
            "Arquivos gerados:",
            "• paysmart_executive_report.pdf — relatório completo",
            "• executive_report.txt — versão em texto",
            "• problematic_transactions.csv — transações com anomalias",
            "• merchant_summary.csv — resumo por estabelecimento",
            "• *.png — gráficos individuais (histograma, timeline, etc.)",
        ]
        self.add_section("SAÍDAS DO PROCESSO", content_list=files_content)

        # Build com NumberedCanvas (para 'Página X de Y')
        try:
            page_conf = {
                "pagesize": self.pagesize,
                "margin_right": self.margin_right,
                "margin_bottom": self.margin_bottom,
                "muted_color": self.PALETTE["muted"],
                "font_name": "Helvetica",
                "font_size": 8.5,
            }
            self.doc.build(
                self.story,
                canvasmaker=lambda *args, **kw: NumberedCanvas(*args, page_conf=page_conf, **kw)
            )
            print(f"✓ Relatório PDF gerado: {self.filename}")
            return True
        except Exception as e:
            print(f"✗ Erro ao gerar PDF: {e}")
            return False
