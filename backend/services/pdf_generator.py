"""
Industrial Product Intelligence PDF Datasheet & Catalog Generator.

Generates professional, commerce-ready PDF technical datasheets and catalog reports
for products processed by SpecSense.
"""
import io
import html
from typing import List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from models import StructuredProduct


def safe_text(s: str | None, max_len: int = None) -> str:
    """Safely escapes HTML special characters for ReportLab XML parser."""
    if s is None:
        return ""
    text = str(s).strip()
    if max_len and len(text) > max_len:
        text = text[:max_len] + "..."
    return html.escape(text)


def _get_styles():
    styles = getSampleStyleSheet()
    
    # Custom Brand Colors
    navy = colors.HexColor("#0d233a")
    amber = colors.HexColor("#f5a623")
    steel = colors.HexColor("#4a6b8f")
    dark = colors.HexColor("#1e293b")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=navy,
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=steel,
        spaceAfter=12,
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=navy,
        spaceBefore=10,
        spaceAfter=6,
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=dark,
    )
    
    meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=navy,
    )
    
    meta_val = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=dark,
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white,
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=dark,
    )

    badge_high = ParagraphStyle(
        'BadgeHigh',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#059669"),
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "section": section_heading,
        "body": body_style,
        "meta_label": meta_label,
        "meta_val": meta_val,
        "th": table_header,
        "td": table_cell,
        "badge_high": badge_high,
    }


def _build_product_flowables(p: StructuredProduct, styles: dict) -> list:
    flowables = []
    
    navy = colors.HexColor("#0d233a")
    amber = colors.HexColor("#f5a623")
    light_bg = colors.HexColor("#f1f5f9")
    line_color = colors.HexColor("#cbd5e1")

    # Header Banner
    flowables.append(Paragraph("<b>SpecSense Industrial Intelligence Datasheet</b>", styles["subtitle"]))
    flowables.append(Paragraph(f"{safe_text(p.brand)} — {safe_text(p.part_number)}", styles["title"]))
    flowables.append(Spacer(1, 4))
    flowables.append(HRFlowable(width="100%", thickness=1.5, color=amber, spaceAfter=12))

    # Summary Metadata Box
    cat_val = p.category.value or "Industrial Component"
    mfr_val = p.manufacturer or p.brand
    engine_val = (p.extraction_engine or "AI Cascade").upper()

    summary_data = [
        [Paragraph("<b>Part Number:</b>", styles["meta_label"]), Paragraph(safe_text(p.part_number), styles["meta_val"]),
         Paragraph("<b>Category:</b>", styles["meta_label"]), Paragraph(safe_text(cat_val), styles["meta_val"])],
        [Paragraph("<b>Brand:</b>", styles["meta_label"]), Paragraph(safe_text(p.brand), styles["meta_val"]),
         Paragraph("<b>Manufacturer:</b>", styles["meta_label"]), Paragraph(safe_text(mfr_val), styles["meta_val"])],
        [Paragraph("<b>Intelligence Engine:</b>", styles["meta_label"]), Paragraph(safe_text(engine_val), styles["meta_val"]),
         Paragraph("<b>Category Confidence:</b>", styles["meta_label"]), Paragraph(f"{(p.category.confidence*100):.0f}%", styles["badge_high"])],
    ]

    t_summary = Table(summary_data, colWidths=[1.3*inch, 2.2*inch, 1.4*inch, 2.3*inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 0.5, line_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, line_color),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    flowables.append(t_summary)
    flowables.append(Spacer(1, 12))

    # Commerce Descriptions
    flowables.append(Paragraph("Commerce & Catalog Descriptions", styles["section"]))
    if p.short_desc.value:
        flowables.append(Paragraph(f"<b>Short Description (Mobile/Invoice):</b> {safe_text(p.short_desc.value)}", styles["body"]))
        flowables.append(Spacer(1, 4))
    if p.long_desc.value:
        flowables.append(Paragraph(f"<b>Marketing & Technical Description:</b> {safe_text(p.long_desc.value)}", styles["body"]))
        flowables.append(Spacer(1, 10))

    # Technical Specifications Table
    flowables.append(Paragraph(f"Structured Technical Attributes ({len(p.attributes)})", styles["section"]))

    table_data = [
        [
            Paragraph("<b>Attribute Label</b>", styles["th"]),
            Paragraph("<b>Value</b>", styles["th"]),
            Paragraph("<b>UOM</b>", styles["th"]),
            Paragraph("<b>Confidence</b>", styles["th"]),
            Paragraph("<b>Source / Validation</b>", styles["th"]),
        ]
    ]

    if p.attributes:
        for attr in p.attributes:
            uom_str = safe_text(attr.uom) if attr.uom else "—"
            conf_str = f"{(attr.confidence*100):.0f}%"
            val_status = "✓ Vocab Verified" if attr.vocab_validated else ("Multi-Source Verified" if attr.agreeing_sources > 1 else "Extracted")

            table_data.append([
                Paragraph(safe_text(attr.label), styles["td"]),
                Paragraph(f"<b>{safe_text(attr.value)}</b>", styles["td"]),
                Paragraph(uom_str, styles["td"]),
                Paragraph(conf_str, styles["td"]),
                Paragraph(safe_text(val_status), styles["td"]),
            ])
    else:
        table_data.append([
            Paragraph("Attributes", styles["td"]),
            Paragraph("No additional attributes extracted", styles["td"]),
            Paragraph("—", styles["td"]),
            Paragraph("—", styles["td"]),
            Paragraph("—", styles["td"]),
        ])

    t_attrs = Table(table_data, colWidths=[2.1*inch, 2.1*inch, 0.7*inch, 0.9*inch, 1.4*inch])
    t_attrs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, line_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
    ]))
    flowables.append(t_attrs)
    flowables.append(Spacer(1, 12))

    # Sources & Traceability
    if p.sources_used:
        flowables.append(Paragraph("Ground-Truth Source Citations", styles["section"]))
        for idx, src in enumerate(p.sources_used[:4], 1):
            flowables.append(Paragraph(f"<b>Ref {idx}:</b> {safe_text(src, 80)}", styles["body"]))
            flowables.append(Spacer(1, 2))

    return flowables


def generate_product_pdf(product: StructuredProduct) -> bytes:
    """Generates a downloadable PDF datasheet for a single product."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = _get_styles()
    flowables = _build_product_flowables(product, styles)
    doc.build(flowables)
    return buffer.getvalue()


def generate_catalog_pdf(products: List[StructuredProduct]) -> bytes:
    """Generates a combined multi-product technical catalog PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = _get_styles()
    all_flowables = []
    
    navy = colors.HexColor("#0d233a")
    amber = colors.HexColor("#f5a623")
    light_bg = colors.HexColor("#f1f5f9")
    line_color = colors.HexColor("#cbd5e1")

    # If small catalog (<= 20 products), build full individual datasheets with page breaks
    if len(products) <= 20:
        for i, p in enumerate(products):
            if i > 0:
                all_flowables.append(PageBreak())
            all_flowables.extend(_build_product_flowables(p, styles))
    else:
        # Large Catalog: Executive Cover & Multi-Item Structured Catalog Report
        all_flowables.append(Paragraph("<b>SpecSense Enterprise Technical Catalog</b>", styles["subtitle"]))
        all_flowables.append(Paragraph(f"Industrial Product Catalog Report ({len(products):,} Items)", styles["title"]))
        all_flowables.append(Spacer(1, 4))
        all_flowables.append(HRFlowable(width="100%", thickness=1.5, color=amber, spaceAfter=12))

        # Executive Summary Stats
        high_conf_count = sum(1 for p in products if p.category.confidence >= 0.8)
        avg_attrs = sum(len(p.attributes) for p in products) / max(1, len(products))
        
        stat_data = [
            [
                Paragraph("<b>Total Catalog Products:</b>", styles["meta_label"]), Paragraph(f"{len(products):,}", styles["meta_val"]),
                Paragraph("<b>High-Confidence Verification:</b>", styles["meta_label"]), Paragraph(f"{(high_conf_count/len(products)*100):.1f}%", styles["badge_high"])
            ],
            [
                Paragraph("<b>Avg. Extracted Attributes:</b>", styles["meta_label"]), Paragraph(f"{avg_attrs:.1f} per item", styles["meta_val"]),
                Paragraph("<b>Export Standard:</b>", styles["meta_label"]), Paragraph("252-Column Spec", styles["meta_val"])
            ]
        ]
        t_stats = Table(stat_data, colWidths=[1.8*inch, 1.8*inch, 2.0*inch, 1.6*inch])
        t_stats.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), light_bg),
            ('BOX', (0,0), (-1,-1), 0.5, line_color),
            ('INNERGRID', (0,0), (-1,-1), 0.5, line_color),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        all_flowables.append(t_stats)
        all_flowables.append(Spacer(1, 14))

        # Tabular Catalog
        all_flowables.append(Paragraph(f"Master Catalog Index (First {min(100, len(products))} Products Displayed)", styles["section"]))
        
        table_rows = [
            [
                Paragraph("<b>Part Number</b>", styles["th"]),
                Paragraph("<b>Brand</b>", styles["th"]),
                Paragraph("<b>Category</b>", styles["th"]),
                Paragraph("<b>Key Technical Specs</b>", styles["th"]),
                Paragraph("<b>Conf.</b>", styles["th"]),
            ]
        ]

        # Display up to 100 products in dense PDF catalog format
        for p in products[:100]:
            top_specs = []
            for a in p.attributes[:3]:
                uom = f" {safe_text(a.uom)}" if a.uom else ""
                top_specs.append(f"{safe_text(a.label)}: <b>{safe_text(a.value)}{uom}</b>")
            specs_summary = "; ".join(top_specs) if top_specs else safe_text(p.short_desc.value or "Standard catalog spec", 50)
            cat_text = safe_text(p.category.value or "Industrial", 28)

            table_rows.append([
                Paragraph(f"<b>{safe_text(p.part_number)}</b>", styles["td"]),
                Paragraph(safe_text(p.brand), styles["td"]),
                Paragraph(cat_text, styles["td"]),
                Paragraph(specs_summary, styles["td"]),
                Paragraph(f"{(p.category.confidence*100):.0f}%", styles["td"]),
            ])

        t_catalog = Table(table_rows, colWidths=[1.8*inch, 1.2*inch, 1.5*inch, 2.2*inch, 0.5*inch])
        t_catalog.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), navy),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
            ('GRID', (0,0), (-1,-1), 0.5, line_color),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ]))
        all_flowables.append(t_catalog)

        if len(products) > 100:
            all_flowables.append(Spacer(1, 10))
            all_flowables.append(Paragraph(f"<i>Note: Complete 252-column dataset of all {len(products):,} products is exported via CSV.</i>", styles["subtitle"]))

    doc.build(all_flowables)
    return buffer.getvalue()
