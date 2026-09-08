"""
Industrial Product Intelligence PDF Datasheet & Catalog Generator.

Generates professional, commerce-ready PDF technical datasheets and catalog reports
for products processed by SpecSense.
"""
import io
from typing import List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from models import StructuredProduct


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
    flowables.append(Paragraph(f"<b>SpecSense Industrial Intelligence Datasheet</b>", styles["subtitle"]))
    flowables.append(Paragraph(f"{p.brand} — {p.part_number}", styles["title"]))
    flowables.append(Spacer(1, 4))
    flowables.append(HRFlowable(width="100%", thickness=1.5, color=amber, spaceAfter=12))

    # Summary Metadata Box
    cat_val = p.category.value or "Industrial Component"
    mfr_val = p.manufacturer or p.brand
    engine_val = (p.extraction_engine or "AI Cascade").upper()

    summary_data = [
        [Paragraph("<b>Part Number:</b>", styles["meta_label"]), Paragraph(p.part_number, styles["meta_val"]),
         Paragraph("<b>Category:</b>", styles["meta_label"]), Paragraph(cat_val, styles["meta_val"])],
        [Paragraph("<b>Brand:</b>", styles["meta_label"]), Paragraph(p.brand, styles["meta_val"]),
         Paragraph("<b>Manufacturer:</b>", styles["meta_label"]), Paragraph(mfr_val, styles["meta_val"])],
        [Paragraph("<b>Intelligence Engine:</b>", styles["meta_label"]), Paragraph(engine_val, styles["meta_val"]),
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
        flowables.append(Paragraph(f"<b>Short Description (Mobile/Invoice):</b> {p.short_desc.value}", styles["body"]))
        flowables.append(Spacer(1, 4))
    if p.long_desc.value:
        flowables.append(Paragraph(f"<b>Marketing & Technical Description:</b> {p.long_desc.value}", styles["body"]))
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
            uom_str = attr.uom if attr.uom else "—"
            conf_str = f"{(attr.confidence*100):.0f}%"
            val_status = "✓ Vocab Verified" if attr.vocab_validated else ("Multi-Source Verified" if attr.agreeing_sources > 1 else "Extracted")

            table_data.append([
                Paragraph(attr.label, styles["td"]),
                Paragraph(f"<b>{attr.value}</b>", styles["td"]),
                Paragraph(uom_str, styles["td"]),
                Paragraph(conf_str, styles["td"]),
                Paragraph(val_status, styles["td"]),
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
            flowables.append(Paragraph(f"<b>Ref {idx}:</b> {src}", styles["body"]))
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

    for i, p in enumerate(products):
        if i > 0:
            all_flowables.append(PageBreak())
        all_flowables.extend(_build_product_flowables(p, styles))

    doc.build(all_flowables)
    return buffer.getvalue()
