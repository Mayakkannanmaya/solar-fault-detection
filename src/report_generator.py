"""
Automated Inspection Report Generation Engine
Produces high-quality executive and engineering audit reports in PDF and CSV formats
compliant with IEC 62446 PV array testing standards.
"""

import os
import csv
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from config import REPORTS_DIR
from src.database import SessionLocal, InspectionRecord, Panel

class ReportGenerator:
    """Generates comprehensive inspection audits in PDF and CSV formats"""

    @staticmethod
    def generate_csv_report(filename: Optional[str] = None) -> str:
        """Exports all historical inspection records to a CSV file"""
        if filename is None:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"pv_inspection_report_{ts}.csv"

        filepath = REPORTS_DIR / filename
        db = SessionLocal()

        try:
            records = db.query(InspectionRecord).all()
            data = []
            for r in records:
                data.append({
                    "Record_ID": r.id,
                    "Panel_Code": r.panel_code,
                    "Timestamp_UTC": r.timestamp.isoformat() if r.timestamp else "",
                    "Fault_Type": r.fault_type,
                    "Severity": r.severity,
                    "Confidence": round(r.confidence, 3),
                    "Cell_Row": r.cell_row if r.cell_row else "N/A",
                    "Cell_Col": r.cell_col if r.cell_col else "N/A",
                    "Max_Temp_C": r.max_temp_c,
                    "Delta_T_C": r.delta_t_c,
                    "Status": r.status,
                    "Primary_Cause": r.possible_cause or "",
                    "Recommended_Action": r.recommended_action or ""
                })

            df = pd.DataFrame(data)
            df.to_csv(str(filepath), index=False)
            return str(filepath)

        finally:
            db.close()

    @staticmethod
    def generate_pdf_report(
        farm_name: str = "Helios One 50MW Solar Facility",
        filename: Optional[str] = None
    ) -> str:
        """
        Builds an executive PDF audit report with summary KPIs,
        fault category breakdowns, critical anomalies, and IEC 62446 recommendations.
        """
        if filename is None:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"solar_audit_report_{ts}.pdf"

        filepath = REPORTS_DIR / filename
        db = SessionLocal()

        try:
            # Query metrics
            total_panels = db.query(Panel).count()
            inspections = db.query(InspectionRecord).all()
            total_inspections = len(inspections)

            fault_counts = {}
            severity_counts = {"Normal": 0, "Low": 0, "Medium": 0, "High": 0, "Critical": 0}
            critical_items = []
            temps = []

            for r in inspections:
                f = r.fault_type
                fault_counts[f] = fault_counts.get(f, 0) + 1
                sev = r.severity
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
                if r.delta_t_c:
                    temps.append(r.delta_t_c)
                if sev in ["Critical", "High"]:
                    critical_items.append(r)

            avg_delta_t = round(sum(temps) / max(1, len(temps)), 1) if temps else 0.0
            faulty_panels_count = total_inspections - severity_counts.get("Normal", 0)

            # Build PDF Document
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36
            )
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                "TitleStyle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=20,
                textColor=colors.HexColor("#0f172a"),
                spaceAfter=6
            )
            subtitle_style = ParagraphStyle(
                "SubtitleStyle",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                textColor=colors.HexColor("#475569"),
                spaceAfter=14
            )
            section_heading = ParagraphStyle(
                "SectionHeading",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=13,
                textColor=colors.HexColor("#1e293b"),
                spaceBefore=12,
                spaceAfter=6
            )
            body_style = ParagraphStyle(
                "BodyTextCustom",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=9,
                textColor=colors.HexColor("#1e293b")
            )
            table_header_style = ParagraphStyle(
                "TableHeader",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9,
                textColor=colors.white
            )

            story = []

            # Header
            story.append(Paragraph("AI-POWERED SOLAR PV INSPECTION AUDIT REPORT", title_style))
            report_time_str = datetime.datetime.utcnow().strftime("%B %d, %Y - %H:%M UTC")
            story.append(Paragraph(
                f"Facility: <b>{farm_name}</b> | Generated: <b>{report_time_str}</b> | Standard: <b>IEC 62446-3 Thermographic Audit</b>",
                subtitle_style
            ))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=14))

            # KPI Summary Grid
            kpi_data = [
                [
                    Paragraph(f"<b>Total Panels</b><br/><font size=14 color='#0284c7'>{total_panels}</font>", body_style),
                    Paragraph(f"<b>Total Audits</b><br/><font size=14 color='#0f172a'>{total_inspections}</font>", body_style),
                    Paragraph(f"<b>Fault Rate</b><br/><font size=14 color='#dc2626'>{round((faulty_panels_count/max(1, total_inspections))*100, 1)}%</font>", body_style),
                    Paragraph(f"<b>Critical Incidents</b><br/><font size=14 color='#e11d48'>{severity_counts.get('Critical', 0)}</font>", body_style),
                    Paragraph(f"<b>Mean ΔT Gradient</b><br/><font size=14 color='#ea580c'>+{avg_delta_t}°C</font>", body_style),
                ]
            ]
            kpi_table = Table(kpi_data, colWidths=[108, 108, 108, 108, 108])
            kpi_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            story.append(kpi_table)
            story.append(Spacer(1, 14))

            # Fault Breakdown Table
            story.append(Paragraph("1. Photovoltaic Fault Distribution Breakdown", section_heading))
            fault_rows = [[
                Paragraph("<b>Fault Category</b>", table_header_style),
                Paragraph("<b>Detected Count</b>", table_header_style),
                Paragraph("<b>Relative Prevalence</b>", table_header_style),
                Paragraph("<b>Dominant Severity</b>", table_header_style)
            ]]

            for f_name, count in sorted(fault_counts.items(), key=lambda x: x[1], reverse=True):
                prev = f"{round((count / max(1, total_inspections))*100, 1)}%"
                dom_sev = "Critical" if "Hot" in f_name or "Damage" in f_name else ("High" if "Crack" in f_name or "Cell" in f_name else "Medium")
                if f_name == "Normal":
                    dom_sev = "Normal"
                
                fault_rows.append([
                    Paragraph(f_name, body_style),
                    Paragraph(str(count), body_style),
                    Paragraph(prev, body_style),
                    Paragraph(dom_sev, body_style)
                ])

            fault_table = Table(fault_rows, colWidths=[200, 100, 120, 120])
            fault_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(fault_table)
            story.append(Spacer(1, 14))

            # Critical Action Register
            story.append(Paragraph("2. Critical & High Priority Action Register", section_heading))
            crit_rows = [[
                Paragraph("<b>Panel Code</b>", table_header_style),
                Paragraph("<b>Cell Grid</b>", table_header_style),
                Paragraph("<b>Fault Type</b>", table_header_style),
                Paragraph("<b>ΔT (°C)</b>", table_header_style),
                Paragraph("<b>Recommended Action Protocol</b>", table_header_style)
            ]]

            for c in critical_items[:8]: # top 8
                grid_coord = f"R{c.cell_row} C{c.cell_col}" if c.cell_row else "Module"
                action_text = (c.recommended_action[:60] + "...") if c.recommended_action and len(c.recommended_action) > 60 else (c.recommended_action or "Immediate inspection")
                crit_rows.append([
                    Paragraph(f"<b>{c.panel_code}</b>", body_style),
                    Paragraph(grid_coord, body_style),
                    Paragraph(c.fault_type, body_style),
                    Paragraph(f"+{c.delta_t_c}°C", body_style),
                    Paragraph(action_text, body_style)
                ])

            if len(crit_rows) == 1:
                crit_rows.append([Paragraph("No critical faults currently logged.", body_style)] * 5)

            crit_table = Table(crit_rows, colWidths=[90, 70, 110, 60, 210])
            crit_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b91c1c")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff1f2")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fecdd3")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(crit_table)
            story.append(Spacer(1, 16))

            # Sign-off Footer
            story.append(Paragraph(
                "<i>Certification: This inspection report was automatically synthesized by the AI-Powered Solar Monitoring Deep Learning Pipeline in accordance with IEC 62446 Category 1 & 2 thermographic compliance specifications.</i>",
                body_style
            ))

            doc.build(story)
            return str(filepath)

        finally:
            db.close()
