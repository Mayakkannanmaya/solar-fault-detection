"""
AI Solar Fault Assistant Module
Generates expert diagnostic explanations, impact assessments, and maintenance protocols
based on detected photovoltaic and thermographic anomalies.
"""

from typing import Dict, Any, Optional

FAULT_KNOWLEDGE_BASE = {
    "Hot Spot": {
        "causes": [
            "Localized cell internal short-circuit or microcrack causing reverse bias heating.",
            "Sub-cell solder joint fatigue or interconnect ribbon failure.",
            "Permanent optical obstruction (hard shadowing or baked-on bird dropping).",
            "Bypass diode degradation allowing high reverse current dissipation."
        ],
        "impact": "Severe localized thermal runaway. Can cause EVA lamination browning, backsheet melting, glass shattering, and up to 35-70% localized string power loss. High risk of electrical fire under peak irradiance.",
        "action": "Immediate thermal audit. Disconnect string from combiner box if Delta-T > 25°C. Check bypass diodes with forward-bias multimeter. Replace module if cell shunting is irreparable.",
        "urgency": "Immediate (Within 24 Hours)",
        "iec_standard": "IEC 62446-3 Thermographic PV Inspection / IEC 61215"
    },
    "Micro Crack": {
        "causes": [
            "Mechanical stress from heavy wind loads, hailstorms, or improper foot traffic during installation.",
            "Thermal cycling and thermo-mechanical expansion mismatch between silicon wafer and copper ribbons.",
            "Micro-fractures formed during wafer slicing or manual transport."
        ],
        "impact": "Disconnected cell sub-regions leading to power generation loss (10-30%). High probability of progressing into catastrophic hot spots over consecutive thermal cycles.",
        "action": "Perform Electroluminescence (EL) imaging to check crack branch continuity. Monitor localized temperature for hot spot initiation. Seal with anti-degradation laminate or plan module replacement in next scheduled overhaul.",
        "urgency": "High (Within 1 Week)",
        "iec_standard": "IEC TS 60904-13 Electroluminescence of PV Modules"
    },
    "Dust & Soiling": {
        "causes": [
            "Atmospheric particulate deposition, desert dust, pollen, agricultural dust, or vehicular exhaust accumulation.",
            "Inadequate cleaning schedule or insufficient rainfall cleaning cycle."
        ],
        "impact": "Direct optical attenuation of solar irradiance. Can decrease overall string yield by 5% to 28%. Non-uniform soiling triggers reverse bias heating.",
        "action": "Deploy automated or semi-automated cleaning robots with demineralized water and soft rotating brushes. Avoid abrasive chemicals and daytime thermal shock washing.",
        "urgency": "Medium (Within Scheduled Cycle)",
        "iec_standard": "IEC 61724-1 Photovoltaic System Performance Monitoring"
    },
    "Bird Dropping": {
        "causes": [
            "Avian activity on array perimeters or absence of bird deterrent spikes/wires on PV mounting racks."
        ],
        "impact": "Complete localized optical blockage of single cell. Forces bypass diodes to activate or causes reverse-bias heating hotspot (> 15°C above adjacent cells).",
        "action": "Targeted spot cleaning using biodegradable non-abrasive detergent and microfiber wipe. Install bird deterrent spikes or ultrasonic repellers on array mounting perimeters.",
        "urgency": "Medium (Within 48-72 Hours)",
        "iec_standard": "IEC 62446 Maintenance Guidelines"
    },
    "Physical Damage": {
        "causes": [
            "Severe hail strike, projectile impact, vandalism, or structural collapse of mounting tracker.",
            "Glass shatter due to excessive mechanical torsional load or thermal stress."
        ],
        "impact": "Destruction of module hermetic seal, moisture ingress, potential-induced degradation (PID), corrosion of internal metallization, and severe shock/ground-fault hazard.",
        "action": "De-energize module string immediately! Lockout/tagout (LOTO) procedures. Safely remove shattered panel wearing personal protective equipment (PPE). Replace with matching IV-curve rated module.",
        "urgency": "Critical / Immediate (Safety Hazard)",
        "iec_standard": "IEC 61730 PV Module Safety Qualification"
    },
    "Discoloration": {
        "causes": [
            "Ethylene Vinyl Acetate (EVA) encapsulant browning caused by UV exposure, high humidity, and poor additive formulation.",
            "Moisture ingress reacting with silver grid fingers causing snail trails."
        ],
        "impact": "Gradual reduction in light transmission (5-15% output loss). Snail trails indicate micro-cracks allowing moisture contact with silver paste.",
        "action": "Measure IV-curve fill factor to evaluate resistance changes. Conduct EL inspection to map microcrack root causes. Log degradation rate into predictive maintenance registry.",
        "urgency": "Low to Medium (Scheduled Monitoring)",
        "iec_standard": "IEC 61215 Degradation Benchmarking"
    },
    "Cell Defect": {
        "causes": [
            "Silicon ingot crystal impurities, wafer dislocation clusters, or metallization peel-off.",
            "Potential Induced Degradation (PID) caused by high system voltage and sodium ion migration."
        ],
        "impact": "Dead cell sub-strings, permanent 33% power reduction in affected sub-module, high shunt resistance loss.",
        "action": "Perform anti-PID voltage reversal during nighttime if PID suspected. If crystallographic dead cell, replace bypass diode or bypass panel.",
        "urgency": "High (Within 1-2 Weeks)",
        "iec_standard": "IEC 62804 Potential-Induced Degradation Testing"
    },
    "Partial Shading": {
        "causes": [
            "Vegetation overgrowth (trees, weeds), nearby structures, overhead transmission lines, or inter-row tracker shading."
        ],
        "impact": "Module output drop proportional to shaded sub-strings. Bypass diodes conduct continuously, increasing operating temperature and reducing inverter MPPT efficiency.",
        "action": "Trim surrounding vegetation. Re-calibrate tracker azimuth and tilt angles. Consider Module-Level Power Electronics (MLPE) such as DC optimizers to mitigate mismatch.",
        "urgency": "Medium (Within 3-5 Days)",
        "iec_standard": "IEC 62446-1 System Documentation & Inspection"
    },
    "Electrical Anomaly": {
        "causes": [
            "Failed or shorted bypass diode in junction box.",
            "Oxidized or loose MC4 connectors, corroded bus ribbons, or ground leakage current."
        ],
        "impact": "String mismatch, total module bypass or open-circuit voltage collapse. Severe arc-fault and fire hazard at junction box.",
        "action": "Perform open-circuit voltage (Voc) and short-circuit current (Isc) string test. Open junction box (when de-energized) to check diode integrity with DMM. Inspect MC4 connectors with thermal camera.",
        "urgency": "Immediate / Critical (Fire & Arc Hazard)",
        "iec_standard": "IEC 62446-1 Category 1 & 2 Testing"
    },
    "Normal": {
        "causes": ["PV module operating within nominal nominal operating cell temperature (NOCT) parameters."],
        "impact": "None. Optimal photon-to-electron energy conversion efficiency.",
        "action": "No corrective maintenance required. Maintain regular quarterly thermographic and visual flyover inspection schedule.",
        "urgency": "None (Routine Surveillance)",
        "iec_standard": "IEC 62446 Compliant"
    }
}

class AISolarFaultAssistant:
    """Intelligent diagnostic and recommendation engine for solar panel faults"""

    @staticmethod
    def analyze_fault(
        fault_type: str,
        confidence: float,
        panel_code: str,
        cell_row: Optional[int] = None,
        cell_col: Optional[int] = None,
        delta_t: float = 0.0,
        max_temp: float = 40.0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive technical diagnosis, cause explanation,
        risk rating, and actionable SOP recommendations.
        """
        kb = FAULT_KNOWLEDGE_BASE.get(fault_type, FAULT_KNOWLEDGE_BASE["Normal"])
        
        # Refine severity and urgency based on thermal metrics if available
        is_immediate = False
        calculated_severity = "Normal"
        
        if fault_type == "Normal":
            calculated_severity = "Normal"
            is_immediate = False
        elif fault_type in ["Hot Spot", "Physical Damage", "Electrical Anomaly"] or delta_t > 20.0 or max_temp > 70.0:
            calculated_severity = "Critical"
            is_immediate = True
        elif fault_type in ["Micro Crack", "Cell Defect"] or delta_t > 12.0:
            calculated_severity = "High"
            is_immediate = False if delta_t <= 15.0 else True
        elif fault_type in ["Bird Dropping", "Partial Shading", "Discoloration"]:
            calculated_severity = "Medium"
        else:
            calculated_severity = "Low"

        # Build location descriptor
        if cell_row is not None and cell_col is not None:
            loc_desc = f"Panel {panel_code} — Cell [Row {cell_row}, Column {cell_col}]"
        else:
            loc_desc = f"Panel {panel_code} — Module-wide / Multiple cells"

        # Formulate root causes
        causes = kb["causes"]
        cause_summary = causes[0] if causes else "Unknown origin"
        
        # Create technical explanation
        explanation = (
            f"Model detected **{fault_type}** with **{confidence*100:.1f}% confidence** at {loc_desc}. "
            f"Thermographic analysis indicates a peak temperature of **{max_temp:.1f}°C** "
            f"with a thermal gradient (ΔT) of **{delta_t:.1f}°C** above ambient reference."
        )

        return {
            "fault_type": fault_type,
            "confidence": round(confidence, 3),
            "severity": calculated_severity,
            "location_description": loc_desc,
            "cell_row": cell_row,
            "cell_col": cell_col,
            "max_temp_c": round(max_temp, 1),
            "delta_t_c": round(delta_t, 1),
            "possible_causes": causes,
            "primary_cause": cause_summary,
            "performance_impact": kb["impact"],
            "recommended_action": kb["action"],
            "urgency": kb["urgency"],
            "immediate_inspection_required": is_immediate,
            "iec_standard": kb["iec_standard"],
            "full_explanation": explanation
        }

    @staticmethod
    def generate_alert_payload(analysis_result: Dict[str, Any], panel_code: str) -> Dict[str, Any]:
        """Format real-time alert payload for notification dispatchers (Email, SMS, WhatsApp, IoT)"""
        urgency_emoji = "🚨" if analysis_result["severity"] in ["Critical", "High"] else "⚠️"
        
        message = (
            f"{urgency_emoji} SOLAR FAULT ALERT — PV ARRAY MONITORING SYSTEM\n"
            f"Panel ID: {panel_code}\n"
            f"Location: {analysis_result['location_description']}\n"
            f"Fault Detected: {analysis_result['fault_type']}\n"
            f"Severity: {analysis_result['severity'].upper()}\n"
            f"Confidence: {analysis_result['confidence']*100:.1f}%\n"
            f"Thermal ΔT: +{analysis_result['delta_t_c']}°C (Max: {analysis_result['max_temp_c']}°C)\n"
            f"Primary Cause: {analysis_result['primary_cause']}\n"
            f"Action Required: {analysis_result['recommended_action']}\n"
            f"Inspection Urgency: {analysis_result['urgency']}"
        )

        return {
            "panel_code": panel_code,
            "fault_type": analysis_result["fault_type"],
            "severity": analysis_result["severity"],
            "confidence": analysis_result["confidence"],
            "location": analysis_result["location_description"],
            "message": message,
            "should_notify": analysis_result["severity"] in ["Critical", "High"]
        }
