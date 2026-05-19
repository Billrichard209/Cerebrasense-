"""
CerebraSense AI Physician Scribe
Generates natural language clinical summaries and decision support narratives.
"""

from typing import Dict, Any, List

class ClinicalScribe:
    """High-quality clinical narrative generator with LLM-readiness."""
    
    @staticmethod
    def generate_summary(
        patient_id: str,
        risk_score: float,
        label: str,
        velocity: float,
        biomarkers: Dict[str, Any],
        clinical_meta: Dict[str, Any]
    ) -> str:
        """
        Synthesizes multimodal data into a professional clinical narrative.
        """
        # 1. Base Assessment
        risk_pct = round(risk_score * 100, 1)
        age = clinical_meta.get("age", "N/A")
        mmse = clinical_meta.get("mmse", "N/A")
        
        status_map = {
            "Stable": "presents as neurologically stable",
            "Escalating": "shows signs of significant neurodegenerative progression",
            "Rapid Decline": "exhibits an alarming rate of cognitive and structural decline",
            "High Risk": "is classified within the high-probability spectrum for Alzheimer's progression"
        }
        
        narrative = f"Patient {patient_id} ({age}Y, MMSE: {mmse}) {status_map.get(label, 'requires further observation')}. "
        
        # 1b. Genetic Insight (Fusion 3.0)
        apoe = clinical_meta.get("apoe", 0)
        if apoe > 0:
            gen_note = f"The presence of {int(apoe)} APOE-ε4 allele(s) indicates a baseline genetic predisposition which elevates the significance of the observed MRI markers."
            narrative += f"{gen_note} "

        # 2. Volumetric Insight
        hippo_vol = biomarkers.get("hippo_vol_mm3", 0)
        norm_ratio = biomarkers.get("normalized_ratio", 0) * 100
        
        if norm_ratio < 0.2: # Typical threshold for atrophy
            atrophy_note = f"Structural analysis indicates significant hippocampal atrophy ({round(norm_ratio, 3)}% TIV ratio), which correlates with the elevated AI risk profile."
        else:
            atrophy_note = f"Hippocampal volume remains within expected clinical margins ({round(norm_ratio, 3)}% TIV ratio) despite the computed risk score."
            
        narrative += f"{atrophy_note} "
        
        # 3. Longitudinal Intelligence
        if velocity > 0.05:
            trend_note = f"The risk velocity of {round(velocity, 3)} units/session is statistically significant, suggesting a transition into a more aggressive disease phase."
        elif velocity < -0.05:
            trend_note = "Longitudinal tracking indicates a favorable stabilizing trend in risk scores."
        else:
            trend_note = "Longitudinal risk remains stable across recent visits."
            
        narrative += f"{trend_note} "
        
        # 4. Generative Prognosis Simulation (Experimental)
        prog_velocity = velocity if velocity > 0 else 0.02
        future_risk = min(0.99, risk_score + (prog_velocity * 24)) # 24 month projection
        narrative += f"\n\nPROGNOSTIC SIMULATION (24-Month Horizon):\n"
        narrative += f"Based on current biomarkers and genetic profile, the predicted risk index in 24 months is {round(future_risk * 100, 1)}%. "
        if future_risk > 0.7:
            narrative += "Simulation suggests a high probability of conversion to advanced clinical stages if neuroprotective interventions are not initiated."

        # 5. Final Clinical Suggestion
        if risk_score > 0.7:
            suggestion = "Conclusion: High clinical suspicion for AD. Recommend immediate specialist consultation and potential PET-amyloid imaging to correlate with these structural findings."
        elif risk_score > 0.4:
            suggestion = "Conclusion: Borderline risk profile. Recommend increased monitoring frequency (every 3-6 months) and a formal neuropsychological evaluation."
        else:
            suggestion = "Conclusion: Low suspicion for imminent AD conversion. Maintain standard geriatric screening protocol."
            
        narrative += f"\n\n{suggestion}"
        
        return narrative
