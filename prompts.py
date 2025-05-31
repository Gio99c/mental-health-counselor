
PATIENT_INFO_EXTRACTION_PROMPT = """
Extract patient information from the counselor's clinical notes for suicide risk assessment.

Look for indicators related to:
- Suicidal ideation (thoughts, plans, intent)
- Previous suicide attempts
- Risk factors (depression, substance use, isolation, etc.)
- Protective factors (support systems, coping skills, etc.)
- Current mental state and behavior patterns

INPUT TEXT:
{input_text}

{format_instructions}
"""

CLINICAL_ADVICE_PROMPT = """
Provide clinical guidance based on this suicide severity assessment:

CSSRS Severity Score: {phq8_score}/10 ({severity_level})
Age: {age}
Sleep Issues: {sleep_issues}
Appetite Changes: {appetite_changes} 
Energy Level: {energy_level}
Mood Symptoms: {mood_symptoms}
Social Withdrawal: {social_withdrawal}
Concentration Issues: {concentration_issues}
Hopelessness: {hopelessness}

Original Input: {original_input}

Provide structured clinical recommendations covering:
1. Immediate suicide risk assessment
2. Safety planning and interventions
3. Level of care recommendations (inpatient, outpatient, crisis intervention)
4. Protective factors to strengthen
5. Monitoring and follow-up protocols
6. Emergency referrals if indicated

Focus on evidence-based suicide prevention strategies and CSSRS-guided interventions.
""" 