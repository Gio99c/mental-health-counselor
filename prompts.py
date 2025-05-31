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

SEVERITY_ASSESSMENT_PROMPT = """
Assess the suicide severity risk based on the CSSRS scale (0-10) from the patient information and original input.

Use this scale:
- 0-1: Minimal risk (no suicidal ideation)
- 2-3: Low risk (passive thoughts, no plan)
- 4-5: Moderate risk (active ideation, vague plan)
- 6-7: High risk (specific plan, intent)
- 8-10: Severe risk (imminent danger, previous attempts)

## Examples:

Example 1:
Patient: Age 25, no sleep issues, normal energy, reports sadness, no social withdrawal
Input: "Patient says they feel down but are looking forward to weekend plans with family"
Output: {{"severity_score": 1}}

Example 2:
Patient: Age 40, sleep issues, low energy, hopelessness present, social withdrawal
Input: "Patient mentions sometimes wishing they wouldn't wake up but has no specific plans"
Output: {{"severity_score": 3}}

Example 3:
Patient: Age 30, multiple issues, hopelessness, mentions specific method but no timeline
Input: "Patient has been researching ways to end their life and feels like a burden to family"
Output: {{"severity_score": 5}}

Example 4:
Patient: Age 45, severe symptoms, has detailed plan and timeline, previous attempt
Input: "Patient has pills saved up, plans to use them this weekend, attempted overdose last year"
Output: {{"severity_score": 8}}

## Your Assessment:

Patient Information:
- Age: {age}
- Sleep Issues: {sleep_issues}
- Appetite Changes: {appetite_changes}
- Energy Level: {energy_level}
- Mood Symptoms: {mood_symptoms}
- Social Withdrawal: {social_withdrawal}
- Concentration Issues: {concentration_issues}
- Hopelessness: {hopelessness}

Original Input: {original_input}

Return only a JSON object with the severity score:
{{"severity_score": <number>}}
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