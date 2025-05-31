import json
from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from pydantic import ValidationError

from models import PatientInfo
from ml_model import SuicideSeverityPredictor
from prompts import PATIENT_INFO_EXTRACTION_PROMPT, CLINICAL_ADVICE_PROMPT


class CounselorAgent:
    """LangGraph agent for mental health counselor assistance."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.1,
            seed=42
        ).bind(response_format={"type": "json_object"})
        
        self.ml_model = SuicideSeverityPredictor()
        self.patient_parser = PydanticOutputParser(pydantic_object=PatientInfo)
        self.fixing_parser = OutputFixingParser.from_llm(
            llm=self.llm,
            parser=self.patient_parser,
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(dict)
        
        workflow.add_node("extract_info", self._extract_patient_info)
        workflow.add_node("predict_phq8", self._predict_phq8)
        workflow.add_node("generate_advice", self._generate_advice)
        
        workflow.add_edge("extract_info", "predict_phq8")
        workflow.add_edge("predict_phq8", "generate_advice")
        workflow.add_edge("generate_advice", END)
        
        workflow.set_entry_point("extract_info")
        return workflow.compile()
    
    def _extract_patient_info(self, state: Dict) -> Dict:
        """Extract patient information using LLM with PydanticOutputParser."""
        prompt = PromptTemplate(
            template=PATIENT_INFO_EXTRACTION_PROMPT,
            input_variables=["input_text"],
            partial_variables={"format_instructions": self.patient_parser.get_format_instructions()}
        )
        
        formatted_prompt = prompt.format(input_text=state["input_text"])
        
        messages = [
            SystemMessage(content="You are a clinical information extraction expert. Return only valid JSON."),
            HumanMessage(content=formatted_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            patient_info = self.fixing_parser.parse(response.content)
            state["patient_info"] = patient_info
        except ValidationError as e:
            print(f"Validation error: {e}")
            state["patient_info"] = PatientInfo()
        
        return state
    
    def _predict_phq8(self, state: Dict) -> Dict:
        """Predict suicide severity score using ML model."""
        patient_info = state["patient_info"]
        original_input = state["input_text"]
        severity_score = self.ml_model.predict(patient_info, original_input)
        state["phq8_score"] = severity_score
        return state
    
    def _generate_advice(self, state: Dict) -> Dict:
        """Generate clinical advice using LLM."""
        patient_info = state["patient_info"]
        phq8_score = state["phq8_score"]
        severity_level = self._get_severity_level(phq8_score)
        
        # Format mood symptoms and energy level for display
        mood_symptoms_str = ", ".join([symptom.value for symptom in patient_info.mood_symptoms]) if patient_info.mood_symptoms else "None identified"
        energy_level_str = patient_info.energy_level.value.title()
        
        prompt = CLINICAL_ADVICE_PROMPT.format(
            phq8_score=phq8_score,
            severity_level=severity_level,
            age=patient_info.age or "Not specified",
            sleep_issues="Yes" if patient_info.sleep_issues else "No",
            appetite_changes="Yes" if patient_info.appetite_changes else "No",
            energy_level=energy_level_str,
            mood_symptoms=mood_symptoms_str,
            social_withdrawal="Yes" if patient_info.social_withdrawal else "No",
            concentration_issues="Yes" if patient_info.concentration_issues else "No",
            hopelessness="Yes" if patient_info.hopelessness else "No",
            original_input=state["input_text"]
        )
        
        advice_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        messages = [
            SystemMessage(content="You are an expert mental health counselor providing colleague guidance."),
            HumanMessage(content=prompt)
        ]
        
        response = advice_llm.invoke(messages)
        state["advice"] = response.content
        state["processed"] = True
        
        return state
    
    def _get_severity_level(self, score: int) -> str:
        """Convert CSSRS score to suicide severity level."""
        if score <= 1:
            return "Minimal risk"
        elif score <= 3:
            return "Low risk"
        elif score <= 5:
            return "Moderate risk"
        elif score <= 7:
            return "High risk"
        else:
            return "Severe risk"
    
    def extract_patient_info_only(self, input_text: str) -> PatientInfo:
        """Extract only patient information for UI display."""
        prompt = PromptTemplate(
            template=PATIENT_INFO_EXTRACTION_PROMPT,
            input_variables=["input_text"],
            partial_variables={"format_instructions": self.patient_parser.get_format_instructions()}
        )
        
        formatted_prompt = prompt.format(input_text=input_text)
        
        messages = [
            SystemMessage(content="You are a clinical information extraction expert. Return only valid JSON."),
            HumanMessage(content=formatted_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return self.fixing_parser.parse(response.content)
        except ValidationError as e:
            print(f"Validation error: {e}")
            return PatientInfo()
    
    def process_with_patient_info(self, input_text: str, patient_info: PatientInfo) -> str:
        """Process input with pre-extracted patient info."""
        # Predict suicide severity score
        severity_score = self.ml_model.predict(patient_info, input_text)
        
        # Generate advice
        severity_level = self._get_severity_level(severity_score)
        
        # Format mood symptoms and energy level for display
        mood_symptoms_str = ", ".join([symptom.value for symptom in patient_info.mood_symptoms]) if patient_info.mood_symptoms else "None identified"
        energy_level_str = patient_info.energy_level.value.title()
        
        prompt = CLINICAL_ADVICE_PROMPT.format(
            phq8_score=severity_score,
            severity_level=severity_level,
            age=patient_info.age or "Not specified",
            sleep_issues="Yes" if patient_info.sleep_issues else "No",
            appetite_changes="Yes" if patient_info.appetite_changes else "No",
            energy_level=energy_level_str,
            mood_symptoms=mood_symptoms_str,
            social_withdrawal="Yes" if patient_info.social_withdrawal else "No",
            concentration_issues="Yes" if patient_info.concentration_issues else "No",
            hopelessness="Yes" if patient_info.hopelessness else "No",
            original_input=input_text
        )
        
        advice_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        messages = [
            SystemMessage(content="You are an expert mental health counselor providing colleague guidance."),
            HumanMessage(content=prompt)
        ]
        
        response = advice_llm.invoke(messages)
        
        return f"""
## CSSRS Assessment Result
**Score: {severity_score}/10** - *{severity_level}*

## Extracted Patient Information
- **Age:** {patient_info.age or 'Not specified'}
- **Sleep Issues:** {'Yes' if patient_info.sleep_issues else 'No'}
- **Appetite Changes:** {'Yes' if patient_info.appetite_changes else 'No'}
- **Energy Level:** {energy_level_str}
- **Mood Symptoms:** {mood_symptoms_str}
- **Social Withdrawal:** {'Yes' if patient_info.social_withdrawal else 'No'}
- **Concentration Issues:** {'Yes' if patient_info.concentration_issues else 'No'}
- **Hopelessness Indicators:** {'Yes' if patient_info.hopelessness else 'No'}

## Clinical Guidance
{response.content}
"""
    
    def process_input(self, input_text: str) -> str:
        """Process counselor input and return clinical guidance."""
        initial_state = {
            "input_text": input_text,
            "patient_info": None,
            "phq8_score": None,
            "advice": "",
            "processed": False
        }
        
        try:
            result = self.graph.invoke(initial_state)
            
            # Format the response
            patient_info = result["patient_info"]
            phq8_score = result["phq8_score"]
            severity = self._get_severity_level(phq8_score)
            advice = result["advice"]
            
            # Format mood symptoms and energy level for display
            mood_symptoms_str = ", ".join([symptom.value for symptom in patient_info.mood_symptoms]) if patient_info.mood_symptoms else "None identified"
            energy_level_str = patient_info.energy_level.value.title()
            
            return f"""
## CSSRS Assessment Result
**Score: {phq8_score}/10** - *{severity}*

## Extracted Patient Information
- **Age:** {patient_info.age or 'Not specified'}
- **Sleep Issues:** {'Yes' if patient_info.sleep_issues else 'No'}
- **Appetite Changes:** {'Yes' if patient_info.appetite_changes else 'No'}
- **Energy Level:** {energy_level_str}
- **Mood Symptoms:** {mood_symptoms_str}
- **Social Withdrawal:** {'Yes' if patient_info.social_withdrawal else 'No'}
- **Concentration Issues:** {'Yes' if patient_info.concentration_issues else 'No'}
- **Hopelessness Indicators:** {'Yes' if patient_info.hopelessness else 'No'}

## Clinical Guidance
{advice}
"""
            
        except Exception as e:
            return f"Error processing input: {str(e)}. Please try rephrasing your input." 