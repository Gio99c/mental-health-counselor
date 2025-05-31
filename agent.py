import json
from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from pydantic import ValidationError

from models import PatientInfo
from prompts import PATIENT_INFO_EXTRACTION_PROMPT, SEVERITY_ASSESSMENT_PROMPT, CLINICAL_ADVICE_PROMPT
from rag_system import RedditRAG, format_similar_posts_for_display


class CounselorAgent:
    """LangGraph agent for mental health counselor assistance."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.1,
            seed=42
        ).bind(response_format={"type": "json_object"})
        
        self.patient_parser = PydanticOutputParser(pydantic_object=PatientInfo)
        self.fixing_parser = OutputFixingParser.from_llm(
            llm=self.llm,
            parser=self.patient_parser,
        )
        self.graph = self._build_graph()
        
        # Initialize RAG system
        try:
            self.rag_system = RedditRAG()
        except Exception as e:
            print(f"Warning: RAG system initialization failed: {e}")
            self.rag_system = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(dict)
        
        workflow.add_node("extract_info", self._extract_patient_info)
        workflow.add_node("assess_severity", self._assess_severity)
        workflow.add_node("generate_advice", self._generate_advice)
        
        workflow.add_edge("extract_info", "assess_severity")
        workflow.add_edge("assess_severity", "generate_advice")
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
    
    def _assess_severity(self, state: Dict) -> Dict:
        """Assess suicide severity using LLM."""
        patient_info = state["patient_info"]
        
        # Format patient info for prompt
        mood_symptoms_str = ", ".join([symptom.value for symptom in patient_info.mood_symptoms]) if patient_info.mood_symptoms else "None"
        energy_level_str = patient_info.energy_level.value.title()
        
        prompt = SEVERITY_ASSESSMENT_PROMPT.format(
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
        
        messages = [
            SystemMessage(content="You are a clinical suicide risk assessment expert. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = json.loads(response.content)
            severity_score = result.get("severity_score", 0)
            state["phq8_score"] = severity_score
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Severity assessment error: {e}")
            state["phq8_score"] = 0
        
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
    
    def _get_similar_cases(self, input_text: str) -> str:
        """Get similar cases from Reddit database."""
        if self.rag_system is None:
            return "\n## 📚 Similar Cases\n*RAG system not available*\n"
        
        try:
            similar_posts = self.rag_system.find_similar_posts(input_text, top_k=3)
            return "\n" + format_similar_posts_for_display(similar_posts)
        except Exception as e:
            print(f"Error retrieving similar cases: {e}")
            return "\n## 📚 Similar Cases\n*Error retrieving similar cases*\n"
    
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
        # Assess severity using LLM
        mood_symptoms_str = ", ".join([symptom.value for symptom in patient_info.mood_symptoms]) if patient_info.mood_symptoms else "None"
        energy_level_str = patient_info.energy_level.value.title()
        
        prompt = SEVERITY_ASSESSMENT_PROMPT.format(
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
        
        messages = [
            SystemMessage(content="You are a clinical suicide risk assessment expert. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = json.loads(response.content)
            severity_score = result.get("severity_score", 0)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Severity assessment error: {e}")
            severity_score = 0
        
        # Generate advice
        severity_level = self._get_severity_level(severity_score)
        
        # Format mood symptoms and energy level for display
        mood_symptoms_str = ", ".join([symptom.value for symptom in patient_info.mood_symptoms]) if patient_info.mood_symptoms else "None identified"
        energy_level_str = patient_info.energy_level.value.title()
        
        advice_prompt = CLINICAL_ADVICE_PROMPT.format(
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
            HumanMessage(content=advice_prompt)
        ]
        
        response = advice_llm.invoke(messages)
        
        # Get similar cases
        similar_cases = self._get_similar_cases(input_text)
        
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

{similar_cases}
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
            
            # Get similar cases
            similar_cases = self._get_similar_cases(input_text)
            
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

{similar_cases}
"""
            
        except Exception as e:
            return f"Error processing input: {str(e)}. Please try rephrasing your input." 