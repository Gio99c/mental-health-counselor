"""
Pydantic models for the Mental Health Counselor Assistant
"""
from enum import Enum
from typing import List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId
from typing_extensions import TypedDict


class EnergyLevel(Enum):
    low = "low"
    normal = "normal"
    high = "high"


class MoodSymptom(Enum):
    sadness = "sadness"
    anxiety = "anxiety"
    irritability = "irritability"
    emptiness = "emptiness"


class AgentState(TypedDict):
    input_text: str
    patient_info: Optional['PatientInfo']
    phq8_score: Optional[int]
    advice: str
    processed: bool


class PatientInfo(BaseModel):
    """Validated patient information extracted from counselor input."""
    age: Optional[int] = Field(None, description="Patient's age in years, null if not mentioned")
    sleep_issues: bool = Field(False, description="True if patient reports sleep problems, insomnia, or sleep disturbances")
    appetite_changes: bool = Field(False, description="True if patient reports eating more/less, weight changes, or food issues")
    energy_level: EnergyLevel = Field(EnergyLevel.normal, description="Patient's energy level: low for fatigue/tiredness, high for restlessness/agitation, normal otherwise")
    mood_symptoms: List[MoodSymptom] = Field(default_factory=list, description="List of mood symptoms from allowed values: sadness, anxiety, irritability, emptiness")
    social_withdrawal: bool = Field(False, description="True if patient mentions isolation, avoiding people, or staying alone")
    concentration_issues: bool = Field(False, description="True if patient mentions focus problems, memory issues, or difficulty thinking")
    hopelessness: bool = Field(False, description="True if patient mentions despair, suicidal thoughts, or feeling worthless")


class ChatMessage(BaseModel):
    """Individual chat message in a conversation."""
    role: str = Field(..., description="Message role: user, assistant, or patient_info")
    content: Union[str, PatientInfo, Any] = Field(..., description="Message content, varies by role")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            PatientInfo: lambda pi: pi.dict()
        }


class Conversation(BaseModel):
    """Complete conversation thread with metadata."""
    id: Optional[str] = Field(None, description="MongoDB ObjectId as string")
    session_id: str = Field(..., description="Unique session identifier")
    title: str = Field(..., description="Conversation title from first user message preview")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of chat messages")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    total_assessments: int = Field(0, description="Total number of assessments in conversation")
    last_phq_score: Optional[int] = Field(None, description="Most recent PHQ-8 score")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            ObjectId: str
        }
    
    def add_message(self, role: str, content: Union[str, PatientInfo, Any]):
        """Add a new message to the conversation."""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        
        # Update metadata
        if role == "user":
            self.total_assessments += 1
            if not self.title:  # Set title from first user message
                preview = content[:50] if isinstance(content, str) else "Assessment"
                self.title = preview + "..." if len(str(content)) > 50 else str(content)
    
    def get_patient_summary(self) -> str:
        """Generate summary of latest patient information."""
        # Find the most recent patient_info message
        for message in reversed(self.messages):
            if message.role == "patient_info" and isinstance(message.content, PatientInfo):
                patient_info = message.content
                summary_parts = []
                
                if patient_info.age:
                    summary_parts.append(f"Age {patient_info.age}")
                
                # Count indicators
                indicators = []
                if patient_info.sleep_issues:
                    indicators.append("Sleep")
                if patient_info.appetite_changes:
                    indicators.append("Appetite")
                if patient_info.social_withdrawal:
                    indicators.append("Social")
                if patient_info.concentration_issues:
                    indicators.append("Focus")
                if patient_info.hopelessness:
                    indicators.append("Hopelessness")
                
                if indicators:
                    summary_parts.append(f"Issues: {', '.join(indicators)}")
                
                if patient_info.mood_symptoms:
                    symptoms = [symptom.value for symptom in patient_info.mood_symptoms[:2]]
                    summary_parts.append(f"Mood: {', '.join(symptoms)}")
                
                return " • ".join(summary_parts) if summary_parts else "Basic assessment"
        
        return "No patient data" 