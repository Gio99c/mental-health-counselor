import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
import os

class SuicideSeverityPredictor:
    def __init__(self, model_path: str = "./suicide_severity_model"):
        """
        Initialize the BERT-based suicide severity predictor.
        
        Args:
            model_path: Path to the trained BERT model
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load model and tokenizer
            print(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ BERT model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Failed to load BERT model: {e}")
            print("⚠️ Falling back to rule-based prediction")
            self.model_loaded = False
            self._init_fallback_weights()
    
    def _init_fallback_weights(self):
        """Initialize fallback rule-based weights if BERT model fails to load."""
        self.feature_weights = {
            'sleep_issues': 1.5,
            'appetite_changes': 1.2,
            'energy_level_low': 2.0,
            'energy_level_high': -0.5,
            'mood_symptoms_count': 1.8,
            'social_withdrawal': 2.2,
            'concentration_issues': 1.5,
            'hopelessness': 3.5,  # Highest weight for suicide risk
            'age_factor': 0.03
        }
        self.base_score = 1.0
    
    def _create_text_from_patient_info(self, patient_info, original_input: str = "") -> str:
        """
        Create a text representation from patient information for BERT input.
        
        Args:
            patient_info: PatientInfo object
            original_input: Original counselor input text
            
        Returns:
            Combined text for BERT processing
        """
        # Start with original input if available
        text_parts = []
        
        if original_input:
            text_parts.append(f"Clinical observation: {original_input}")
        
        # Add structured patient information
        patient_details = []
        
        if patient_info.age:
            patient_details.append(f"Patient age: {patient_info.age}")
        
        # Add boolean indicators
        if patient_info.sleep_issues:
            patient_details.append("Sleep disturbances present")
        
        if patient_info.appetite_changes:
            patient_details.append("Appetite changes observed")
        
        if patient_info.social_withdrawal:
            patient_details.append("Social withdrawal behavior")
        
        if patient_info.concentration_issues:
            patient_details.append("Concentration problems noted")
        
        if patient_info.hopelessness:
            patient_details.append("Hopelessness and despair indicators")
        
        # Add energy level
        if hasattr(patient_info.energy_level, 'value'):
            energy = patient_info.energy_level.value
        else:
            energy = str(patient_info.energy_level)
        patient_details.append(f"Energy level: {energy}")
        
        # Add mood symptoms
        if patient_info.mood_symptoms:
            symptoms = []
            for symptom in patient_info.mood_symptoms:
                if hasattr(symptom, 'value'):
                    symptoms.append(symptom.value)
                else:
                    symptoms.append(str(symptom))
            patient_details.append(f"Mood symptoms: {', '.join(symptoms)}")
        
        if patient_details:
            text_parts.append("Patient information: " + ". ".join(patient_details))
        
        return " ".join(text_parts)
    
    def predict(self, patient_info, original_input: str = "") -> int:
        """
        Predict suicide severity score from patient information.
        
        Args:
            patient_info: PatientInfo object
            original_input: Original counselor input text
            
        Returns:
            Severity score (0-10)
        """
        if self.model_loaded:
            return self._predict_with_bert(patient_info, original_input)
        else:
            return self._predict_with_fallback(patient_info)
    
    def _predict_with_bert(self, patient_info, original_input: str = "") -> int:
        """Predict using BERT model."""
        try:
            # Create text input
            text = self._create_text_from_patient_info(patient_info, original_input)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = outputs.logits.squeeze().cpu().numpy()
            
            # Convert to 0-10 scale and ensure integer
            score = np.clip(np.round(float(prediction)), 0, 10).astype(int)
            
            return int(score)
            
        except Exception as e:
            print(f"❌ BERT prediction failed: {e}")
            print("⚠️ Falling back to rule-based prediction")
            return self._predict_with_fallback(patient_info)
    
    def _predict_with_fallback(self, patient_info) -> int:
        """Fallback rule-based prediction."""
        score = self.base_score
        
        # Sleep issues contribution
        if patient_info.sleep_issues:
            score += self.feature_weights['sleep_issues']
        
        # Appetite changes contribution
        if patient_info.appetite_changes:
            score += self.feature_weights['appetite_changes']
        
        # Energy level contribution
        energy_level = patient_info.energy_level
        if hasattr(energy_level, 'value'):
            energy_level = energy_level.value
        
        if str(energy_level).lower() == 'low':
            score += self.feature_weights['energy_level_low']
        elif str(energy_level).lower() == 'high':
            score += self.feature_weights['energy_level_high']
        
        # Mood symptoms contribution
        mood_count = len(patient_info.mood_symptoms) if patient_info.mood_symptoms else 0
        score += mood_count * self.feature_weights['mood_symptoms_count']
        
        # Social withdrawal contribution
        if patient_info.social_withdrawal:
            score += self.feature_weights['social_withdrawal']
        
        # Concentration issues contribution
        if patient_info.concentration_issues:
            score += self.feature_weights['concentration_issues']
        
        # Hopelessness contribution (highest weight for suicide risk)
        if patient_info.hopelessness:
            score += self.feature_weights['hopelessness']
        
        # Age factor
        if patient_info.age:
            if patient_info.age < 25:
                score += 0.8  # Higher risk for younger adults
            elif patient_info.age > 65:
                score += 0.5  # Elevated risk for elderly
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.3)
        score += noise
        
        # Ensure score is within valid range (0-10)
        score = max(0, min(10, round(score)))
        
        return int(score)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model for transparency."""
        if self.model_loaded:
            return {
                "model_type": "BERT-based Suicide Severity Predictor",
                "model_path": self.model_path,
                "architecture": "DistilBERT",
                "score_range": "0-10 (CSSRS-based)",
                "description": "Fine-tuned BERT model for suicide risk assessment",
                "device": str(self.device)
            }
        else:
            return {
                "model_type": "Rule-based Suicide Severity Predictor (Fallback)",
                "features": list(self.feature_weights.keys()),
                "score_range": "0-10 (CSSRS-based)",
                "description": "Fallback model when BERT is unavailable"
            }


# For backward compatibility, create an alias
PHQ8Predictor = SuicideSeverityPredictor 