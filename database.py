import json
import os
from datetime import datetime
from typing import Optional
from pymongo import MongoClient
from models import Conversation, ChatMessage, PatientInfo

class Database:
    """MongoDB storage for mental health conversation data."""
    
    def __init__(self,):
        try:
            MONGODB_URI = os.getenv("MONGODB_URI")
            MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
            MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")
            
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[MONGODB_DATABASE]
            self.conversations = self.db[MONGODB_COLLECTION]
            self.client.admin.command('ping')
        except Exception:
            self.client = None
            self.db = None
            self.conversations = None
    
    def save_conversation(self, conversation: Conversation) -> bool:
        """Save conversation to MongoDB."""
        if self.conversations is None:
            return False
            
        try:
            data = {
                "_id": "global_conversation",
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": datetime.utcnow(),
                "total_assessments": conversation.total_assessments,
                "last_phq_score": conversation.last_phq_score,
                "messages": []
            }
            
            for msg in conversation.messages:
                msg_data = {
                    "role": msg.role,
                    "timestamp": msg.timestamp,
                }
                
                if msg.role == "patient_info" and isinstance(msg.content, PatientInfo):
                    msg_data["content"] = json.loads(msg.content.model_dump_json())
                    msg_data["content_type"] = "patient_info"
                else:
                    msg_data["content"] = str(msg.content)
                    msg_data["content_type"] = "text"
                
                data["messages"].append(msg_data)
            
            self.conversations.replace_one(
                {"_id": "global_conversation"}, 
                data, 
                upsert=True
            )
            
            return True
            
        except Exception:
            return False
    
    def load_conversation(self) -> Optional[Conversation]:
        """Load conversation from MongoDB."""
        if self.conversations is None:
            return None
            
        try:
            data = self.conversations.find_one({"_id": "global_conversation"})
            if not data:
                return None
            
            messages = []
            for msg_data in data.get("messages", []):
                content = msg_data["content"]
                
                if msg_data.get("content_type") == "patient_info":
                    try:
                        content = PatientInfo(**content)
                    except Exception:
                        content = str(content)
                
                message = ChatMessage(
                    role=msg_data["role"],
                    content=content,
                    timestamp=msg_data["timestamp"]
                )
                messages.append(message)
            
            conversation = Conversation(
                id=data.get("id"),
                session_id="global",
                title=data.get("title", "Mental Health Conversations"),
                messages=messages,
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                total_assessments=data.get("total_assessments", 0),
                last_phq_score=data.get("last_phq_score")
            )
            
            return conversation
            
        except Exception:
            return None 