import streamlit as st
import time
import uuid
from agent import CounselorAgent
from models import Conversation, PatientInfo
from database import Database
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Suicide Severity Assessment Assistant", 
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS styling."""
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def init_app():
    """Initialize the app."""
    if "db" not in st.session_state:
        st.session_state.db = Database()
    
    if "agent" not in st.session_state:
        st.session_state.agent = CounselorAgent()
    
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    
    # Always load the global conversation
    if "conversation" not in st.session_state:
        existing = st.session_state.db.load_conversation()
        if existing:
            st.session_state.conversation = existing
            print(f"✅ Loaded conversation with {len(existing.messages)} messages")
        else:
            st.session_state.conversation = Conversation(
                id=str(uuid.uuid4()),
                session_id="global",
                title="Mental Health Conversations"
            )
            print("✅ Created new conversation")

def save_conversation():
    """Save conversation to file."""
    if st.session_state.conversation:
        success = st.session_state.db.save_conversation(st.session_state.conversation)
        if success:
            print(f"✅ Saved conversation with {len(st.session_state.conversation.messages)} messages")

def display_patient_info_card(patient_info, message_id=""):
    """Display patient information in a visual card format."""
    
    # Determine mood indicators
    risk_indicators = []
    if patient_info.sleep_issues:
        risk_indicators.append("💤 Sleep Disturbances")
    if patient_info.appetite_changes:
        risk_indicators.append("🍽️ Appetite Changes")
    if patient_info.social_withdrawal:
        risk_indicators.append("🚪 Social Isolation")
    if patient_info.concentration_issues:
        risk_indicators.append("🧠 Concentration Problems")
    if patient_info.hopelessness:
        risk_indicators.append("⚠️ Hopelessness/Despair")
    
    # Energy level icon - handle enum
    energy_icon = {"low": "🔋", "normal": "⚡", "high": "⚡⚡"}
    energy_value = patient_info.energy_level.value if hasattr(patient_info.energy_level, 'value') else str(patient_info.energy_level)
    energy_display = f"{energy_icon.get(energy_value, '⚡')} {energy_value.title()}"
    age_display = patient_info.age or 'Not specified'
    
    # Create the basic card structure
    html_parts = [
        f'<div class="patient-info-card" id="patient-info-{message_id}">',
        '    <div class="patient-info-header">',
        '        <h3>🚨 Suicide Risk Factors Identified</h3>',
        '    </div>',
        '    <div class="patient-info-content">',
        '        <div class="info-row">',
        '            <div class="info-item">',
        '                <span class="info-label">👤 Age</span>',
        f'                <span class="info-value">{age_display}</span>',
        '            </div>',
        '            <div class="info-item">',
        '                <span class="info-label">⚡ Energy</span>',
        f'                <span class="info-value">{energy_display}</span>',
        '            </div>',
        '        </div>'
    ]
    
    # Add mood symptoms section if present - handle enum list
    if patient_info.mood_symptoms:
        html_parts.extend([
            '        <div class="mood-symptoms">',
            '            <span class="info-label">🎭 Risk Symptoms:</span>'
        ])
        for symptom in patient_info.mood_symptoms:
            symptom_value = symptom.value if hasattr(symptom, 'value') else str(symptom)
            html_parts.append(f'            <span class="mood-badge">{symptom_value}</span>')
        html_parts.append('        </div>')
    
    # Add indicators section
    if risk_indicators:
        html_parts.extend([
            '        <div class="indicators-grid">'
        ])
        for indicator in risk_indicators:
            html_parts.append(f'            <div class="indicator">{indicator}</div>')
        html_parts.append('        </div>')
    else:
        html_parts.append('        <div class="no-indicators">✅ No significant suicide risk indicators detected</div>')
    
    # Close the card
    html_parts.extend([
        '    </div>',
        '</div>'
    ])
    
    # Join all parts and display
    complete_html = '\n'.join(html_parts)
    st.markdown(complete_html, unsafe_allow_html=True)

def format_cssrs_result(content):
    """Extract and format CSSRS result for better display."""
    lines = content.split('\n')
    formatted_content = ""
    
    for line in lines:
        if "## CSSRS Assessment Result" in line:
            continue
        elif "**Score:" in line and "/10**" in line:
            # Extract score and severity
            import re
            score_match = re.search(r'\*\*Score: (\d+)/10\*\* - \*([^*]+)\*', line)
            if score_match:
                score = score_match.group(1)
                severity = score_match.group(2)
                formatted_content += f"""
<div class="phq-score-card">
    <div class="score-section">
        <div class="score-number">{score}</div>
        <div class="score-label">/ 10</div>
    </div>
    <div class="severity-section">
        <div class="severity-label">Suicide Risk</div>
        <div class="severity-text">{severity}</div>
    </div>
</div>
"""
            else:
                formatted_content += line + "\n"
        else:
            formatted_content += line + "\n"
    
    return formatted_content

def display_conversation_messages(conversation):
    """Display all messages in the conversation."""
    assessment_counter = 0
    i = 0
    
    while i < len(conversation.messages):
        if conversation.messages[i].role == "user":
            assessment_counter += 1
            user_msg = conversation.messages[i]
            
            # Display user message
            st.markdown(f"""
            <div class="message-container user-message">
                <div class="message-header">
                    <span class="role-label">Crisis Report</span>
                    <span class="message-timestamp">Assessment #{assessment_counter}</span>
                </div>
                <div class="message-content">{user_msg.content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Look for patient info
            if i + 1 < len(conversation.messages) and conversation.messages[i + 1].role == "patient_info":
                patient_info_msg = conversation.messages[i + 1]
                
                if isinstance(patient_info_msg.content, PatientInfo):
                    display_patient_info_card(patient_info_msg.content, f"msg-{assessment_counter}")
                
                # Look for assistant response
                if i + 2 < len(conversation.messages) and conversation.messages[i + 2].role == "assistant":
                    assistant_msg = conversation.messages[i + 2]
                    formatted_content = format_cssrs_result(str(assistant_msg.content))
                    st.markdown(f"""
                    <div class="message-container assistant-message">
                        <div class="message-header">
                            <span class="role-label">Risk Assessment</span>
                            <span class="message-timestamp">Assessment #{assessment_counter}</span>
                        </div>
                        <div class="message-content">{formatted_content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    i += 3
                else:
                    i += 2
            else:
                i += 1
        else:
            i += 1

def main():
    load_css()
    init_app()
    
    conversation = st.session_state.conversation
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-container">
            <div class="sidebar-header">
                <h2>🚨 Suicide Assessment</h2>
                <p>Crisis intervention & risk evaluation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div style="background: #f8f9fa; padding: 0.7rem; border-radius: 6px; margin: 0.7rem 0; border-left: 3px solid #007bff;">
            <h4 style="margin: 0 0 0.3rem 0; color: #007bff; font-size: 0.95rem;">📋 How to Use</h4>
            <ol style="margin: 0; padding-left: 1rem; font-size: 0.8rem; color: #333; line-height: 1.3;">
                <li style="color: #333; margin-bottom: 0.2rem;">Describe patient's suicidal thoughts, behaviors, and risk factors</li>
                <li style="color: #333; margin-bottom: 0.2rem;">Include relevant clinical observations</li>
                <li style="color: #333; margin-bottom: 0.2rem;">Review the extracted risk factors</li>
                <li style="color: #333; margin-bottom: 0.2rem;">Analyze the CSSRS severity score (0-10)</li>
                <li style="color: #333;">Follow the clinical guidance provided</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # CSSRS Scale Reference
        st.markdown("""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #ffc107;">
            <h4 style="margin: 0 0 0.5rem 0; color: #856404;">📊 CSSRS Scale</h4>
            <div style="font-size: 0.85rem; color: #856404;">
                <div><strong>0-1:</strong> Minimal risk</div>
                <div><strong>2-3:</strong> Low risk</div>
                <div><strong>4-5:</strong> Moderate risk</div>
                <div><strong>6-7:</strong> High risk</div>
                <div><strong>8-10:</strong> Severe risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #17a2b8;">
            <h4 style="margin: 0 0 0.5rem 0; color: #0c5460;">📚 Similar Cases</h4>
            <div style="font-size: 0.85rem; color: #0c5460;">
                Each assessment includes 3 similar cases from a database of 500 Reddit posts with expert labels:
                <div style="margin-top: 0.5rem;">
                    <div>🟢 <strong>Supportive</strong> - Low/no risk</div>
                    <div>🟡 <strong>Indicator</strong> - Warning signs</div>
                    <div>🟠 <strong>Ideation</strong> - Suicidal thoughts</div>
                    <div>🔴 <strong>Behavior</strong> - Concerning actions</div>
                    <div>🚨 <strong>Attempt</strong> - Suicide attempts</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show conversation stats or empty state
        if conversation.messages:
            st.markdown(f"""
            <div class="sidebar-stats">
                <div class="stat-item">
                    <span class="stat-number">{conversation.total_assessments}</span>
                    <span class="stat-label">Risk Assessments</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear All History", key="clear_btn", help="Clear all assessments", use_container_width=True):
                conversation.messages = []
                conversation.total_assessments = 0
                conversation.last_phq_score = None
                save_conversation()
                st.rerun()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666; border: 1px dashed #ccc; border-radius: 8px; margin: 1rem 0;">
                <h4>No assessment history yet</h4>
                <p>Start by describing patient's suicidal ideation, behaviors, and risk factors.</p>
            </div>
            """, unsafe_allow_html=True)
        
    
    # Main content
    st.markdown("""
    <div class="main-content">
        <div class="header-container">
            <h1 class="main-title">Suicide Severity Assessment Assistant</h1>
            <p class="subtitle">CSSRS-based crisis intervention and risk evaluation platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Always display conversation messages
    if not st.session_state.is_generating:
        display_conversation_messages(conversation)
    
    # Handle generation
    if st.session_state.is_generating:
        # Show existing messages first (excluding the current one being processed)
        if len(conversation.messages) > 1:
            temp_conversation = Conversation(
                id=conversation.id,
                session_id=conversation.session_id,
                title=conversation.title,
                messages=conversation.messages[:-1],  # Exclude last message
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                total_assessments=conversation.total_assessments,
                last_phq_score=conversation.last_phq_score
            )
            display_conversation_messages(temp_conversation)
        
        # Then show the generation process
        latest_message = conversation.messages[-1]
        assessment_num = conversation.total_assessments
        
        # Display current user message being processed
        st.markdown(f"""
        <div class="message-container user-message">
            <div class="message-header">
                <span class="role-label">Crisis Report</span>
                <span class="message-timestamp">Assessment #{assessment_num} (Processing...)</span>
            </div>
            <div class="message-content">{latest_message.content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Extract patient info
        with st.spinner("🔍 Extracting suicide risk factors..."):
            patient_info = st.session_state.agent.extract_patient_info_only(latest_message.content)
        
        display_patient_info_card(patient_info, f"current-{assessment_num}")
        conversation.add_message("patient_info", patient_info)
        save_conversation()
        
        # Generate response with streaming
        response_placeholder = st.empty()
        
        with st.spinner("🚨 Generating suicide risk assessment..."):
            full_response = st.session_state.agent.process_with_patient_info(latest_message.content, patient_info)
            
            # Simulate streaming
            words = full_response.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                formatted_current = format_cssrs_result(current_text + "...")
                
                with response_placeholder:
                    st.markdown(f"""
                    <div class="message-container assistant-message streaming">
                        <div class="message-header">
                            <span class="role-label">Risk Assessment</span>
                            <span class="message-timestamp">Generating...</span>
                        </div>
                        <div class="message-content">{formatted_current}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if i % 4 == 0:
                    time.sleep(0.08)
            
            # Add complete response
            conversation.add_message("assistant", full_response)
            
            # Update PHQ score
            import re
            score_match = re.search(r'\*\*Score: (\d+)/10\*\*', full_response)
            if score_match:
                conversation.last_phq_score = int(score_match.group(1))
            
            save_conversation()
            st.session_state.is_generating = False
            
            # Show final response
            formatted_final = format_cssrs_result(full_response)
            with response_placeholder:
                st.markdown(f"""
                <div class="message-container assistant-message">
                    <div class="message-header">
                        <span class="role-label">Risk Assessment</span>
                        <span class="message-timestamp">Assessment #{assessment_num}</span>
                    </div>
                    <div class="message-content">{formatted_final}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.rerun()
    
    # Input
    st.markdown('<div class="input-area"></div>', unsafe_allow_html=True)
    
    if st.session_state.is_generating:
        st.chat_input("Please wait for current risk assessment to complete...", disabled=True)
    else:
        if prompt := st.chat_input("Describe patient's suicidal thoughts, behaviors, risk factors, and clinical observations..."):
            conversation.add_message("user", prompt)
            save_conversation()
            st.session_state.is_generating = True
            st.rerun()

if __name__ == "__main__":
    main() 