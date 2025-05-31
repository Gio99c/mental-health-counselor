# Suicide Severity Assessment Assistant

A clinical decision support system for mental health professionals to assess suicide risk using the Columbia Suicide Severity Rating Scale (CSSRS).

## Features

- **CSSRS-based Risk Assessment**: Standardized 0-10 severity scoring
- **Intelligent Information Extraction**: Automated parsing of clinical notes
- **Real-time Risk Scoring**: Immediate assessment feedback
- **Clinical Guidance**: Evidence-based intervention recommendations
- **Secure Data Storage**: MongoDB-based conversation persistence
- **Professional Interface**: Healthcare-focused UI design

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB (local or cloud instance)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-counselor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export MONGODB_URI="your-mongodb-connection-string"
export MONGODB_DATABASE="mental_health"
export MONGODB_COLLECTION="conversations"
```

4. Launch the application:
```bash
streamlit run app.py
```

## Usage

1. Describe patient's suicidal thoughts, behaviors, and risk factors
2. Review extracted risk indicators
3. Analyze the CSSRS severity score (0-10)
4. Follow the provided clinical guidance

## Risk Scale

- **0-1**: Minimal risk
- **2-3**: Low risk  
- **4-5**: Moderate risk
- **6-7**: High risk
- **8-10**: Severe risk

## Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, OpenAI GPT-4
- **Database**: MongoDB
- **Data Models**: Pydantic
- **Workflow**: LangGraph

