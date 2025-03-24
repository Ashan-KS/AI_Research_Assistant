# AI Research Assistant

## Overview
AI Research Assistant is an intelligent research agent built using **LangChain** for advanced reasoning and **Streamlit** for an interactive and user-friendly interface. This tool assists users in gathering, summarizing, and analyzing research materials efficiently, leveraging the power of AI-driven natural language processing.

## Features
- 🚀 **Automated Research**: Fetch and analyze research materials using AI-powered search.
- 📄 **Contextual Summarization**: Generate concise summaries of long articles or papers.
- 🔗 **Citation Extraction**: Identify and list key sources.
- 🎨 **Interactive UI**: User-friendly **Streamlit** front-end for seamless interaction.
- 📊 **Multi-Source Aggregation**: Collect data from multiple sources for a comprehensive research overview.

## Tech Stack
- **Backend**: LangChain, GPT-4o
- **Frontend**: Streamlit
- **Integration**: API connectors for external data sources

## Installation

### Prerequisites
Before installation, ensure you have the following dependencies installed:
- Python **3.10+**
- Pip (Python package manager)
- Virtual Environment (recommended for dependency isolation)

### Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Ai_Research_Assistant.git
   cd Ai_Research_Assistant
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows use `source venv/bin/activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To launch the AI Research Assistant, run the following command:
```bash
streamlit run app.py
```
This will start a local server and open the application in your web browser.

## Configuration
To use external APIs (e.g., OpenAI), create a `.env` file in the project root and add your API key:
```env
OPENAI_API_KEY=your_api_key_here
```
