# **ThreadWise Planner** 
AI-Powered Adaptive Study Planner with Smart Scheduling

> **Creativity, Science, and Innovation (CSI) Course Project**  
> Politecnico di Milano - January 2026

---

##  About the Project

ThreadWise is an intelligent web application designed to help students overcome **"decision fatigue"** and optimize their study schedules. It uses Large Language Models (**Qwen 2.5-Coder-32B**) to convert unstructured, free-form study inputs into scientifically optimized schedules based on proven learning strategies.

**Key Problem Solved:** Students often waste time deciding *what*, *when*, and *how* to study. ThreadWise automates this process by parsing natural language inputs and generating structured, deadline-aware study plans.

---

##  Features

###  **Smart Natural Language Parsing**
- Accepts **any format** input (e.g., "Wireless internet, 10/01/2026, Hard" or "Math exam Friday, difficult")
- AI-powered validation ensures all required fields (Course Name, Deadline, Difficulty) are present
- Automatically normalizes dates to ISO 8601 format (`YYYY-MM-DD`)

###  **Scientific Study Strategies**
Choose from four evidence-based approaches:

1. **Waterfall (Cascade):** Prioritizes high-difficulty tasks first (Hard → Medium → Easy)
2. **Sandwich (Interleaving):** Alternates hard and easy tasks to prevent burnout
3. **Sequential (Focus):** Groups tasks by subject to minimize context switching (Deep Work mode)
4. **Random Mix:** Randomizes topics to simulate exam conditions and improve recall

### ⚙️ **Flexible Configuration**
- Set **Max hours/day** (default: 8 hours)
- Customize **break duration** (default: 30 minutes)
- Automatic time allocation based on deadlines and difficulty levels

###  **Export Options**
- **CSV (Excel):** Download your schedule as a spreadsheet
- **ICS (Calendar):** Import directly into Google Calendar, Outlook, or Apple Calendar

###  **AI Guardrails**
- Rejects non-academic queries to maintain focus
- Validates input completeness before generating schedules
- Provides clear, actionable feedback for missing information

---

##  Installation & Usage

### Prerequisites
- Python 3.8 or higher
- A [Hugging Face](https://huggingface.co/) account and API token

### Step 1: Clone the Repository
```bash
git clone https://github.com/Mahdi-sr95/ThreadWise-Planner-CSI.git
cd ThreadWise-Planner-CSI
```

### Step 2: Install Dependencies
```bash
  pip install -r requirements.txt
```

### Step 3: Configure API Token
Create a folder named .streamlit in the project root:
```bash
mkdir .streamlit
```
### Inside .streamlit, create a file named secrets.toml and add your Hugging Face token:
HF_TOKEN = "huggingface_token"

### Step 4: Run the Application
```bash
streamlit run app.py
```
The app will open in your browser at http://localhost:8501

### 📝 Example Usage
Input (Free-form)
1- Wireless internet, 10/01/2026, Hard
2- Multimedia, 12/01/2026, Medium  
3- CSI, 09/JAN/2026, Hard

### Output
A structured study plan with:
Day: Date and time for each study session
Subject: Course name
Task: Recommended activity (e.g., "Practice Problems", "Review Concepts")
Duration: Time allocated (formatted as "2h 30 min")
Difficulty: Easy/Medium/Hard

### 🛠️ Technical Stack
Frontend: Streamlit
AI Model: Qwen 2.5-Coder-32B via Hugging Face Inference API
Data Processing: Pandas, Python datetime
Calendar Export: ics library

### 👥 Team
Team ThreadWise:

Mahdi Soltani Renani
Emad Karimianshamsabadi
Lucas Lescure

### Course: Creativity, Science, and Innovation (CSI)
Institution: Politecnico di Milano
Date: January 2026

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments
Thanks to the Hugging Face team for providing free API access to state-of-the-art LLMs
Inspired by research on Interleaving, Spaced Repetition, and Cognitive Load Theory
Special thanks to the CSI course instructors at Politecnico di Milano

### 📧 Contact
For questions or feedback, please open an issue or contact the team via GitHub.
⭐ If you find this project useful, please star the repository!


