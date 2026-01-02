import streamlit as st
import pandas as pd
from ics import Calendar, Event
from datetime import datetime, timedelta
from huggingface_hub import InferenceClient
import io
import re
import math
import json

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Threadwise Study Planner",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto"
)

# -----------------------------------------------------------------------------
# 2. AUTHENTICATION & CONFIG
# -----------------------------------------------------------------------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    st.error("HF_TOKEN not found in .streamlit/secrets.toml")
    st.stop()

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# -----------------------------------------------------------------------------
# 3. CSS & STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Add your custom CSS here if needed */
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3.5 AI GATEKEEPER (SCOPE + COMPLETENESS) - FIXED
# -----------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def _safe_extract_json(text: str):
    """Extract the first JSON object found in text; return dict or None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def ai_gatekeeper(accumulated_text: str, latest_prompt: str, plan_exists: bool):
    """LLM-based scope + completeness classifier.
    FIXED: Now REQUIRES Difficulty for each course.
    """
    system_gate = (
        "You are a strict classifier for a study-planning web app. "
        "Decide whether the user's input is: "
        "(1) unrelated to study planning, "
        "(2) related but incomplete, or "
        "(3) complete enough to generate a study plan. "
        "\n\n"
        "The app domain is academic study planning: courses/subjects, deadlines/due dates, and difficulty levels. "
        "The user may provide details in any format, order, language, or across multiple messages. "
        "You will receive the accumulated context and the latest prompt. "
        "\n\n"
        "If PLAN_EXISTS is true, the user may be asking to modify or refine an existing study plan; treat such requests as RELATED (not unrelated). "
        "Only classify as UNRELATED when the request is clearly not about study planning (e.g., politics, cooking, travel), especially general questions. "
        "\n\n"
        "For initial plan creation (PLAN_EXISTS=false), classify as COMPLETE only if ALL of these are satisfied: "
        "\n- At least one course/subject name exists"
        "\n- For EVERY mentioned course, there is a deadline/due date somewhere in the accumulated text (accept many date formats)"
        "\n- For EVERY mentioned course, there is a difficulty level (Easy/Medium/Hard or Low/Medium/High) somewhere in the accumulated text"
        "\n\n"
        "IMPORTANT PARSING RULES:"
        "\n- Course names can appear ANYWHERE before or after dates in the text"
        "\n- Accept formats like: '1- CourseName, Date' OR 'CourseName, Date' OR 'CourseName Date' (with or without separators)"
        "\n- Even if courses and dates are not clearly separated by commas/dashes, try to extract course names from the text chunks between dates"
        "\n- Extract course names by finding text tokens that appear near dates and are not themselves dates"
        "\n- If you find text like 'Wireless internet 10/01/2026 Multimedia 12/01/2026', parse it as two courses: 'Wireless internet' and 'Multimedia'"
        "\n\n"
        "Do NOT require daily study time per course; the app uses a global Max hours/day input. "
        "If deadlines OR difficulty levels are missing for one or more courses, classify as INCOMPLETE. "
        "\n\n"
        "Output MUST be a single JSON object with exactly these keys:"
        "\n- status: one of [\"unrelated\", \"incomplete\", \"complete\"]"
        "\n- message: a short user-facing Markdown message in English"
        "\n- missing: an array of missing items (only for status=incomplete), allowed values: Courses, Deadlines, Difficulty"
        "\n\nDo not output any other text."
    )

    user_gate = (
        "PLAN_EXISTS: " + ("true" if plan_exists else "false") +
        "\n\nACCUMULATED_TEXT:\n" + str(accumulated_text) +
        "\n\nLATEST_PROMPT:\n" + str(latest_prompt)
    )

    raw = get_ai_response([
        {"role": "system", "content": system_gate},
        {"role": "user", "content": user_gate},
    ])
    data = _safe_extract_json(raw)
    if isinstance(data, dict) and "status" in data and "message" in data:
        return data
    
    return {
        "status": "incomplete",
        "missing": ["Courses", "Deadlines", "Difficulty"],
        "message": (
            "⚠️ **Incomplete Prompt**\n\n"
            "Your prompt:\n`" + _normalize_text(latest_prompt) + "`\n\n"
            "Please provide:\n"
            "- Course names\n"
            "- A deadline/due date for each course (any format is OK)\n"
            "- Difficulty level for each course (Easy/Medium/Hard)"
        ),
    }

# -----------------------------------------------------------------------------
# 4. LOGIC ALGORITHMS & HELPERS
# -----------------------------------------------------------------------------
def normalize_difficulty(val):
    val = str(val).lower().strip()
    if 'high' in val or 'hard' in val: return 3
    if 'medium' in val: return 2
    if 'low' in val or 'easy' in val: return 1
    return 2

def format_duration_display(val):
    """Convert decimal hours to a clear human-readable string."""
    try:
        val_str = str(val).strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)", val_str)
        if not match:
            return str(val).strip()
        hours_decimal = float(match.group(1))
        if hours_decimal < 0:
            hours_decimal = 0.0
        total_minutes = int(round(hours_decimal * 60))
        hours = total_minutes // 60
        minutes = total_minutes % 60
        if hours <= 0:
            return f"{minutes} min"
        if minutes == 0:
            return f"{hours}h"
        return f"{hours}h {minutes} min"
    except Exception:
        return str(val)

def standardize_columns(df):
    """Renames columns to ensure they match the required format."""
    if df is None or df.empty:
        return df
    
    col_map = {
        'Date': 'Day',
        'Time': 'Day',
        'Course': 'Subject',
        'Topic': 'Task',
        'Activity': 'Task',
        'Length': 'Duration',
        'Hours': 'Duration',
        'Level': 'Difficulty',
        'Diff': 'Difficulty'
    }
    
    new_cols = {}
    for col in df.columns:
        for key, val in col_map.items():
            if key.lower() in col.lower():
                new_cols[col] = val
    df = df.rename(columns=new_cols)
    
    required = ['Day', 'Subject', 'Task', 'Duration', 'Difficulty']
    for req in required:
        if req not in df.columns:
            if req == 'Difficulty':
                df[req] = 'Medium'
            elif req == 'Duration':
                df[req] = 1.0
            else:
                df[req] = 'Unknown'
    return df[required]

def apply_study_approach(df, approach, max_hours_per_day, break_min, day_start_hour=9):
    if df is None or df.empty:
        return df
    
    df = df.copy()
    if 'Day' not in df.columns:
        return df
    
    df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
    df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0]
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0.5)
    df['Diff_Score'] = df['Difficulty'].apply(normalize_difficulty)
    
    if approach == "Waterfall (Cascade)":
        df = df.sort_values(by='Diff_Score', ascending=False)
    elif approach == "Sequential (Focus)":
        df = df.sort_values(by='Subject')
    elif approach == "Random Mix":
        df = df.sample(frac=1)
    elif approach == "Sandwich (Interleaving)":
        hard = df[df['Diff_Score'] >= 2]
        easy = df[df['Diff_Score'] == 1]
        mixed = []
        h, e = hard.to_dict('records'), easy.to_dict('records')
        while h or e:
            if h: mixed.append(h.pop(0))
            if e: mixed.append(e.pop(0))
        df = pd.DataFrame(mixed)
    
    df = df.reset_index(drop=True)
    
    result_rows = []
    current_day = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    current_time = pd.Timedelta(hours=day_start_hour)
    study_used_today = 0.0
    break_delta = pd.Timedelta(minutes=break_min)
    
    for _, row in df.iterrows():
        duration = row['Duration']
        if duration <= 0:
            duration = 0.5
        
        while duration > 0.01:
            if current_time >= pd.Timedelta(hours=24):
                current_day += pd.Timedelta(days=1)
                current_time = pd.Timedelta(hours=day_start_hour)
                study_used_today = 0.0
                continue
            
            if study_used_today >= max_hours_per_day:
                current_day += pd.Timedelta(days=1)
                current_time = pd.Timedelta(hours=day_start_hour)
                study_used_today = 0.0
                continue
            
            remaining_study_today = max_hours_per_day - study_used_today
            hours_now = min(duration, remaining_study_today)
            start_datetime = current_day + current_time
            
            new_row = row.copy()
            new_row['Day'] = start_datetime
            new_row['Duration'] = hours_now
            result_rows.append(new_row)
            
            duration -= hours_now
            study_used_today += hours_now
            current_time += pd.Timedelta(hours=hours_now)
            
            if study_used_today < max_hours_per_day:
                current_time += break_delta
    
    df_result = pd.DataFrame(result_rows)
    if df_result.empty:
        return df
    
    df_result['Day'] = df_result['Day'].dt.strftime("%Y-%m-%d %H:%M:00")
    return df_result.drop(columns=['Diff_Score'])

def parse_duration_to_timedelta(duration):
    duration = str(duration).strip().lower()
    try:
        hours = float(duration)
        return timedelta(hours=hours)
    except ValueError:
        pass
    
    hours = 0
    minutes = 0
    h_match = re.search(r'(\d+(\.\d+)?)\s*h', duration)
    if h_match:
        hours = float(h_match.group(1))
    m_match = re.search(r'(\d+)\s*min', duration)
    if m_match:
        minutes = int(m_match.group(1))
    hours_word_match = re.search(r'(\d+(\.\d+)?)\s*hours?', duration)
    if hours_word_match:
        hours = float(hours_word_match.group(1))
    return timedelta(hours=hours, minutes=minutes)

# -----------------------------------------------------------------------------
# 5. HELPER FUNCTIONS (AI & Parsing)
# -----------------------------------------------------------------------------
def get_ai_response(messages):
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=2500,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Connection Error: {str(e)}"

def parse_markdown_table_to_df(text):
    lines = text.split('\n')
    table_lines = [line for line in lines if '|' in line]
    if len(table_lines) < 3:
        return None
    
    try:
        data = []
        headers = [h.strip() for h in table_lines[0].split('|') if h.strip()]
        for line in table_lines[2:]:
            if '---' in line: continue
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(row) == len(headers):
                data.append(row)
            elif len(row) > len(headers):
                data.append(row[:len(headers)])
            elif len(row) < len(headers):
                row += [''] * (len(headers) - len(row))
                data.append(row)
        
        if not data: return None
        df = pd.DataFrame(data, columns=headers)
        df = standardize_columns(df)
        return df
    except:
        return None

def generate_ics(df):
    cal = Calendar()
    try:
        for _, row in df.iterrows():
            event = Event()
            if 'Day' not in row: continue
            start = pd.to_datetime(row['Day'])
            event.begin = start
            dur_val = row.get('Duration', 1.0)
            delta = parse_duration_to_timedelta(dur_val)
            event.end = start + delta
            subj = row.get('Subject', 'Study')
            task = row.get('Task', 'Session')
            event.name = f"{subj} - {task}"
            event.description = f"Study session: {task} ({dur_val})"
            cal.events.add(event)
    except Exception as e:
        print(f"ICS Gen Error: {e}")
        pass
    return io.BytesIO(str(cal).encode('utf-8'))

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("📚 Threadwise Planner")
    with col2:
        if st.button("Clear"):
            st.session_state.messages = []
            st.session_state.current_plan_df = None
            st.rerun()
    
    st.markdown("### 1. Select Your Strategy")
    approach_options = {
        "Waterfall (Cascade)": "Prioritizes high-energy tasks. Starts with High Difficulty and moves to Low.",
        "Sandwich (Interleaving)": "Prevents burnout. Mixes Hard/Medium tasks with Easy tasks systematically.",
        "Sequential (Focus)": "Deep Work mode. Groups tasks by Subject to minimize context switching.",
        "Random Mix": "Exam Simulation. Randomizes topics to reduce monotony and test recall."
    }
    
    selected_approach = st.selectbox(
        "Choose a study approach:",
        options=list(approach_options.keys()),
        help="Hover over options to see descriptions."
    )
    st.info(f"💡 **Strategy Info:** {approach_options[selected_approach]}")
    
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        max_hours_per_day = st.number_input(
            "Max hours/day:",
            min_value=1.0, max_value=24.0, value=8.0, step=0.5
        )
    with col_conf2:
        break_minutes = st.number_input(
            "Break (mins):",
            min_value=0, max_value=120, value=30, step=5
        )
    
    st.markdown("### 2. Enter Details")
    st.markdown("Tell me about your **Courses**, **Deadlines**, and **Difficulty Levels** (Easy/Medium/Hard).")
    
    system_instruction = (
        "You are Threadwise, an intelligent Study Planning Assistant. "
        "Your goal is to parse messy user inputs into a structured study plan. "
        "\n\n"
        "**DATA EXTRACTION RULES (SMART PARSE):**\n"
        "1. **Free-form Parsing:** Accept ANY input format. Course names can appear with or without numbers, separators, or clear delimiters.\n"
        "2. **Extract Courses from Context:** If you see text like 'Wireless internet 10/01/2026 Multimedia 12/01/2026', parse TWO courses: 'Wireless internet' (deadline 10/01/2026) and 'Multimedia' (deadline 12/01/2026).\n"
        "3. **Connect Disconnected Data:** If courses are listed in one place and difficulties in another, connect them logically by position/context.\n"
        "4. **Required Fields:** EVERY course MUST have: Name, Deadline (YYYY-MM-DD), Difficulty (Easy/Medium/Hard).\n"
        "5. **Infer Intelligently:** If difficulty is mentioned separately (e.g., a list at the end saying 'Hard, Hard, Medium'), map them to courses in order.\n"
        "6. **Output Format:** ONLY return a Markdown Table with columns: Day | Subject | Task | Duration | Difficulty.\n"
        "7. **No Conversational Text:** Do not add explanations, just the table.\n"
        "\n\n"
        "**CRITICAL:** Read the ENTIRE prompt and assemble the table from all available information."
    )
    
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [
            {"role": "system", "content": system_instruction},
            {"role": "assistant", "content": "Hello! Please provide your courses with deadlines and difficulty levels. I can handle any format!"}
        ]
    if "current_plan_df" not in st.session_state:
        st.session_state.current_plan_df = None
    
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if "draft_text" not in st.session_state:
        st.session_state.draft_text = ""
    
    if prompt := st.chat_input("Paste your course list, deadlines, difficulty levels, or raw notes here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        plan_exists = (
            st.session_state.current_plan_df is not None
            and not st.session_state.current_plan_df.empty
        )
        
        accumulated = (st.session_state.draft_text + "\n" + str(prompt)).strip()
        verdict = ai_gatekeeper(
            accumulated_text=accumulated,
            latest_prompt=prompt,
            plan_exists=plan_exists
        )
        
        status = str(verdict.get("status", "incomplete")).lower().strip()
        message = str(verdict.get("message", ""))
        
        if status == "unrelated":
            with st.chat_message("assistant"):
                st.markdown(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
        elif status == "incomplete":
            st.session_state.draft_text = accumulated
            with st.chat_message("assistant"):
                st.markdown(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
        else:
            st.session_state.draft_text = ""
            planner_payload = (
                accumulated
                + "\n\nConstraints:"
                + f" Max hours/day = {max_hours_per_day}."
                + f" Break minutes = {break_minutes}."
                + f" Strategy = {selected_approach}."
                + "\nAllocate the available hours across all courses using deadlines and difficulty."
            )
            st.session_state.messages.append({"role": "user", "content": planner_payload})
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing and restructuring data..."):
                    response_text = get_ai_response(st.session_state.messages)
                    df_parsed = parse_markdown_table_to_df(response_text)
                    
                    if df_parsed is not None and not df_parsed.empty:
                        try:
                            sorted_df = apply_study_approach(
                                df_parsed,
                                selected_approach,
                                max_hours_per_day,
                                break_minutes
                            )
                            st.session_state.current_plan_df = sorted_df
                            st.markdown("✅ **Plan Generated Successfully!**")
                        except Exception as e:
                            st.error(f"Logic Error: {e}")
                    else:
                        st.markdown(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    if st.session_state.current_plan_df is not None and not st.session_state.current_plan_df.empty:
        display_df = st.session_state.current_plan_df.copy()
        if 'Duration' in display_df.columns:
            display_df['Duration'] = display_df['Duration'].apply(format_duration_display)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("📥 Download Your Plan")
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_data = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Download CSV (Excel)",
                data=csv_data,
                file_name="Threadwise_Plan.csv",
                mime="text/csv"
            )
        with col2:
            ics_file = generate_ics(st.session_state.current_plan_df)
            st.download_button(
                label="Download .ICS (Calendar)",
                data=ics_file,
                file_name="Threadwise_Plan.ics",
                mime="text/calendar"
            )

if __name__ == "__main__":
    main()
