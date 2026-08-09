import cv2
import os
import json
import streamlit as st
import pandas as pd
import random
from datetime import datetime

# ==========================================
# Configuration & Paths (Updated Categories)
# ==========================================
CATEGORIES = ["None", "Lips", "Eyes", "Nose", "Eyebrows"]

# Dictionary to map your long folder/file names to the short categories
CATEGORY_MAP = {
    "no_manipulation": "None",
    "lip": "Lips",
    "eye_color": "Eyes",
    "eyes": "Eyes",
    "nose": "Nose",
    "eyebrow": "Eyebrows"
}

JSON_PATH = r"C:\Users\97258\engineering_try_2\Engineering-Project\test_paths.json"
GRADCAM_DIR = "gradcam_results"  
LOG_FILE_PATH = "human_predictions_log.csv"

# ==========================================
# Helper Functions
# ==========================================
def get_short_category(raw_string):
    """Converts long folder or file names into short one-word categories."""
    raw_string_lower = raw_string.lower()
    for key, short_name in CATEGORY_MAP.items():
        if key in raw_string_lower:
            return short_name
    return "None"  # Default fallback

@st.cache_data(show_spinner=False)
def load_game_dataset(json_file_path, gradcam_dir):
    """
    Loads images from JSON and matches them with pre-generated Grad-CAM images
    using the exact same sequential index (img_num).
    """
    if not os.path.exists(json_file_path):
        st.error(f"JSON file not found at: {json_file_path}")
        return []
        
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "path" in item:
                items.append((item["path"], item.get("label", "unknown_label")))
            else:
                items.append((str(item), "unknown_label"))
    elif isinstance(data, dict):
        if "paths" in data and isinstance(data["paths"], list):
            for p in data["paths"]: items.append((p, "unknown_label"))
        else:
            for k, v in data.items():
                if os.path.exists(str(k)) or "/" in str(k) or "\\" in str(k): items.append((str(k), str(v)))
                else: items.append((str(v), str(k)))

    if not os.path.exists(gradcam_dir):
        st.error(f"Gradcam directory not found at: {gradcam_dir}")
        return []
        
    gradcam_files = os.listdir(gradcam_dir)
    dataset = []

    for idx, (img_path, true_label) in enumerate(items):
        img_num = idx + 1
        
        # Get true label in short format
        if true_label == "unknown_label":
            folder_name = os.path.basename(os.path.dirname(img_path))
            true_label_short = get_short_category(folder_name)
        else:
            true_label_short = get_short_category(true_label)
        
        prefix = f"{img_num:03d}_"
        matched_file = None
        model_prediction_short = "None"
        
        for f_name in gradcam_files:
            if f_name.startswith(prefix):
                matched_file = os.path.join(gradcam_dir, f_name)
                
                if "_Pred_" in f_name:
                    raw_pred = os.path.splitext(f_name)[0].split("_Pred_")[-1]
                    model_prediction_short = get_short_category(raw_pred)
                break
        
        if os.path.exists(img_path) and matched_file is not None:
            dataset.append({
                "image_path": img_path,
                "true_label": true_label_short,
                "gradcam_path": matched_file,
                "model_prediction": model_prediction_short
            })
            
    return dataset

def log_human_data(img_path, true_label, human_pred, model_pred):
    new_data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Image_Path": [img_path],
        "True_Label": [true_label],
        "Human_Prediction": [human_pred],
        "Model_Prediction": [model_pred],
        "Human_Correct": [human_pred == true_label],
        "Model_Correct": [model_pred == true_label]
    }
    df = pd.DataFrame(new_data)
    if not os.path.exists(LOG_FILE_PATH):
        df.to_csv(LOG_FILE_PATH, index=False)
    else:
        df.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="Human vs. AI Challenge", layout="centered")

st.title("🧠 Human vs. Machine: Manipulation Challenge")
st.write("Can you spot facial manipulations better than our trained AI model? Let's find out over 5 rounds!")

if 'game_started' not in st.session_state:
    st.session_state.game_started = False
    st.session_state.current_round = 0
    st.session_state.human_score = 0
    st.session_state.model_score = 0
    st.session_state.selected_rounds = []
    st.session_state.round_answered = False

def start_game():
    with st.spinner("Loading dataset..."):
        all_ready_data = load_game_dataset(JSON_PATH, GRADCAM_DIR)
        
    if not all_ready_data:
        st.error(f"Error: No matching pre-saved images found in '{GRADCAM_DIR}'.")
        return
    
    st.session_state.selected_rounds = random.sample(all_ready_data, min(5, len(all_ready_data)))
    st.session_state.game_started = True
    st.session_state.current_round = 0
    st.session_state.human_score = 0
    st.session_state.model_score = 0
    st.session_state.round_answered = False

if st.button("🔄 Start New Game (5 Random Images)"):
    start_game()
    st.rerun()

if not st.session_state.game_started:
    start_game()
    if st.session_state.game_started:
        st.rerun()

# Game Loop
if st.session_state.game_started and st.session_state.current_round < len(st.session_state.selected_rounds):
    current_idx = st.session_state.current_round
    round_data = st.session_state.selected_rounds[current_idx]
    
    st.subheader(f"Round {current_idx + 1} of {len(st.session_state.selected_rounds)}")
    
    img = cv2.imread(round_data["image_path"])
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Examine this image closely:", use_container_width=True)
        
        st.write("**What type of manipulation do you think was done?**")
        human_choice = st.radio("Choose category:", CATEGORIES, key=f"radio_{current_idx}")
        
        if not st.session_state.round_answered:
            if st.button("Submit Answer 🗳️", key=f"submit_{current_idx}"):
                st.session_state.round_answered = True
                
                if human_choice == round_data["true_label"]: st.session_state.human_score += 1
                if round_data["model_prediction"] == round_data["true_label"]: st.session_state.model_score += 1
                
                log_human_data(round_data["image_path"], round_data["true_label"], human_choice, round_data["model_prediction"])
                st.rerun()
        else:
            st.write("---")
            if human_choice == round_data["true_label"]:
                st.success(f"✅ Correct! The true label is: `{round_data['true_label']}`")
            else:
                st.error(f"❌ Incorrect. You chose `{human_choice}`. The correct label is: `{round_data['true_label']}`")
            
            if round_data["model_prediction"] == round_data["true_label"]:
                st.success(f"🤖 AI Model was Correct! It predicted: `{round_data['model_prediction']}`")
            else:
                st.error(f"🤖 AI Model was Incorrect! It predicted: `{round_data['model_prediction']}`")
            
            st.write("**Where the model looked (Grad-CAM heatmap):**")
            cam_img = cv2.imread(round_data["gradcam_path"])
            if cam_img is not None:
                st.image(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if st.button("Next Image ➡️", key=f"next_{current_idx}"):
                st.session_state.current_round += 1
                st.session_state.round_answered = False
                st.rerun()
    else:
        st.warning(f"Failed to load image: {round_data['image_path']}. Skipping...")
        st.session_state.current_round += 1
        st.rerun()

# End Screen
elif st.session_state.game_started and st.session_state.current_round >= len(st.session_state.selected_rounds):
    st.write("---")
    st.header("🏆 Match Finished!")
    h_score = st.session_state.human_score
    m_score = st.session_state.model_score
    total_rounds = len(st.session_state.selected_rounds)
    
    col1, col2 = st.columns(2)
    col1.metric(label="Your Score", value=f"{h_score}/{total_rounds}")
    col2.metric(label="AI Model Score", value=f"{m_score}/{total_rounds}")
    
    if h_score > m_score:
        st.balloons()
        st.success(f"🎉 You beat the AI machine! Outstanding! ({h_score} vs {m_score})")
    elif m_score > h_score:
        st.warning(f"🤖 AI Model wins! The machine outsmarted you this time. ({m_score} vs {h_score})")
    else:
        st.info(f"🤝 It's a Tie! Both scored {h_score} points.")
        
    st.write(f"💾 User insights saved to: `{LOG_FILE_PATH}`")