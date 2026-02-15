import streamlit as st
from PIL import Image
from detector import HumanDetector
# Page Config
st.set_page_config(
    page_title="Digital Forensic AI - Human Detection",
    page_icon="🕵️‍♂️",
    layout="wide"
)
# Initialize Detector (Cache it to avoid reloading model on every run)
@st.cache_resource
def load_detector():
    return HumanDetector(model_path='yolov8n.pt')
detector = load_detector()
# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This is a prototype for a Digital Forensic AI Agent.\n"
        "It uses **YOLOv8** to detect humans in images."
    )
    st.markdown("---")
    st.write("Project by: [Your Name]")
# Main UI
st.title("🕵️‍♂️ Human Detection System")
st.markdown("Upload an image to check for the presence of humans.")
st.markdown("""
---
### 👩‍💻 Project by: **Neha Vaid**  
B.Tech CSE  
Manipal University Jaipur  
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add a button to process
            process_btn = st.button("Detect Humans", type="primary")
            
        except Exception as e:
            st.error(f"Error opening image: {e}")
            process_btn = False
    if process_btn:
        with st.spinner("Analyzing image..."):
            try:
                # Run detection
                processed_image, human_found = detector.detect(image)
                
                with col2:
                    st.subheader("Detection Result")
                    st.image(processed_image, caption="Processed Image", use_container_width=True)
                    
                    if human_found:
                        st.success("### ✅ Human Detected")
                        st.balloons()
                    else:
                        st.warning("### ⚠️ No Human Detected")
                        
            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
st.markdown("---")
st.caption("Powered by YOLOv8 and Streamlit")