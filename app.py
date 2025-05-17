import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

# --- SETUP ---
st.set_page_config(
    page_title="🌱 Plant Doctor",
    page_icon="🌿",
    layout="centered"
)

# --- CLASS NAMES ---
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite ',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy',
    
]

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, len(CLASS_NAMES))
    model_path = 'plant_disease_model.pth'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {os.path.abspath(model_path)}")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
# --- MODEL SANITY CHECK ---
@st.cache_data
def check_model_performance():
    model = load_model()
    
    # Test 1: Verify output shape
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch of 1, 3-channel, 224x224 image
    with torch.no_grad():
        output = model(dummy_input)
    
    # Test 2: Verify prediction distribution
    softmax_output = torch.nn.functional.softmax(output[0], dim=0)
    
    return {
        "output_shape": output.shape,
        "output_range": (output.min().item(), output.max().item()),
        "softmax_sum": softmax_output.sum().item(),  # Should be ~1.0
        "class_with_highest_prob": CLASS_NAMES[torch.argmax(softmax_output).item()],
        "top3_classes": [
            (CLASS_NAMES[i], softmax_output[i].item()) 
            for i in torch.topk(softmax_output, 3).indices
        ]
    }

# Run diagnostics when app starts
if 'diagnostics' not in st.session_state:
    st.session_state.diagnostics = check_model_performance()

# Show diagnostics in an expander
with st.expander("🧪 Model Diagnostics", expanded=False):
    st.json(st.session_state.diagnostics)

# Add to your preprocessing
def debug_preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    debug_img = transform(image)
    st.image(debug_img, caption="After Preprocessing", use_container_width=True)
    return preprocess_image(image)  # Your original function
# --- IMAGE PROCESSING ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def format_name(class_name):
    name = (class_name.replace("___", " - ")
             .replace("__", " - ")
             .replace("_", " ")
             .title())
    return f"🌿 {name.split('-')[0].strip()} (Healthy)" if "healthy" in class_name.lower() else f"🦠 {name}"

# --- STREAMLIT UI ---
st.title("🌿 Plant Doctor")
st.markdown("Upload a clear photo of a plant leaf for disease diagnosis")

with st.expander("ℹ️ Supported Plants & Diseases"):
    st.markdown("""
    **Pepper**  
    - Bacterial spot  
    - Healthy plants  
    
    **Potato**  
    - Early blight  
    - Late blight  
    - Healthy plants  
    
    **Tomato**  
    - 10 different diseases  
    - Healthy plants
    """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)  # Updated parameter here
    
    if st.button('🔍 Diagnose', type="primary"):
        with st.spinner('Analyzing leaf health...'):
            model = load_model()
            tensor = preprocess_image(img)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                top_p, top_c = torch.topk(probs, 3)
            
            with col2:
                main_pred = CLASS_NAMES[top_c[0]]
                conf = top_p[0].item() * 100
                
                if "healthy" in main_pred.lower():
                    st.success(f"""
                    **{format_name(main_pred)}**  
                    **Confidence:** {conf:.1f}%  
                    **Status:** ✅ Healthy
                    """)
                    st.balloons()
                else:
                    st.error(f"""
                    **{format_name(main_pred)}**  
                    **Confidence:** {conf:.1f}%  
                    **Status:** ❗ Disease Detected
                    """)
                    st.warning("""
                    **Recommended Actions:**  
                    - Isolate affected plants  
                    - Remove infected leaves  
                    - Apply appropriate treatment
                    """)
                
                if top_p[1] > 0.15:
                    st.divider()
                    st.write("Other possible diagnoses:")
                    for i in range(1, 3):
                        st.write(
                            f"{i}. {format_name(CLASS_NAMES[top_c[i]])} "
                            f"({top_p[i].item()*100:.1f}%)"
                        )

st.markdown("---")
st.caption("Plant Doctor v1.1 | Detects 12 diseases across 3 plant species")
