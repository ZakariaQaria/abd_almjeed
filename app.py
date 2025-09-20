import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import random

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="مختبر معالجة الصور المتقدم",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إضافة CSS مخصص لتطبيق التصميم
def inject_custom_css():
    st.markdown("""
    <style>
    :root {
        --primary: #2c3e50;
        --secondary: #8e44ad;
        --accent: #3498db;
        --dark-bg: #1e1e1e;
        --code-bg: #2d2d2d;
        --light: #ecf0f1;
        --success: #27ae60;
        --warning: #f39c12;
        --danger: #e74c3c;
    }

    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
        color: var(--light) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    .main .block-container {
        padding: 2rem 1rem !important;
        max-width: 1400px !important;
    }

    .custom-header {
        background: linear-gradient(to right, #2c3e50, #8e44ad) !important;
        color: white !important;
        padding: 1.5rem 2rem !important;
        text-align: center !important;
        border-bottom: 5px solid #3498db !important;
        margin: -2rem -1rem 2rem -1rem !important;
    }

    .section {
        background: rgba(45, 45, 45, 0.8) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px) !important;
    }

    h1, h2, h3 {
        color: #3498db !important;
        margin-bottom: 1rem !important;
    }

    .stButton>button {
        background: linear-gradient(to right, #3498db, #8e44ad) !important;
        color: white !important;
        border: none !important;
        padding: 12px 25px !important;
        border-radius: 50px !important;
        font-weight: bold !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4) !important;
    }

    .stSlider>div>div>div {
        background: #3498db !important;
    }

    .upload-section {
        border: 2px dashed #3498db !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        text-align: center !important;
        margin: 1rem 0 !important;
        background: rgba(52, 152, 219, 0.1) !important;
        transition: all 0.3s ease !important;
    }

    .upload-section:hover {
        background: rgba(52, 152, 219, 0.2) !important;
    }

    .code-editor {
        background: #2d2d2d !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3) !important;
    }

    .editor-header {
        background: rgba(0, 0, 0, 0.4) !important;
        padding: 10px 15px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .editor-dot {
        width: 12px !important;
        height: 12px !important;
        border-radius: 50% !important;
    }

    .red { background: #ff5f56 !important; }
    .yellow { background: #ffbd2e !important; }
    .green { background: #27ca3f !important; }

    .editor-title {
        font-size: 0.9rem !important;
        color: #aaa !important;
        font-family: 'Courier New', monospace !important;
    }

    pre {
        background: #1e1e1e !important;
        padding: 1.5rem !important;
        border-radius: 0 0 10px 10px !important;
        margin: 0 !important;
    }

    code {
        color: #d4d4d4 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem !important;
    }

    .image-comparison {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 20px !important;
        margin: 2rem 0 !important;
    }

    .image-box {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }

    .image-box:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
    }

    .image-label {
        display: block !important;
        margin-top: 10px !important;
        font-weight: bold !important;
        color: #3498db !important;
        font-size: 1.1rem !important;
    }

    .nav-buttons {
        display: flex !important;
        justify-content: center !important;
        gap: 15px !important;
        margin: 1.5rem 0 !important;
        flex-wrap: wrap !important;
    }

    .nav-btn {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border: none !important;
        padding: 12px 20px !important;
        border-radius: 50px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-family: 'Segoe UI', sans-serif !important;
        font-weight: bold !important;
        min-width: 120px !important;
        text-align: center !important;
    }

    .nav-btn:hover {
        background: rgba(52, 152, 219, 0.4) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3) !important;
    }

    footer {
        text-align: center !important;
        padding: 2rem !important;
        background: rgba(0, 0, 0, 0.4) !important;
        color: #aaa !important;
        margin-top: 2rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 0 0 15px 15px !important;
    }

    .stMarkdown {
        color: #ecf0f1 !important;
    }

    .stSlider label {
        color: #ecf0f1 !important;
        font-weight: bold !important;
    }

    .stAlert {
        background: rgba(231, 76, 60, 0.2) !important;
        border: 1px solid #e74c3c !important;
        border-radius: 10px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: rgba(30, 30, 30, 0.7) !important;
        padding: 10px !important;
        border-radius: 15px !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        white-space: pre !important;
        background-color: rgba(45, 45, 45, 0.8) !important;
        border-radius: 10px !important;
        gap: 10px !important;
        padding: 10px 20px !important;
        color: #ecf0f1 !important;
        font-weight: bold !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# دالة لعرض محرر الأكواد
def display_code_editor(code, language="python", filename="code.py"):
    st.markdown(f"""
    <div class="code-editor">
        <div class="editor-header">
            <div class="editor-dot red"></div>
            <div class="editor-dot yellow"></div>
            <div class="editor-dot green"></div>
            <div class="editor-title">{filename}</div>
        </div>
        
    </div>

    """, unsafe_allow_html=True)
    st.code(code,language=language)
# ==================== دوال معالجة الصور ====================

def adjust_brightness_contrast(image, brightness=0, contrast=100):
    brightness = 0 if brightness is None else brightness
    contrast = 100 if contrast is None else contrast
    
    contrast = float(contrast + 100) / 100.0
    contrast = contrast ** 2
    
    adjusted_image = cv2.addWeighted(image, contrast, image, 0, brightness - 100)
    return adjusted_image

def convert_color_space(image, conversion_code):
    return cv2.cvtColor(image, conversion_code)

def apply_threshold(image, threshold_type, threshold_value=127):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if threshold_type == "THRESH_BINARY":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    elif threshold_type == "THRESH_BINARY_INV":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    elif threshold_type == "THRESH_TRUNC":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TRUNC)
    elif threshold_type == "THRESH_TOZERO":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TOZERO)
    elif threshold_type == "THRESH_TOZERO_INV":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TOZERO_INV)
    elif threshold_type == "THRESH_OTSU":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        result = gray
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

def apply_filter(image, filter_type, kernel_size=3):
    if filter_type == "Blur":
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "Median Blur":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "Sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Emboss":
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return image

def add_noise(image, noise_type):
    if noise_type == "Gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 50
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "Salt & Pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
        return noisy
    return image

def remove_noise(image, filter_type):
    if filter_type == "Median":
        return cv2.medianBlur(image, 5)
    elif filter_type == "Gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "Bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image

def detect_edges(image, method, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "Canny":
        edges = cv2.Canny(gray, threshold1, threshold2)
    elif method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / np.max(edges) * 255)
    elif method == "Laplacian":
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    else:
        edges = gray
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_morphological_operation(image, operation, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    if operation == "Erosion":
        result = cv2.erode(binary, kernel, iterations=1)
    elif operation == "Dilation":
        result = cv2.dilate(binary, kernel, iterations=1)
    elif operation == "Opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        result = binary
        
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

def apply_geometric_transform(image, transform, angle=0, scale=1.0):
    h, w = image.shape[:2]
    
    if transform == "Rotation":
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(image, matrix, (w, h))
    elif transform == "Scaling":
        result = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    elif transform == "Translation":
        matrix = np.float32([[1, 0, 50], [0, 1, 50]])
        result = cv2.warpAffine(image, matrix, (w, h))
    elif transform == "Flipping":
        result = cv2.flip(image, 1)  # 0: vertical, 1: horizontal
    else:
        result = image
        
    return result

# ==================== دوال المحاضرات ====================

def lecture_1():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    الصورة الرقمية هي تمثيل رقمي للصورة الطبيعية تتكون من مصفوفة من البكسلات. كل بكسل يحمل قيمة رقمية تمثل شدة الإضاءة واللون في تلك النقطة. 
    تعتمد جودة الصورة على ثلاثة عوامل رئيسية: الأبعاد (الطول والعرض)، عدد القنوات اللونية، وعمق البت الذي يحدد عدد الألوان المحتملة لكل بكسل.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# تحميل الصورة وعرض معلوماتها
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# عرض معلومات الصورة
print("أبعاد الصورة:", image.shape)
print("عدد القنوات:", image.shape[2] if len(image.shape) > 2 else 1)
print("نوع البيانات:", image.dtype)
print("القيم الدنيا والعليا:", np.min(image), np.max(image))

# حفظ الصورة
cv2.imwrite('processed_image.jpg', image)
'''
    display_code_editor(code, filename="image_info.py")

def lecture_2():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    أنظمة الألوان المختلفة تستخدم لأغراض متعددة في معالجة الصور. نظام RGB هو النظام الأساسي المستخدم في العرض، 
    بينما نظام HSV مفيد لفصل الإضاءة عن اللون. نظام Grayscale يبسط الصورة لتحليل الشدة فقط.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# التحويل بين أنظمة الألوان
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# التحويل إلى Grayscale
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# التحويل إلى HSV
hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# تقسيم القنوات في نظام RGB
red_channel = image_rgb[:, :, 0]
green_channel = image_rgb[:, :, 1]
blue_channel = image_rgb[:, :, 2]

# تقسيم القنوات في نظام HSV
hue_channel = hsv[:, :, 0]
saturation_channel = hsv[:, :, 1]
value_channel = hsv[:, :, 2]
'''
    display_code_editor(code, filename="color_spaces.py")

def lecture_3():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    تعديل السطوع والتباين من العمليات الأساسية في معالجة الصور. يتم تعديل السطوع بإضافة قيمة ثابتة 
    إلى جميع وحدات البكسل في الصورة، بينما يتم تعديل التباين بضرب قيم البكسل في معامل ثابت. هذه 
    العمليات تساعد في تحسين جودة الصورة وإبراز التفاصيل المخفية.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# تعديل السطوع والتباين والعتبة
import cv2
import numpy as np

def adjust_brightness_contrast(image, brightness=0, contrast=100):
    # ضبط القيم الافتراضية
    brightness = 0 if brightness is None else brightness
    contrast = 100 if contrast is None else contrast
    
    # حساب معامل التباين
    contrast_factor = float(contrast + 100) / 100.0
    contrast_factor = contrast_factor ** 2
    
    # تطبيق المعادلة: output = input * contrast + brightness
    adjusted_image = cv2.addWeighted(
        image, contrast_factor, 
        image, 0, 
        brightness - 100
    )
    
    return adjusted_image

def apply_threshold(image, threshold_value=127):
    # تحويل إلى تدرج الرمادي
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # تطبيق العتبة
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)

def apply_negative(image):
    # تطبيق الصورة السالبة
    return 255 - image

# استخدام الدوال
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تطبيق التعديلات
bright_contrast = adjust_brightness_contrast(image_rgb, brightness=30, contrast=120)
thresholded = apply_threshold(image_rgb, threshold_value=150)
negative = apply_negative(image_rgb)
'''
    display_code_editor(code, filename="point_operations.py")

def lecture_4():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    المرشحات والالتفاف من العمليات الأساسية في معالجة الصور. تعتمد هذه العمليات على نواة (Kernel) 
    يتم تطبيقها على الصورة لإجراء عمليات مثل التنعيم، الحدة، الكشف عن الحواف، وغيرها من العمليات.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# تطبيق المرشحات والالتفاف
import cv2
import numpy as np

def apply_filter(image, filter_type, kernel_size=3):
    if filter_type == "Blur":
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "Median Blur":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "Sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Emboss":
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return image

# إنشاء نواة مخصصة
def create_custom_kernel():
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return kernel

# تحميل الصورة وتطبيق المرشحات
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تطبيق مرشحات مختلفة
blurred = apply_filter(image_rgb, "Gaussian Blur", 5)
sharpened = apply_filter(image_rgb, "Sharpen")
edges = apply_filter(image_rgb, "Edge Detection")

# تطبيق نواة مخصصة
custom_kernel = create_custom_kernel()
custom_filtered = cv2.filter2D(image_rgb, -1, custom_kernel)
'''
    display_code_editor(code, filename="filters.py")

def lecture_5():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    الضوضاء في الصور هي تشوهات غير مرغوب فيها يمكن أن تنتج عن ظروف التصوير المختلفة. 
    توجد أنواع متعددة من الضوضاء مثل الضوضاء الغوسية وضوضاء الملح والفلفل. 
    تهدف عمليات إزالة الضوضاء إلى تحسين جودة الصورة مع الحفاظ على التفاصيل المهمة.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# إضافة وإزالة الضوضاء
import cv2
import numpy as np

def add_noise(image, noise_type):
    if noise_type == "Gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 50
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "Salt & Pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
        return noisy
    return image

def remove_noise(image, filter_type):
    if filter_type == "Median":
        return cv2.medianBlur(image, 5)
    elif filter_type == "Gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "Bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image

# تحميل الصورة ومعالجتها
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# إضافة ضوضاء
noisy_image = add_noise(image_rgb, "Salt & Pepper")

# إزالة الضوضاء
denoised_median = remove_noise(noisy_image, "Median")
denoised_bilateral = remove_noise(noisy_image, "Bilateral")
'''
    display_code_editor(code, filename="denoising.py")

def lecture_6():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    كشف الحواف هو عملية أساسية في معالجة الصور تهدف إلى تحديد المناطق التي تتغير فيها شدة 
    الصورة بشكل مفاجئ. هذه الحواف تمثل عادة حدودًا بين مناطق مختلفة في الصورة ويمكن 
    استخدامها في تطبيقات مثل التجزئة والتعرف على الأشياء.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# كشف الحواف في الصور
import cv2
import numpy as np

def detect_edges(image, method, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if method == "Canny":
        edges = cv2.Canny(gray, threshold1, threshold2)
    elif method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / np.max(edges) * 255)
    elif method == "Laplacian":
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    elif method == "Prewitt":
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)
        edges = np.sqrt(prewittx**2 + prewitty**2)
        edges = np.uint8(edges / np.max(edges) * 255)
    else:
        edges = gray
        
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# تحميل الصورة
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# كشف الحواف بطرق مختلفة
canny_edges = detect_edges(image_rgb, "Canny", 100, 200)
sobel_edges = detect_edges(image_rgb, "Sobel")
laplacian_edges = detect_edges(image_rgb, "Laplacian")

# تطبيق Gaussian Blur قبل كشف الحواف للتقليل من الضوضاء
blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)
smoothed_edges = detect_edges(blurred, "Canny", 50, 150)
'''
    display_code_editor(code, filename="edge_detection.py")

def lecture_7():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    العمليات المورفولوجية هي تقنيات لمعالجة الصور الثنائية تعتمد على شكل وهيكل الأشياء في الصورة. 
    هذه العمليات مفيدة في تنظيف الصور الثنائية، وإزالة الضوضاء، وعزل الأشياء، وتحسين الملامح.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# العمليات المورفولوجية
import cv2
import numpy as np

def apply_morphological_operation(image, operation, kernel_size=3, iterations=1):
    # تحويل الصورة إلى تدرج الرمادي ثم إلى ثنائية
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # إنشاء النواة
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # تطبيق العملية المورفولوجية
    if operation == "Erosion":
        result = cv2.erode(binary, kernel, iterations=iterations)
    elif operation == "Dilation":
        result = cv2.dilate(binary, kernel, iterations=iterations)
    elif operation == "Opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "Closing":
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "Gradient":
        result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    elif operation == "Top Hat":
        result = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
    elif operation == "Black Hat":
        result = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
    else:
        result = binary
        
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

# إنشاء نواة مخصصة
def create_custom_kernel(shape="rect", size=3):
    if shape == "rect":
        return np.ones((size, size), np.uint8)
    elif shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        return np.ones((size, size), np.uint8)

# تحميل الصورة
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تطبيق عمليات مورفولوجية مختلفة
eroded = apply_morphological_operation(image_rgb, "Erosion", 3, 1)
dilated = apply_morphological_operation(image_rgb, "Dilation", 3, 1)
opened = apply_morphological_operation(image_rgb, "Opening", 5, 1)
closed = apply_morphological_operation(image_rgb, "Closing", 5, 1)

# استخدام نواة مخصصة
custom_kernel = create_custom_kernel("ellipse", 5)
custom_morph = cv2.morphologyEx(
    cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 
    cv2.MORPH_OPEN, 
    custom_kernel
)
custom_morph = cv2.cvtColor(custom_morph, cv2.COLOR_GRAY2RGB)
'''
    display_code_editor(code, filename="morphological.py")

def lecture_8():
    st.markdown("### 📚 الشرح النظري")
    st.markdown("""
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    التحويلات الهندسية هي عمليات تغير الهندسة المكانية للصورة. تشمل هذه العمليات التدوير، القياس، 
    الانعكاس، القص، والانزياح. هذه التحويلات مفيدة في تصحيح التشوهات، ومطابقة الصور، وتطبيقات الرؤية الحاسوبية.
    </p>
    """, unsafe_allow_html=True)
    
    code = '''
# التحويلات الهندسية
import cv2
import numpy as np

def apply_geometric_transform(image, transform, **kwargs):
    h, w = image.shape[:2]
    
    if transform == "Rotation":
        angle = kwargs.get('angle', 0)
        scale = kwargs.get('scale', 1.0)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(image, matrix, (w, h))
        
    elif transform == "Scaling":
        scale_x = kwargs.get('scale_x', 1.0)
        scale_y = kwargs.get('scale_y', 1.0)
        result = cv2.resize(image, None, fx=scale_x, fy=scale_y, 
                           interpolation=cv2.INTER_LINEAR)
        
    elif transform == "Translation":
        tx = kwargs.get('tx', 50)
        ty = kwargs.get('ty', 50)
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(image, matrix, (w, h))
        
    elif transform == "Flipping":
        flip_code = kwargs.get('flip_code', 1)  # 0: vertical, 1: horizontal, -1: both
        result = cv2.flip(image, flip_code)
        
    elif transform == "Cropping":
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        width = kwargs.get('width', w // 2)
        height = kwargs.get('height', h // 2)
        result = image[y:y+height, x:x+width]
        
    elif transform == "Affine":
        pts1 = np.float32([[50,50], [200,50], [50,200]])
        pts2 = np.float32([[10,100], [200,50], [100,250]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(image, matrix, (w, h))
        
    elif transform == "Perspective":
        pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
        pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(image, matrix, (300,300))
        
    else:
        result = image
        
    return result

# تحميل الصورة
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تطبيق تحويلات هندسية مختلفة
rotated = apply_geometric_transform(image_rgb, "Rotation", angle=45, scale=1.0)
scaled = apply_geometric_transform(image_rgb, "Scaling", scale_x=1.5, scale_y=1.5)
translated = apply_geometric_transform(image_rgb, "Translation", tx=100, ty=50)
flipped = apply_geometric_transform(image_rgb, "Flipping", flip_code=1)
cropped = apply_geometric_transform(image_rgb, "Cropping", x=100, y=100, width=200, height=200)

# تحويلات متقدمة
affine = apply_geometric_transform(image_rgb, "Affine")
perspective = apply_geometric_transform(image_rgb, "Perspective")
'''
    display_code_editor(code, filename="geometric_transforms.py")

def main():
    # حقن CSS المخصص
    inject_custom_css()

    # الرأس الرئيسي
    st.markdown("""
    <div class="custom-header">
        <h1>🎨 مختبر معالجة الصور المتقدم</h1>
        <h3>مشروع محاضرات معالجة الصور الرقمية</h3>
    </div>
    """, unsafe_allow_html=True)

    # إنشاء قائمة المحاضرات
    lectures = [
        "المحاضرة 1: مدخل ومعايرة الصور",
        "المحاضرة 2: أنظمة الألوان",
        "المحاضرة 3: تعديل السطوع والتباين",
        "المحاضرة 4: المرشحات والالتفاف",
        "المحاضرة 5: إزالة الضوضاء",
        "المحاضرة 6: كشف الحواف",
        "المحاضرة 7: العمليات المورفولوجية",
        "المحاضرة 8: التحويلات الهندسية",
        "المشروع النهائي"
    ]

    # إنشاء تبويبات للمحاضرات
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(lectures)

    with tab1:
        st.markdown("## المحاضرة 1: مدخل ومعايرة الصور الرقمية")
        lecture_1()
        
        # التطبيق العملي للمحاضرة 1
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture1")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
            
            with col2:
                st.info(f"**معلومات الصورة:**")
                st.write(f"**الأبعاد:** {image.shape[1]} x {image.shape[0]}")
                st.write(f"**عدد القنوات:** {image.shape[2]}")
                st.write(f"**نوع البيانات:** {image.dtype}")
                st.write(f"**مدى القيم:** {np.min(image)} إلى {np.max(image)}")

    with tab2:
        st.markdown("## المحاضرة 2: أنظمة الألوان (Color Spaces)")
        lecture_2()
        
        # التطبيق العملي للمحاضرة 2
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture2")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            color_space = st.selectbox("اختر نظام الألوان للتحويل:", 
                                     ["RGB", "GRAY", "HSV", "LAB", "YUV"])
            
            if color_space == "GRAY":
                converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2RGB)
            elif color_space == "HSV":
                converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == "LAB":
                converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            elif color_space == "YUV":
                converted = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            else:
                converted = image
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="🖼️ الصورة الأصلية (RGB)", use_container_width=True)
            with col2:
                st.image(converted, caption=f"🔄 الصورة المحولة ({color_space})", use_container_width=True)

    with tab3:
        st.markdown("## المحاضرة 3: تعديل السطوع والتباين")
        lecture_3()
        
        # التطبيق العملي للمحاضرة 3
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture3")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
            
            with col2:
                brightness = st.slider("🔆 السطوع", -100, 100, 0, key="brightness_slider")
                contrast = st.slider("🌈 التباين", 0, 200, 100, key="contrast_slider")
                threshold = st.slider("⚫️ العتبة", 0, 255, 127, key="threshold_slider")
                
                if st.button("🚀 تطبيق التعديلات", key="apply_btn"):
                    adjusted = adjust_brightness_contrast(image, brightness, contrast)
                    thresholded = apply_threshold(image, "THRESH_BINARY", threshold)
                    negative = 255 - image
                    
                    tab_a, tab_b, tab_c = st.tabs(["السطوع والتباين", "العتبة", "الصورة السالبة"])
                    
                    with tab_a:
                        st.image(adjusted, caption="🔄 السطوع والتباين", use_container_width=True)
                    with tab_b:
                        st.image(thresholded, caption="⚫️ العتبة", use_container_width=True)
                    with tab_c:
                        st.image(negative, caption="🌗 الصورة السالبة", use_container_width=True)

    with tab4:
        st.markdown("## المحاضرة 4: المرشحات والالتفاف")
        lecture_4()
        
        # التطبيق العملي للمحاضرة 4
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture4")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            filter_type = st.selectbox("اختر نوع المرشح:", 
                                     ["Blur", "Gaussian Blur", "Median Blur", "Sharpen", "Emboss", "Edge Detection"])
            
            kernel_size = st.slider("حجم النواة", 3, 15, 5, 2, key="kernel_size")
            
            if st.button("🚀 تطبيق المرشح", key="apply_filter_btn"):
                filtered = apply_filter(image, filter_type, kernel_size)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
                with col2:
                    st.image(filtered, caption=f"✨ الصورة بعد {filter_type}", use_container_width=True)

    with tab5:
        st.markdown("## المحاضرة 5: إزالة الضوضاء")
        lecture_5()
        
        # التطبيق العملي للمحاضرة 5
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture5")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            noise_type = st.selectbox("اختر نوع الضوضاء:", ["Gaussian", "Salt & Pepper"])
            denoise_type = st.selectbox("اختر طريقة إزالة الضوضاء:", ["Median", "Gaussian", "Bilateral"])
            
            if st.button("🚀 تطبيق المعالجة", key="apply_denoise_btn"):
                # إضافة الضوضاء
                noisy = add_noise(image, noise_type)
                
                # إزالة الضوضاء
                denoised = remove_noise(noisy, denoise_type)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
                with col2:
                    st.image(noisy, caption=f"🔊 الصورة مع {noise_type} noise", use_container_width=True)
                with col3:
                    st.image(denoised, caption=f"🔇 الصورة بعد {denoise_type} filter", use_container_width=True)

    with tab6:
        st.markdown("## المحاضرة 6: كشف الحواف")
        lecture_6()
        
        # التطبيق العملي للمحاضرة 6
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture6")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            edge_method = st.selectbox("اختر طريقة كشف الحواف:", ["Canny", "Sobel", "Laplacian", "Prewitt"])
            
            if edge_method == "Canny":
                threshold1 = st.slider("العتبة الدنيا", 0, 255, 100, key="threshold1")
                threshold2 = st.slider("العتبة العليا", 0, 255, 200, key="threshold2")
            else:
                threshold1, threshold2 = 100, 200
            
            if st.button("🚀 كشف الحواف", key="detect_edges_btn"):
                edges = detect_edges(image, edge_method, threshold1, threshold2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
                with col2:
                    st.image(edges, caption=f"🔍 الحواف بـ {edge_method}", use_container_width=True)

    with tab7:
        st.markdown("## المحاضرة 7: العمليات المورفولوجية")
        lecture_7()
        
        # التطبيق العملي للمحاضرة 7
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture7")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            morph_operation = st.selectbox("اختر العملية المورفولوجية:", 
                                         ["Erosion", "Dilation", "Opening", "Closing", "Gradient"])
            
            kernel_size = st.slider("حجم النواة", 3, 15, 5, 2, key="morph_kernel_size")
            iterations = st.slider("عدد التكرارات", 1, 10, 1, key="morph_iterations")
            
            if st.button("🚀 تطبيق العملية", key="apply_morph_btn"):
                result = apply_morphological_operation(image, morph_operation, kernel_size, iterations)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
                with col2:
                    st.image(result, caption=f"✨ بعد {morph_operation}", use_container_width=True)

    with tab8:
        st.markdown("## المحاضرة 8: التحويلات الهندسية")
        lecture_8()
        
        # التطبيق العملي للمحاضرة 8
        st.markdown("### 🧪 التجربة العملية")
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="lecture8")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            transform_type = st.selectbox("اختر نوع التحويل:", 
                                        ["Rotation", "Scaling", "Translation", "Flipping", "Cropping"])
            
            if transform_type == "Rotation":
                angle = st.slider("زاوية الدوران", -180, 180, 45, key="rotation_angle")
                scale = st.slider("مقياس التكبير", 0.1, 3.0, 1.0, 0.1, key="rotation_scale")
                result = apply_geometric_transform(image, transform_type, angle=angle, scale=scale)
                
            elif transform_type == "Scaling":
                scale_x = st.slider("مقياس التكبير الأفقي", 0.1, 3.0, 1.5, 0.1, key="scale_x")
                scale_y = st.slider("مقياس التكبير العمودي", 0.1, 3.0, 1.5, 0.1, key="scale_y")
                result = apply_geometric_transform(image, transform_type, scale_x=scale_x, scale_y=scale_y)
                
            elif transform_type == "Translation":
                tx = st.slider("الإزاحة الأفقية", -200, 200, 100, key="translation_x")
                ty = st.slider("الإزاحة العمودية", -200, 200, 50, key="translation_y")
                result = apply_geometric_transform(image, transform_type, tx=tx, ty=ty)
                
            elif transform_type == "Flipping":
                flip_code = st.selectbox("اختر اتجاه الانعكاس:", 
                                       [("افقي", 1), ("عمودي", 0), ("كلاهما", -1)], 
                                       format_func=lambda x: x[0])
                result = apply_geometric_transform(image, transform_type, flip_code=flip_code[1])
                
            elif transform_type == "Cropping":
                x = st.slider("النقطة X", 0, image.shape[1]-100, 100, key="crop_x")
                y = st.slider("النقطة Y", 0, image.shape[0]-100, 100, key="crop_y")
                width = st.slider("العرض", 100, image.shape[1]-x, 200, key="crop_width")
                height = st.slider("الارتفاع", 100, image.shape[0]-y, 200, key="crop_height")
                result = apply_geometric_transform(image, transform_type, x=x, y=y, width=width, height=height)
            
            if st.button("🚀 تطبيق التحويل", key="apply_transform_btn"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
                with col2:
                    st.image(result, caption=f"✨ بعد {transform_type}", use_container_width=True)

    with tab9:
        st.markdown("## المشروع النهائي: pipeline معالجة الصور")
        st.markdown("### 🧪 أنشئ pipeline معالجة الصور الخاص بك")
        
        uploaded_file = st.file_uploader("اختر صورة لمعالجتها", type=["jpg", "jpeg", "png"], key="final_project")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
            
            # إعداد pipeline المعالجة
            st.markdown("### ⚙️ خطوات المعالجة:")
            steps = st.multiselect(
                "اختر خطوات المعالجة:",
                ["تحويل إلى رمادي", "ضبط السطوع والتباين", "تطبيق مرشح", "كشف الحواف", "عتبة", "عمليات مورفولوجية"],
                default=["تحويل إلى رمادي", "كشف الحواف"]
            )
            
            processed_image = image.copy()
            process_history = []
            
            for step in steps:
                if step == "تحويل إلى رمادي":
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                    process_history.append("تحويل إلى تدرج الرمادي")
                
                elif step == "ضبط السطوع والتباين":
                    processed_image = adjust_brightness_contrast(processed_image, 20, 120)
                    process_history.append("ضبط السطوع (+20) والتباين (+20%)")
                
                elif step == "تطبيق مرشح":
                    processed_image = apply_filter(processed_image, "Gaussian Blur", 5)
                    process_history.append("تطبيق مرشح Gaussian Blur (5x5)")
                
                elif step == "كشف الحواف":
                    processed_image = detect_edges(processed_image, "Canny", 100, 200)
                    process_history.append("كشف الحواف بـ Canny")
                
                elif step == "عتبة":
                    processed_image = apply_threshold(processed_image, "THRESH_BINARY", 127)
                    process_history.append("تطبيق العتبة الثنائية (127)")
                
                elif step == "عمليات مورفولوجية":
                    processed_image = apply_morphological_operation(processed_image, "Closing", 3)
                    process_history.append("عملية Closing مورفولوجية (3x3)")
            
            if st.button("▶️ تشغيل Pipeline", key="run_pipeline"):
                st.markdown("### 📊 نتائج المعالجة:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
                with col2:
                    st.image(processed_image, caption="✨ الصورة النهائية", use_container_width=True)
                
                st.markdown("### 📝 سجل المعالجة:")
                for i, step in enumerate(process_history, 1):
                    st.write(f"{i}. {step}")
                
                # إمكانية حفظ الصورة الناتجة
                buf = io.BytesIO()
                processed_pil = Image.fromarray(processed_image)
                processed_pil.save(buf, format="JPEG", quality=95)
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 تحميل الصورة النهائية",
                    data=byte_im,
                    file_name="processed_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

    # التذييل
    st.markdown("""
    <footer>
        <p>تم التصميم والتطوير بواسطة المهندس زكريا قارية</p>
        <p>مشروع معالجة الصور الرقمية - كلية الهندسة © 2023</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()