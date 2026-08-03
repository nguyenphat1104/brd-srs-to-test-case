import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# Cài đặt giao diện trang Streamlit
st.set_page_config(page_title="SRS to Requirement Model AI", page_icon="📋", layout="wide")

st.title("📋 Chuyển đổi SRS PDF thành Requirement Model")
st.write("Tải lên tài liệu SRS (PDF) của bạn để Gemini phân tích và trích xuất thành Mô hình Yêu cầu có cấu trúc.")

# Nhập Gemini API Key qua thanh sidebar
st.sidebar.header("Cấu hình API")
api_key_input = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

# Hàm trích xuất text từ tệp PDF tải lên
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# Hàm gọi Gemini để phân tích SRS và tạo Requirement Model
def generate_requirement_model(srs_text, api_key):
    genai.configure(api_key=api_key)
    # Sử dụng mô hình Gemini hỗ trợ văn bản nhanh và mạnh
    model = genai.GenerativeModel("gemini-3.5-flash")
    
    prompt = f"""
    Bạn là một kỹ sư hệ thống và chuyên gia phân tích nghiệp vụ cao cấp.
    Dựa vào nội dung tài liệu SRS (Software Requirements Specification) dưới đây, hãy phân tích và tạo ra một Mô hình Yêu cầu (Requirement Model) hoàn chỉnh.
    
    Yêu cầu cấu trúc đầu ra bằng Markdown gồm các phần sau:
    1. **System Overview (Tổng quan hệ thống):** Mục đích và phạm vi.
    2. **Actors (Tác nhân hệ thống):** Danh sách người dùng hoặc hệ thống ngoài tương tác.
    3. **Functional Requirements (Yêu cầu chức năng):** Chia thành các Use Case chính, mỗi Use Case có: ID, Tên, Mô tả, Luồng sự kiện (Chính/Phụ).
    4. **Data / Entity Requirements (Yêu cầu dữ liệu/Thực thể):** Các thực thể chính và thuộc tính cơ bản.
    5. **Non-Functional Requirements (Yêu cầu phi chức năng):** Hiệu năng, bảo mật, khả năng mở rộng.
    
    Nội dung SRS:
    {srs_text}
    """
    
    response = model.generate_content(prompt)
    return response.text

# Khu vực xử lý giao diện chính
uploaded_file = st.file_uploader("Chọn file SRS định dạng PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("Đã tải lên tệp PDF thành công!")
    
    if st.button("Bắt đầu phân tích & Tạo Requirement Model"):
        if not api_key_input:
            st.error("Vui lòng nhập Google Gemini API Key ở thanh bên trái (Sidebar).")
        else:
            with st.spinner("Đang đọc file PDF và yêu cầu Gemini xử lý..."):
                try:
                    # Trích xuất văn bản
                    srs_content = extract_text_from_pdf(uploaded_file)
                    
                    if not srs_content.strip():
                        st.warning("Không thể đọc được nội dung văn bản từ PDF này. File có thể là ảnh quét (scanned).")
                    else:
                        # Gọi Gemini API
                        result_model = generate_requirement_model(srs_content, api_key_input)
                        
                        st.subheader("🎯 Kết quả: Requirement Model")
                        st.markdown(result_model)
                        
                        # Nút tải xuống kết quả dưới dạng file markdown
                        st.download_button(
                            label="Tải xuống Mô hình Yêu cầu (.md)",
                            data=result_model,
                            file_name="Requirement_Model.md",
                            mime="text/markdown"
                        )
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
