import io
import pandas as pd
from google import genai
import pypdf
import streamlit as st

st.set_page_config(page_title="SRS to Requirement Model", layout="wide")

st.title("Chuyển đổi SRS PDF sang Requirement Model (Excel)")
st.write(
    "Tải lên tài liệu SRS định dạng PDF. Ứng dụng sẽ dùng Gemini để trích xuất"
    " các yêu cầu thành mô hình bảng và cho phép bạn tải file Excel."
)

# Nhập API Key của Gemini
api_key = st.text_input("Nhập Google Gemini API Key của bạn:", type="password")

# Tải file PDF
uploaded_file = st.file_uploader("Chọn file SRS (PDF)", type=["pdf"])


def extract_text_from_pdf(pdf_file):
  reader = pypdf.PdfReader(pdf_file)
  text = ""
  for page in reader.pages:
    text += page.extract_text() or ""
  return text


if uploaded_file and api_key:
  if st.button("Phân tích và Tạo Model"):
    with st.spinner(
        "Đang đọc PDF và gọi Gemini xử lý, vui lòng chờ trong giây lát..."
    ):
      try:
        # Đọc nội dung PDF
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Khởi tạo client Gemini mới
        client = genai.Client(api_key=api_key)

        # Prompt yêu cầu phân tích yêu cầu phần mềm
        prompt = f"""
                Bạn là một chuyên gia Phân tích Nghiệp vụ (Business Analyst). 
                Dựa vào nội dung tài liệu SRS dưới đây, hãy trích xuất các yêu cầu thành một bảng gồm các cột sau:
                1. Req_ID (Mã yêu cầu, ví dụ: REQ-01)
                2. Requirement_Name (Tên yêu cầu ngắn gọn)
                3. Type (Loại yêu cầu: Functional / Non-Functional / Business)
                4. Description (Mô tả chi tiết yêu cầu)
                5. Priority (Mức độ ưu tiên: High / Medium / Low)
                6. Source (Nguồn gốc hoặc phần trong tài liệu)

                Hãy trả về kết quả thuần túy dưới dạng bảng Markdown (có dấu | và -) để có thể phân tích cú pháp dễ dàng. Không kèm theo lời mở đầu hay kết luận rườm rà.

                Nội dung SRS:
                {pdf_text[:15000]} 
                """

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

        result_text = response.text

        # Chuyển bảng Markdown thành Pandas DataFrame
        # Tìm các dòng chứa bảng markdown
        lines = [
            line.strip()
            for line in result_text.split("\n")
            if line.strip().startswith("|")
        ]

        if len(lines) > 2:
          # Lọc bỏ dòng phân cách (ví dụ |---|---|)
          table_lines = [
              line
              for line in lines
              if not all(c in "-|: " for c in line)
          ]

          # Tách dữ liệu các cột
          data = []
          header = [col.strip() for col in table_lines[0].split("|")[1:-1]]
          for line in table_lines[1:]:
            row = [col.strip() for col in line.split("|")[1:-1]]
            if len(row) == len(header):
              data.append(row)

          df = pd.DataFrame(data, columns=header)

          st.success("Trích xuất yêu cầu thành công!")
          st.dataframe(df)

          # Xuất ra Excel
          output = io.BytesIO()
          with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Requirement Model")
          excel_data = output.getvalue()

          st.download_button(
              label="Tải xuống file Excel",
              data=excel_data,
              file_name="Requirement_Model.xlsx",
              mime=(
                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              ),
          )
        else:
          st.error(
              "Không tìm thấy cấu trúc bảng hợp lệ từ phản hồi của AI. Vui"
              " lòng thử lại."
          )

      except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
else:
  st.info(
      "Vui lòng nhập API Key và tải lên file PDF để bắt đầu sử dụng ứng dụng."
  )
