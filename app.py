import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

# Set Hugging Face API token from secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]

# Choose a compatible model (text-generation)
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.7, "max_new_tokens": 1024}
)

# Prompt template
template = """
Compare the following two products or services:

Product 1:
{product1}

Product 2:
{product2}

Instructions:
1. Provide a feature-by-feature comparison.
2. Generate a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for each.
3. Summarize key differentiators between them.
"""

prompt = PromptTemplate(
    input_variables=["product1", "product2"],
    template=template,
)

comparison_chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("🔍 Competitive Analysis Tool")
st.write("Compare two products or services using LLM-powered insights.")

product1 = st.text_area("Enter Product/Service 1 Description", height=200)
product2 = st.text_area("Enter Product/Service 2 Description", height=200)

if st.button("Compare"):
    if not product1 or not product2:
        st.warning("Please enter descriptions for both products.")
    else:
        result = comparison_chain.run(product1=product1, product2=product2)
        st.subheader("📊 Comparison Results")
        st.write(result)

        # Basic chart: Number of keywords in each description (just an example)
        p1_len = len(product1.split())
        p2_len = len(product2.split())
        fig, ax = plt.subplots()
        ax.bar(["Product 1", "Product 2"], [p1_len, p2_len], color=["skyblue", "salmon"])
        ax.set_ylabel("Word Count")
        ax.set_title("Word Count Comparison")
        st.pyplot(fig)

        # Create a downloadable PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt="Competitive Analysis Report\n\n" + result)
        # Generate PDF content as string and convert to BytesIO
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output = BytesIO(pdf_bytes)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_output,
            file_name="competitive_analysis_report.pdf",
            mime="application/pdf"
        )
