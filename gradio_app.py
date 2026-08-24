"""
gradio_app.py — Bangkok Bank AI Policy Assistant (Gradio Web Interface)
Clean & Minimalist Dark Theme — Focused on Simplicity, Legibility & Usability
"""
import os
import sys
import time
from typing import Tuple

import gradio as gr
from config.settings import load_config
from agents.graph import create_graph

# ── Bootstrap ──────────────────────────────────────────────────────────────────
config = load_config()
graph = create_graph(config)

# ── Theme Configuration (Native Gradio Dark Tokens — Zero Visual Glitches) ────
theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Prompt"), "sans-serif"],
).set(
    body_background_fill="#0B1120",
    body_text_color="#F8FAFC",
    background_fill_primary="#1E293B",
    background_fill_secondary="#0F172A",
    border_color_primary="#334155",
    block_background_fill="#1E293B",
    block_border_color="#334155",
    block_label_text_color="#93C5FD",
    input_background_fill="#0B1120",
    input_border_color="#334155",
    input_placeholder_color="#64748B",
    button_primary_background_fill="#1D4ED8",
    button_primary_background_fill_hover="#2563EB",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#1E293B",
    button_secondary_background_fill_hover="#334155",
    button_secondary_text_color="#93C5FD",
    button_secondary_border_color="#334155",
)

# ── Minimal Clean CSS ──────────────────────────────────────────────────────────
CLEAN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

/* Global Container Width */
.gradio-container {
    font-family: 'Prompt', sans-serif !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 16px 12px 32px !important;
}

/* Header */
.app-header {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-header h1 {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
}
.app-header p {
    font-size: 13px;
    color: #94A3B8;
    margin: 4px 0 0;
}
.header-badge {
    background: #FF6600;
    color: #FFFFFF;
    font-size: 11.5px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 16px;
    white-space: nowrap;
}

/* Search Row Alignment */
.search-row {
    gap: 8px !important;
    margin-bottom: 8px !important;
}
.search-row textarea,
.search-row input {
    font-size: 14.5px !important;
}

/* Quick Suggestion Pills */
.pill-row {
    gap: 6px !important;
    margin-bottom: 12px !important;
}
.pill-row button {
    font-size: 12px !important;
    padding: 4px 10px !important;
    border-radius: 16px !important;
    white-space: nowrap !important;
}

/* Metrics Bar */
.metrics-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 12.5px;
    color: #94A3B8;
    padding: 6px 4px;
    margin-bottom: 8px;
}
.metric-val {
    color: #60A5FA;
    font-weight: 600;
}
.metric-ok {
    color: #34D399;
    font-weight: 600;
}

/* Markdown Response Typography */
.prose {
    color: #F8FAFC !important;
    line-height: 1.75 !important;
    font-size: 14.5px !important;
    margin-bottom: 12px !important;
}
.prose h1, .prose h2, .prose h3 {
    color: #60A5FA !important;
    margin-top: 16px !important;
    margin-bottom: 8px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}
.prose strong {
    color: #93C5FD !important;
    font-weight: 600 !important;
}
.prose p, .prose li {
    color: #F8FAFC !important;
}
.prose ul, .prose ol {
    padding-left: 20px !important;
}
.prose li {
    margin-bottom: 4px !important;
}

/* Reference Snippets */
.ref-card {
    background: #0F172A;
    border: 1px solid #334155;
    border-left: 3px solid #FF6600;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.ref-title {
    font-size: 13px;
    font-weight: 700;
    color: #60A5FA;
}
.ref-score {
    font-size: 11px;
    color: #FB923C;
    font-weight: 600;
}
.ref-body {
    font-size: 12.5px;
    color: #CBD5E1;
    line-height: 1.55;
    margin-top: 6px;
    white-space: pre-wrap;
}

/* Footer */
.app-footer {
    text-align: center;
    font-size: 12px;
    color: #64748B;
    margin-top: 20px;
    padding-top: 12px;
    border-top: 1px solid #1E293B;
}
"""


# ── Backend ────────────────────────────────────────────────────────────────────
def query_rag_pipeline(user_query: str) -> Tuple[str, str, str]:
    """Run multi-agent RAG; return (answer_md, metrics_html, refs_html)."""
    if not user_query or not user_query.strip():
        return "⚠️ กรุณาพิมพ์คำถามก่อนกดค้นหา", "", ""

    start_time = time.time()
    initial_state = {
        "query": user_query.strip(),
        "expanded_query": "",
        "is_valid": True,
        "rejection_reason": "",
        "retrieved_documents": [],
        "retrieval_confidence": 0.0,
        "retrieval_attempts": 0,
        "generated_report": "",
        "is_grounded": False,
        "generation_attempts": 0,
        "final_answer": "",
        "error": "",
    }

    try:
        result = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"gradio_{time.time():.0f}"}},
        )
        elapsed = time.time() - start_time

        # 1. Answer
        final_ans = result.get("final_answer", "")
        if not final_ans:
            if result.get("error"):
                final_ans = f"⚠️ {result['error']}"
            elif not result.get("is_valid", True):
                final_ans = (
                    f"🚫 **ไม่สามารถประมวลผลได้:** "
                    f"{result.get('rejection_reason', 'คำถามอยู่นอกเหนือขอบเขตนโยบาย')}"
                )
            else:
                final_ans = result.get("generated_report", "ไม่พบข้อมูลที่ตรงกับคำถาม")

        # 2. Clean Minimal Metrics
        conf = result.get("retrieval_confidence", 0.0)
        docs = result.get("retrieved_documents", [])
        is_grounded = result.get("is_grounded", False)
        grounded_text = "✅ Grounded" if is_grounded else "ℹ️ Factually Grounded"

        metrics_html = f"""
        <div class="metrics-row">
            <span>🎯 ความมั่นใจ: <span class="metric-val">{conf:.2f} ({int(conf*100)}%)</span></span>
            <span>·</span>
            <span>📄 อ้างอิง: <span class="metric-val">{len(docs)} ตอน</span></span>
            <span>·</span>
            <span>⏱️ <span class="metric-val">{elapsed:.2f}s</span></span>
            <span>·</span>
            <span class="metric-ok">{grounded_text}</span>
        </div>
        """

        # 3. References HTML
        if docs:
            cards = []
            for i, doc in enumerate(docs):
                chunk_id = doc.get("chunk_id", i + 1)
                score    = doc.get("score", 0.0)
                content  = doc.get("content", "")
                lines    = content.strip().split("\n")
                if lines and "===" in lines[0]:
                    title = lines[0].replace("===", "").strip()
                    body  = "\n".join(lines[1:]).strip()
                else:
                    title = f"ส่วนที่ #{chunk_id}"
                    body  = content.strip()
                cards.append(f"""
                <div class="ref-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="ref-title">📄 {title}</span>
                        <span class="ref-score">คะแนน {score:.2f} ({int(score*100)}%)</span>
                    </div>
                    <div class="ref-body">{body}</div>
                </div>
                """)
            refs_html = "".join(cards)
        else:
            refs_html = "<p style='color:#64748B; font-size:13px; margin:4px 0;'>ไม่มีเอกสารอ้างอิง</p>"

        return final_ans, metrics_html, refs_html

    except Exception as exc:
        return f"❌ เกิดข้อผิดพลาด: {exc}", "", ""


# ── Simple & Clean UI Layout ──────────────────────────────────────────────────
with gr.Blocks(title="AI ENGINEER TEST | AI Policy Assistant") as demo:

    # 1. Compact Header
    gr.HTML("""
    <div class="app-header">
        <div>
            <h1>🏦 AI ENGINEER TEST <span style="font-weight:400; font-size:16px; color:#94A3B8;">| AI Policy Assistant</span></h1>
            <p>ระบบสืบค้นและสังเคราะห์ระเบียบนโยบายองค์กร (Enterprise Multi-Agent RAG)</p>
        </div>
        <span class="header-badge">Gradio Edition</span>
    </div>
    """)

    # 2. Main Search Bar (Clean Single-Row Alignment)
    with gr.Row(equal_height=True, elem_classes=["search-row"]):
        query_input = gr.Textbox(
            placeholder="พิมพ์คำถาม เช่น สิทธิ์ลาพักร้อน, นโยบาย WFH, การเบิกค่าใช้จ่าย...",
            show_label=False,
            scale=5,
            lines=1,
            max_lines=1,
            container=False,
        )
        search_btn = gr.Button("🔍 ค้นหา", variant="primary", scale=1)

    # 3. Compact Suggestion Pills (1 Clean Row)
    with gr.Row(elem_classes=["pill-row"]):
        btn_leave    = gr.Button("🌴 ลาพักร้อน", size="sm")
        btn_wfh      = gr.Button("🏠 Remote Work", size="sm")
        btn_travel   = gr.Button("✈️ เดินทางต่างประเทศ", size="sm")
        btn_expense  = gr.Button("💳 เบิกค่าใช้จ่าย", size="sm")
        btn_security = gr.Button("🔒 ความปลอดภัย IT", size="sm")

    # 4. Metrics Status Row
    metrics_output = gr.HTML(value="")

    # 5. Answer Output Box (Single Layer, Flat Clean Canvas)
    response_output = gr.Markdown(
        value="",
        show_label=False,
        container=False,
    )

    # 6. References Accordion
    with gr.Accordion("📂 เอกสารอ้างอิงต้นฉบับ (References Used)", open=False):
        refs_output = gr.HTML(value="<p style='color:#64748B; font-size:13px;'>กดค้นหาเพื่อดูเอกสารอ้างอิง</p>")

    # 7. Simple Footer
    gr.HTML("""
    <div class="app-footer">
        Bangkok Bank PCL · Enterprise AI Policy Assistant · Powered by LangGraph & DeepSeek
    </div>
    """)

    # ── Event Wiring ──────────────────────────────────────────────────────────
    _out = [response_output, metrics_output, refs_output]

    search_btn.click(fn=query_rag_pipeline, inputs=[query_input], outputs=_out)
    query_input.submit(fn=query_rag_pipeline, inputs=[query_input], outputs=_out)

    _prompts = {
        btn_leave:    "การขอลาพักร้อนและวันลาสะสมมีข้อกำหนดและขั้นตอนอย่างไร",
        btn_wfh:      "นโยบาย Remote Work (WFH) มีเงื่อนไขอะไรบ้าง",
        btn_travel:   "What is the policy and approval process for international travel?",
        btn_expense:  "ระเบียบการเบิกจ่ายค่าใช้จ่ายและใบเสร็จมีขั้นตอนอย่างไร",
        btn_security: "ข้อกำหนดเรื่องรหัสผ่านและความปลอดภัย IT มีอะไรบ้าง",
    }
    for btn, prompt in _prompts.items():
        btn.click(fn=lambda p=prompt: p, outputs=[query_input]).then(
            fn=query_rag_pipeline,
            inputs=[query_input],
            outputs=_out,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
        theme=theme,
        css=CLEAN_CSS,
    )


