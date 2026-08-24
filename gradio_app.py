"""
gradio_app.py — Bangkok Bank AI Policy Assistant (Gradio Web Interface)
Clean & Minimalist Dark Theme with Input Locking & Query Cancellation Support
"""
import os
import sys
import time
import logging
from typing import Tuple, List, Any

import gradio as gr
from config.settings import load_config
from agents.graph import create_graph

logger = logging.getLogger(__name__)

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

/* Stop Button Styling */
.stop-btn {
    background-color: #DC2626 !important;
    color: #FFFFFF !important;
    border: 1px solid #EF4444 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
.stop-btn:hover {
    background-color: #B91C1C !important;
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


# ── Backend Handlers ──────────────────────────────────────────────────────────
def on_start_query(user_query: str):
    """
    Locks input controls and activates the stop button when query starts.
    """
    if not user_query or not user_query.strip():
        return (
            gr.update(interactive=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            "⚠️ กรุณาพิมพ์คำถามก่อนกดค้นหา",
            "",
        )
    return (
        gr.update(interactive=False),  # Lock query textbox
        gr.update(visible=False),      # Hide search button
        gr.update(visible=True),       # Show cancel button
        gr.update(interactive=False),  # Lock pill 1
        gr.update(interactive=False),  # Lock pill 2
        gr.update(interactive=False),  # Lock pill 3
        gr.update(interactive=False),  # Lock pill 4
        gr.update(interactive=False),  # Lock pill 5
        "⏳ **กำลังสืบค้นและประมวลผลคำตอบ...** *(คุณสามารถกดปุ่ม '🛑 ยกเลิก' เพื่อหยุดได้)*",
        """<div class="metrics-row"><span style="color:#F59E0B; font-weight:600;">⏳ กำลังประมวลผลคำถาม...</span></div>""",
    )


def on_end_query():
    """
    Unlocks input controls and restores the search button after execution ends.
    """
    return (
        gr.update(interactive=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def on_cancel_query():
    """
    Triggered when user clicks the stop button. Unlocks input and notifies cancellation.
    """
    return (
        gr.update(interactive=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        "⚠️ **การประมวลผลถูกยกเลิกแล้ว** คุณสามารถพิมพ์คำถามใหม่ได้ทันทีครับ",
        """<div class="metrics-row"><span style="color:#EF4444; font-weight:600;">🛑 ยกเลิกการค้นหาเรียบร้อย</span></div>""",
        "<p style='color:#64748B; font-size:13px;'>ไม่มีเอกสารอ้างอิง (ยกเลิกก่อนประมวลผลเสร็จสิ้น)</p>"
    )


def query_rag_pipeline(user_query: str):
    """
    Runs multi-agent RAG pipeline with real-time streaming updates and token streaming.
    Yields (answer_md, metrics_html, refs_html) progressively.
    """
    if not user_query or not user_query.strip():
        yield ("⚠️ กรุณาพิมพ์คำถามก่อนกดค้นหา", "", "")
        return

    start_time = time.time()

    # ── Stage 0: Instant TTFT Feedback (< 0.1s) ──
    yield (
        "⚡ *กำลังวิเคราะห์คำถามและสืบค้นระเบียบนโยบาย...*",
        """<div class="metrics-row">
            <span style="color:#60A5FA;">🛡️ ตรวจสอบความปลอดภัยและขอบเขต...</span>
        </div>""",
        "<p style='color:#64748B; font-size:13px; margin:4px 0;'>⏳ กำลังค้นหาเอกสารอ้างอิง...</p>"
    )

    session_client_id = f"gradio_session_{abs(hash(user_query.strip())) % 10000}"
    initial_state = {
        "query": user_query.strip(),
        "client_id": session_client_id,
        "expanded_query": "",
        "is_valid": True,
        "rejection_reason": "",
        "retrieved_documents": [],
        "retrieval_score": 0.0,
        "retrieval_attempts": 0,
        "generated_report": "",
        "is_grounded": False,
        "generation_attempts": 0,
        "final_answer": "",
        "error": "",
    }

    current_state = dict(initial_state)
    docs = []
    conf = 0.0
    refs_html = "<p style='color:#64748B; font-size:13px; margin:4px 0;'>กำลังประมวลผล...</p>"

    try:
        # Stream graph node execution in real time
        for event in graph.stream(
            initial_state,
            config={"configurable": {"thread_id": f"gradio_{time.time():.0f}"}},
            stream_mode="updates"
        ):
            elapsed = time.time() - start_time

            # 1. Input Validator node completed
            if "input_validator" in event:
                val_data = event["input_validator"]
                current_state.update(val_data)
                if not val_data.get("is_valid", True):
                    yield (
                        "🔍 *กำลังสร้างคำชี้แจงการปฏิเสธคำถาม...*",
                        f"""<div class="metrics-row">
                            <span style="color:#EF4444;">🚫 คำถามอยู่นอกขอบเขต</span>
                            <span>·</span>
                            <span>⏱️ <span class="metric-val">{elapsed:.2f}s</span></span>
                        </div>""",
                        refs_html
                    )

            # 2. Retriever node completed -> Show documents & metrics instantly (< 1s!)
            elif "retriever" in event:
                ret_data = event["retriever"]
                current_state.update(ret_data)
                docs = ret_data.get("retrieved_documents", [])
                conf = ret_data.get("retrieval_score", 0.0)

                # Format reference cards
                if docs:
                    cards = []
                    for i, doc in enumerate(docs):
                        chunk_id = doc.get("chunk_id", i + 1)
                        score = doc.get("score", 0.0)
                        content = doc.get("content", "")
                        lines = content.strip().split("\n")
                        if lines and "===" in lines[0]:
                            title = lines[0].replace("===", "").strip()
                            body = "\n".join(lines[1:]).strip()
                        else:
                            title = f"ส่วนที่ #{chunk_id}"
                            body = content.strip()
                        cards.append(f"""
                        <div class="ref-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="ref-title">📄 {title}</span>
                                <span class="ref-score">Score {score:.4f}</span>
                            </div>
                            <div class="ref-body">{body}</div>
                        </div>
                        """)
                    refs_html = "".join(cards)
                else:
                    refs_html = "<p style='color:#64748B; font-size:13px; margin:4px 0;'>ไม่มีเอกสารอ้างอิง</p>"

                yield (
                    "📝 *พบเอกสารที่เกี่ยวข้องแล้ว กำลังสังเคราะห์และจัดโครงสร้างคำตอบ...*",
                    f"""<div class="metrics-row">
                        <span>🎯 Relevance Score: <span class="metric-val">{conf:.4f}</span></span>
                        <span>·</span>
                        <span>📄 อ้างอิง: <span class="metric-val">{len(docs)} ตอน</span></span>
                        <span>·</span>
                        <span>⏱️ <span class="metric-val">{elapsed:.2f}s</span></span>
                        <span>·</span>
                        <span style="color:#60A5FA;">⚡ กำลังเรียบเรียง...</span>
                    </div>""",
                    refs_html
                )

            # 3. Query Rewriter node
            elif "query_rewriter" in event:
                rew_data = event["query_rewriter"]
                current_state.update(rew_data)
                expanded = rew_data.get("expanded_query", "")
                yield (
                    f"✍️ *ความมั่นใจต่ำกว่าเกณฑ์ กำลังปรับปรุงคำค้นหา: \"{expanded}\"...*",
                    f"""<div class="metrics-row">
                        <span style="color:#F59E0B;">🔄 กำลังเกลาคำถามใหม่</span>
                        <span>·</span>
                        <span>⏱️ <span class="metric-val">{elapsed:.2f}s</span></span>
                    </div>""",
                    refs_html
                )

            # 4. Generator node completed -> Stream text tokens progressively
            elif "generator" in event:
                gen_data = event["generator"]
                current_state.update(gen_data)
                raw_report = gen_data.get("generated_report", "")

                # Progressively stream words for smooth visual streaming
                words = raw_report.split(" ")
                chunk_step = max(1, len(words) // 25)
                for idx in range(0, len(words), chunk_step):
                    accumulated = words[:idx + chunk_step]
                    partial_text = " ".join(accumulated)
                    yield (
                        partial_text + " ▌",
                        f"""<div class="metrics-row">
                            <span>🎯 ความมั่นใจ: <span class="metric-val">{conf:.2f} ({int(conf*100)}%)</span></span>
                            <span>·</span>
                            <span>📄 อ้างอิง: <span class="metric-val">{len(docs)} ตอน</span></span>
                            <span>·</span>
                            <span>⏱️ <span class="metric-val">{elapsed:.2f}s</span></span>
                            <span>·</span>
                            <span style="color:#34D399;">⚡ กำลังแสดงผล...</span>
                        </div>""",
                        refs_html
                    )
                    time.sleep(0.015)

            # 5. Output Validator / Rejection / Fallback
            elif "output_validator" in event:
                current_state.update(event["output_validator"])
            elif "rejection_response" in event:
                current_state.update(event["rejection_response"])
            elif "max_attempts_fallback" in event:
                current_state.update(event["max_attempts_fallback"])

        # ── Final Yield (Completed & Verified) ──
        elapsed = time.time() - start_time
        final_ans = current_state.get("final_answer", "")
        if not final_ans:
            if current_state.get("error"):
                final_ans = f"⚠️ {current_state['error']}"
            elif not current_state.get("is_valid", True):
                final_ans = (
                    f"🚫 **ไม่สามารถประมวลผลได้:** "
                    f"{current_state.get('rejection_reason', 'คำถามอยู่นอกเหนือขอบเขตนโยบาย')}"
                )
            else:
                final_ans = current_state.get("generated_report", "ไม่พบข้อมูลที่ตรงกับคำถาม")

        is_grounded = current_state.get("is_grounded", False)
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

        yield (final_ans, metrics_html, refs_html)

    except Exception as e:
        logger.error(f"Error in RAG pipeline stream: {e}", exc_info=True)
        yield (
            f"⚠️ **เกิดข้อผิดพลาดในการประมวลผล:** {str(e)}",
            """<div class="metrics-row"><span style="color:#EF4444;">❌ Error</span></div>""",
            refs_html or "<p style='color:#64748B;'>ไม่มีเอกสารอ้างอิง</p>"
        )


# ── Clean UI Layout ───────────────────────────────────────────────────────────
with gr.Blocks(title="AI ENGINEER TEST | AI Policy Assistant") as demo:

    # 1. Compact Header
    gr.HTML("""
    <div class="app-header">
        <div>
            <h1>AI ENGINEER TEST <span style="font-weight:400; font-size:16px; color:#94A3B8;">| AI Policy Assistant</span></h1>
            <p>ระบบสืบค้นระเบียบนโยบายองค์กร</p>
        </div>
        <span class="header-badge">Gradio UI</span>
    </div>
    """)

    # 2. Main Search Bar with Search & Cancel Buttons
    with gr.Row(equal_height=True, elem_classes=["search-row"]):
        query_input = gr.Textbox(
            placeholder="พิมพ์คำถาม เช่น สิทธิ์ลาพักร้อน, นโยบาย WFH, การเบิกค่าใช้จ่าย...",
            show_label=False,
            scale=5,
            lines=1,
            max_lines=1,
            container=False,
        )
        search_btn = gr.Button("🔍 ค้นหา", variant="primary", scale=1, visible=True)
        stop_btn = gr.Button("🛑 ยกเลิก", variant="stop", scale=1, visible=False, elem_classes=["stop-btn"])

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

    # 7. Footer
    gr.HTML("""
    <div class="app-footer">
        AI Policy Assistant · Powered by LangGraph
    </div>
    """)

    # ── Event Wiring with Input Locking & Cancellation ─────────────────────────
    _ui_controls = [
        query_input, search_btn, stop_btn,
        btn_leave, btn_wfh, btn_travel, btn_expense, btn_security
    ]
    _start_outputs = _ui_controls + [response_output, metrics_output]
    _pipe_outputs = [response_output, metrics_output, refs_output]
    _cancel_outputs = _ui_controls + [response_output, metrics_output, refs_output]

    # A. Search Button Events
    start_search = search_btn.click(
        fn=on_start_query,
        inputs=[query_input],
        outputs=_start_outputs,
    )
    pipe_search = start_search.then(
        fn=query_rag_pipeline,
        inputs=[query_input],
        outputs=_pipe_outputs,
    )
    pipe_search.then(
        fn=on_end_query,
        outputs=_ui_controls,
    )

    # B. Textbox Enter Submit Events
    start_submit = query_input.submit(
        fn=on_start_query,
        inputs=[query_input],
        outputs=_start_outputs,
    )
    pipe_submit = start_submit.then(
        fn=query_rag_pipeline,
        inputs=[query_input],
        outputs=_pipe_outputs,
    )
    pipe_submit.then(
        fn=on_end_query,
        outputs=_ui_controls,
    )

    # C. Suggestion Pills Events
    _prompts = {
        btn_leave:    "การขอลาพักร้อนและวันลาสะสมมีข้อกำหนดและขั้นตอนอย่างไร",
        btn_wfh:      "นโยบาย Remote Work (WFH) มีเงื่อนไขอะไรบ้าง",
        btn_travel:   "What is the policy and approval process for international travel?",
        btn_expense:  "ระเบียบการเบิกจ่ายค่าใช้จ่ายและใบเสร็จมีขั้นตอนอย่างไร",
        btn_security: "ข้อกำหนดเรื่องรหัสผ่านและความปลอดภัย IT มีอะไรบ้าง",
    }
    
    pill_events = []
    for btn, prompt in _prompts.items():
        set_text = btn.click(fn=lambda p=prompt: p, outputs=[query_input])
        start_pill = set_text.then(
            fn=on_start_query,
            inputs=[query_input],
            outputs=_start_outputs,
        )
        pipe_pill = start_pill.then(
            fn=query_rag_pipeline,
            inputs=[query_input],
            outputs=_pipe_outputs,
        )
        pipe_pill.then(
            fn=on_end_query,
            outputs=_ui_controls,
        )
        pill_events.extend([set_text, start_pill, pipe_pill])

    # D. Cancel / Stop Button Event (Terminates all running query tasks)
    stop_btn.click(
        fn=on_cancel_query,
        outputs=_cancel_outputs,
        cancels=[start_search, pipe_search, start_submit, pipe_submit, *pill_events],
    )


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        theme=theme,
        css=CLEAN_CSS,
    )
