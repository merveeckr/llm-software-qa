import os
import sys
import time
import difflib
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Python 3.13 guard (PyTorch henüz desteklemiyor olabilir)
if sys.version_info >= (3, 13):
    st.error("Python 3.13 ile PyTorch/Transformers henüz stabil değil. Lütfen Python 3.10–3.12 kullanın.")
    st.stop()

# Lazy import: torch/transformers/bil yalnızca guard sonrası
try:
    from transformers import AutoTokenizer
    from bil import (
        BertBiLSTMCNN,
        LABEL_COLS,
        TEXT_COL,
        MODEL_NAME,
        MAX_LEN,
        DEVICE,
        THRESH,
        load_gemma_model,
        generate_ai_suggestion,
    )
    from english import (
        load_english_model_and_tokenizer,
        batch_predict_english,
        EN_MODEL_NAME as EN_DEFAULT_MODEL_NAME,
        EN_MAX_LEN as EN_DEFAULT_MAX_LEN,
        EN_THRESH as EN_DEFAULT_THRESH,
        compare_tr_en_predictions,
    )
except Exception as import_err:
    st.error(f"Kütüphane import hatası: {import_err}. Lütfen Python sürümünüzü ve paket kurulumlarınızı kontrol edin.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS))
    import torch  # local import to avoid top-level import on unsupported envs
    model.to(DEVICE)
    # Eğer eğitilmiş ağırlıklar varsa yükle
    if os.path.exists("best_model.pt"):
        state = torch.load("best_model.pt", map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_en_model_and_tokenizer(model_name: str):
    model, tokenizer = load_english_model_and_tokenizer(model_name=model_name)
    return model, tokenizer


def batch_predict(model: BertBiLSTMCNN, tokenizer, texts: List[str], max_len: int, threshold: float) -> np.ndarray:
    predictions = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        import torch  # local import
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
        predictions.append(preds)
    return np.vstack(predictions) if predictions else np.zeros((0, len(LABEL_COLS)), dtype=int)


def build_editable_frame(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    pred_df = pd.DataFrame(preds, columns=[f"AI_{c}" for c in LABEL_COLS])
    merged = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    # Kullanıcı işaretlemesi için sütunlar
    for c in LABEL_COLS:
        merged[f"User_{c}"] = merged.get(c, 0)
    return merged


st.set_page_config(page_title="Gereksinim Analizi", layout="wide")

def build_llm_prompt(requirement: str, missing: List[str]) -> str:
    missing_str = ", ".join(missing)
    return (
        "Aşağıdaki gereksinimi, eksik yönlerini gidererek TEK CÜMLE hâlinde yeniden yaz.\n"
        "- Girişi tekrar etme, alıntılama yapma.\n"
        "- Yalnızca şu formatta dön: İyileştirilmiş gereksinim: <cümle>\n"
        "- Türkçe yaz. Belirsizlikten kaçın (net, ölçülebilir, doğrulanabilir).\n"
        "- Gerektiğinde kabul ölçütlerini cümle içine açık ve sayısal geçir.\n"
        "- Madde işareti, liste, açıklama ekleme.\n"
        f"Eksikler: {missing_str}\n"
        f"Girdi: {requirement}"
    )

def _normalize_text(s: str) -> str:
    return str(s).strip().lower().rstrip('.').replace('\n', ' ')

CLAUSE_BY_LABEL = {
    'Unambiguous': "belirsiz terimler kullanılmadan net olarak",
    'Verifiable': "ölçülebilir ve doğrulanabilir kabul kriterleri ile",
    'Complete': "girdi, işlem ve çıktıları tanımlanmış olarak",
    'Conforming': "kurumsal standart ve yönergelere uygun şekilde",
    'Correct': "alan tanımlarına ve iş kurallarına uygun olarak",
    'Feasible': "mevcut sistem kısıtları içinde uygulanabilir düzeyde",
    'Necessary': "iş hedefleri açısından gerekli olan kapsamda",
    'Singular': "tek bir amacı ifade edecek şekilde",
    'Appropriate': "hedef kullanıcı kitlesine uygun biçimde",
}

def rule_based_improvement(requirement: str, missing: List[str]) -> str:
    base = _normalize_text(requirement)
    clauses = [CLAUSE_BY_LABEL.get(m, "") for m in missing]
    clauses = [c for c in clauses if c]
    if clauses:
        clause_text = ", ".join(clauses)
        improved = f"İyileştirilmiş gereksinim: {base} ve {clause_text} olacaktır."
    else:
        improved = f"İyileştirilmiş gereksinim: {base} olacaktır."
    return improved

def enforce_improvement(requirement: str, generated: str, missing: List[str]) -> str:
    text = generated.strip()
    low = text.lower()
    if "iyileştirilmiş gereksinim" in low:
        try:
            start = low.index("iyileştirilmiş gereksinim")
            text = text[start:].split("\n",1)[0]
            if ':' in text:
                text = text.split(':',1)[1].strip()
        except Exception:
            pass
    text = text.split("\n",1)[0].strip().strip('"').strip("'")
    r_base = _normalize_text(requirement)
    r_out = _normalize_text(text)
    sim = difflib.SequenceMatcher(None, r_base, r_out).ratio()
    if sim >= 0.9 or len(r_out) <= len(r_base):
        return rule_based_improvement(requirement, missing)
    return f"İyileştirilmiş gereksinim: {text.rstrip('.')}." if not text.lower().startswith("iyileştirilmiş gereksinim") else text
st.title("Gereksinim Analizi: BERTürk + Gemma Öneri")

with st.sidebar:
    st.markdown("**Model Ayarları**")
    model_name = st.text_input("BERT Model", value=MODEL_NAME)
    threshold = st.slider("Eşik (sigmoid)", min_value=0.05, max_value=0.95, value=float(THRESH), step=0.05)
    tr_text_col_default = TEXT_COL
    tr_text_col_input = st.text_input("Turkish Text Column", value=tr_text_col_default)
    st.markdown("**English Model**")
    en_model_name = st.text_input("English BERT Model", value=EN_DEFAULT_MODEL_NAME)
    en_threshold = st.slider("EN Threshold (sigmoid)", min_value=0.05, max_value=0.95, value=float(EN_DEFAULT_THRESH), step=0.05)
    en_text_col = st.text_input("English Text Column", value="Requirement_EN")
    compare_tr_en = st.checkbox("Compare Turkish vs English", value=True)
    st.divider()
    st.markdown("**LLM Ayarları**")
    llm_backend = st.selectbox("LLM Backend", options=["HF", "GGUF"], index=0)
    llm_offline = st.checkbox("HF offline", value=True)
    llm_models_input = st.text_input("LLM modelleri (; ile)", value="")
    llm_models = [m.strip() for m in llm_models_input.split(";") if m.strip()]
    active_llm = st.selectbox("Kullanılacak LLM", llm_models) if llm_models else None

uploaded_tr = st.file_uploader("Türkçe CSV yükleyin", type=["csv"], key="tr_csv")
uploaded_en = st.file_uploader("İngilizce CSV yükleyin", type=["csv"], key="en_csv")

if uploaded_tr is None or uploaded_en is None:
    st.write("Her iki CSV'yi de yükleyin ve analiz edin.")
    st.stop()

df_tr = pd.read_csv(uploaded_tr)
df_en = pd.read_csv(uploaded_en)
if tr_text_col_input not in df_tr.columns:
    st.error(f"TR CSV içinde '{tr_text_col_input}' kolonu bulunamadı.")
    st.stop()
if en_text_col not in df_en.columns:
    st.error(f"EN CSV içinde '{en_text_col}' kolonu bulunamadı.")
    st.stop()

st.success(f"TR yüklendi: {uploaded_tr.name}, satır: {len(df_tr)}")
st.success(f"EN yüklendi: {uploaded_en.name}, satır: {len(df_en)}")

with st.spinner("Modeller yükleniyor..."):
    model, tokenizer = load_model_and_tokenizer(model_name)
    en_model, en_tokenizer = load_en_model_and_tokenizer(en_model_name)

tr_texts = df_tr[tr_text_col_input].fillna("").astype(str).tolist()
en_texts = df_en[en_text_col].fillna("").astype(str).tolist()
n = min(len(tr_texts), len(en_texts))
tr_texts = tr_texts[:n]
en_texts = en_texts[:n]

with st.spinner("Tahmin yapılıyor..."):
    preds = batch_predict(model, tokenizer, tr_texts, MAX_LEN, threshold)
    preds_en = batch_predict_english(en_model, en_tokenizer, en_texts, max_len=EN_DEFAULT_MAX_LEN, threshold=en_threshold)

# Görünüm tabloları: solda TR, sağda EN
work_tr = pd.DataFrame(preds, columns=LABEL_COLS).astype(bool)
work_tr.insert(0, tr_text_col_input, tr_texts)
work_en = pd.DataFrame(preds_en, columns=LABEL_COLS).astype(bool)
work_en.insert(0, en_text_col, en_texts)

tr_text_col_for_view = tr_text_col_input
en_text_col_for_view = en_text_col
can_compare = True

st.subheader("Sonuçlar")
tr_cols = [f"TR_{c}" for c in LABEL_COLS]
en_cols = [f"EN_{c}" for c in LABEL_COLS] if can_compare else []
left, right = st.columns(2)
with left:
    st.markdown("**Türkçe (BERTürk)**")
    st.dataframe(
        work_tr[[tr_text_col_for_view] + LABEL_COLS],
        use_container_width=True,
        height=500,
    )
with right:
    if can_compare:
        st.markdown("**İngilizce (English BERT)**")
        cols_to_show = [en_text_col_for_view] if en_text_col_for_view in work_en.columns else [en_text_col_for_view]
        st.dataframe(
            work_en[cols_to_show + LABEL_COLS],
            use_container_width=True,
            height=500,
        )
    else:
        st.info("İngilizce kolon bulunamadığı için karşılaştırma gösterilemiyor.")
# TR vs EN agreement
if can_compare:
    try:
        tr_mat = work_tr[LABEL_COLS].astype(int).to_numpy()
        en_mat = work_en[LABEL_COLS].astype(int).to_numpy()
        agree_ratio = (tr_mat == en_mat).mean(axis=1)
        summary = pd.DataFrame({
            tr_text_col_for_view: work_tr[tr_text_col_for_view].astype(str).str.slice(0, 80) + "...",
            "TR_EN_Uyum_orani": np.round(agree_ratio, 3),
            "TR_EN_Uyum_sayisi": (tr_mat == en_mat).sum(axis=1),
            "Toplam_label": tr_mat.shape[1]
        })
        st.markdown("**BERTürk vs English BERT Uyum Özeti**")
        st.dataframe(summary, use_container_width=True, height=240)
        # Per-label aggregate using helper
        comp = compare_tr_en_predictions(tr_mat, en_mat)
        per_label_df = pd.DataFrame.from_dict(comp["per_label"], orient="index")
        per_label_df = per_label_df.reset_index().rename(columns={"index": "Label"})
        st.dataframe(per_label_df, use_container_width=True, height=300)
    except Exception as e:
        st.warning(f"TR-EN uyum hesabı yapılamadı: {e}")
# Birleştirilmiş çerçeve (TR + EN) diğer adımlar için
merged = pd.DataFrame({
    tr_text_col_for_view: work_tr[tr_text_col_for_view],
    en_text_col_for_view: work_en[en_text_col_for_view],
})
for lab in LABEL_COLS:
    merged[f"TR_{lab}"] = work_tr[lab].astype(int)
    merged[f"EN_{lab}"] = work_en[lab].astype(int)
# Eksik etiketlerin çıkarımı (TR ve EN)
def row_missing_tr(row) -> List[str]:
    return [c for c in LABEL_COLS if int(row.get(f"TR_{c}", 0)) == 0]
def row_missing_en(row) -> List[str]:
    return [c for c in LABEL_COLS if int(row.get(f"EN_{c}", 0)) == 0]
merged["Eksikler_TR"] = merged.apply(row_missing_tr, axis=1)
if can_compare:
    merged["Eksikler_EN"] = merged.apply(row_missing_en, axis=1)
st.download_button(
    label="CSV indir (TR/EN)",
    data=merged.to_csv(index=False).encode("utf-8"),
    file_name="gereksinim_sonuclari_tr_en.csv",
    mime="text/csv",
)
st.subheader("LLM ile Öneri Üretimi (TR tahminlerine göre)")
with st.expander("Satır seçerek öneri üret"):
    sel_idx = st.number_input("Satır index", min_value=0, max_value=len(merged)-1, value=0, step=1)
    if st.button("Seçili satıra öneri üret"):
        req_text = merged.iloc[sel_idx][tr_text_col_for_view]
        miss = merged.iloc[sel_idx]["Eksikler_TR"]
        if not miss:
            st.warning("BERTürk'e göre eksik yok.")
        else:
            try:
                prompt = build_llm_prompt(req_text, miss)
                if active_llm is None:
                    st.error("LLM modeli seçin.")
                else:
                    if llm_backend == "HF":
                        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                        import torch
                        cache = st.session_state.setdefault("_hf_gen_cache", {})
                        key = (active_llm, llm_offline)
                        if key not in cache:
                            tok = AutoTokenizer.from_pretrained(active_llm, local_files_only=llm_offline)
                            mdl = AutoModelForCausalLM.from_pretrained(active_llm, local_files_only=llm_offline)
                            cache[key] = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if torch.cuda.is_available() else -1, return_full_text=False)
                        gen = cache[key]
                        t0 = time.perf_counter()
                        out = gen(prompt, max_new_tokens=96, do_sample=False, num_beams=4)
                        dt = time.perf_counter() - t0
                        text = out[0].get('generated_text','').strip()
                    else:
                        try:
                            from llama_cpp import Llama
                        except Exception:
                            st.error("llama-cpp-python yüklü değil.")
                            raise
                        cache = st.session_state.setdefault("_llama_cache", {})
                        if active_llm not in cache:
                            cache[active_llm] = Llama(model_path=active_llm, n_ctx=2048)
                        llm = cache[active_llm]
                        t0 = time.perf_counter()
                        out = llm.create_completion(prompt=prompt, max_tokens=96, temperature=0.2, top_p=0.9)
                        dt = time.perf_counter() - t0
                        text = out["choices"][0]["text"].strip()
                    final_text = enforce_improvement(req_text, text, miss)
                    st.text_area("AI Önerisi", value=final_text, height=200)
                    st.info(f"Süre: {dt:.2f} sn")
            except Exception as e:
                st.error(f"LLM çalıştırılamadı: {e}")
st.subheader("Tekil Gereksinim Analizi ve Öneri (TR)")
single_req = st.text_area("Gereksinimi girin", placeholder="Gereksinimi buraya yapıştırın", height=120)
col1, col2 = st.columns(2)
with col1:
    if st.button("Eksikleri Analiz Et") and single_req.strip():
        enc = tokenizer(
            [single_req],
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        import torch  # local import
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            single_preds = (probs >= threshold).astype(int)
        single_missing = [LABEL_COLS[i] for i, v in enumerate(single_preds) if v == 0]
        st.session_state["single_missing"] = single_missing
        st.write("Eksikler:", single_missing if single_missing else "Yok")
with col2:
    if st.button("LLM ile Öneri Üret") and single_req.strip():
        miss = st.session_state.get("single_missing", [])
        if not miss:
            st.warning("Önce eksikleri analiz edin veya eksik yok.")
        else:
            try:
                prompt = build_llm_prompt(single_req, miss)
                if active_llm is None:
                    st.error("LLM modeli seçin.")
                else:
                    if llm_backend == "HF":
                        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                        import torch
                        cache = st.session_state.setdefault("_hf_gen_cache", {})
                        key = (active_llm, llm_offline)
                        if key not in cache:
                            tok = AutoTokenizer.from_pretrained(active_llm, local_files_only=llm_offline)
                            mdl = AutoModelForCausalLM.from_pretrained(active_llm, local_files_only=llm_offline)
                            cache[key] = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if torch.cuda.is_available() else -1, return_full_text=False)
                        gen = cache[key]
                        t0 = time.perf_counter()
                        out = gen(prompt, max_new_tokens=96, do_sample=False, num_beams=4)
                        dt = time.perf_counter() - t0
                        text = out[0].get('generated_text','').strip()
                    else:
                        try:
                            from llama_cpp import Llama
                        except Exception:
                            st.error("llama-cpp-python yüklü değil.")
                            raise
                        cache = st.session_state.setdefault("_llama_cache", {})
                        if active_llm not in cache:
                            cache[active_llm] = Llama(model_path=active_llm, n_ctx=2048)
                        llm = cache[active_llm]
                        t0 = time.perf_counter()
                        out = llm.create_completion(prompt=prompt, max_tokens=96, temperature=0.2, top_p=0.9)
                        dt = time.perf_counter() - t0
                        text = out["choices"][0]["text"].strip()
                    final_text = enforce_improvement(single_req, text, miss)
                    st.text_area("İyileştirilmiş Gereksinim", value=final_text, height=180)
                    st.info(f"Süre: {dt:.2f} sn")
            except Exception as e:
                st.error(f"LLM çalıştırılamadı: {e}")
                    
    # end uploaded path
