#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG tabanlı Anayasa (ve diğer hukuk metinleri) sohbet asistanı
- FAISS vektör indeksi (LangChain)
- Basit bilgi grafı (NetworkX) + çapraz referans (Madde -> Madde) kenarları
- Hibrit geri getirme: (Vektör) + (Graf genişletme)
- Reranking: Sentence Transformers CrossEncoder (opsiyonel)
- LLM: **Gemini API (varsayılan)** veya OpenAI/Ollama
- UI: Gradio ChatInterface

Kullanım:
1) Ortamı kurun (aşağıdaki `pip install` notlarına bakın).
2) İçeri aktarım (PDF veya TXT):
   python main.py ingest --path ./data/tc_anayasa.pdf
3) Soru-cevap (CLI):
   python main.py ask --q "Cumhurbaşkanının görev süresi kaç yıldır?"
4) Arayüz:
   python main.py ui

Notlar:
- İlk çalıştırmada embedding ve (varsa) reranker modelleri HF üzerinden indirilecektir.
- Uygulama yerelde çalışır; yanıt üretimi **Gemini API** ile yapılır. `GOOGLE_API_KEY` gereklidir.
- Türkçe için çok dilli embedding modeli varsayılan olarak seçilmiştir.
"""

import argparse
import os
import re
import json
import time
import uuid
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# ---- IO & NLP utils ----
try:
    from PyPDF2 import PdfReader  # hafif ve stabil
    _HAVE_PDF = True
except Exception:
    _HAVE_PDF = False

import networkx as nx

# LangChain + Vectorstore (FAISS)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

# LLM backend (LangChain)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_community.chat_models import ChatOllama
    _HAVE_OLLAMA = True
except Exception:
    _HAVE_OLLAMA = False

# Reranker (opsiyonel)
try:
    from sentence_transformers import CrossEncoder
    _HAVE_CROSS = True
except Exception:
    _HAVE_CROSS = False

# UI
import gradio as gr

# ---------------- CONFIG ----------------
@dataclass
class Config:
    data_dir: str = "./data"
    index_dir: str = "./indices/anayasa_faiss"
    graph_path: str = "./indices/anayasa_graph.gpickle"
    store_manifest: str = "./indices/anayasa_manifest.json"

    # Embedding (çok dilli ve güçlü bir genel amaçlı model)
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"  # "cuda" mevcutsa hızlanır
    normalize_embeddings: bool = True

    # Cross-encoder reranker (opsiyonel)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # hafif & hızlı
    reranker_top_k: int = 8

    # Retrieval
    # Ana Top-K ve grafik genişletme
    top_k_vector: int = 16
    graph_neighbor_k: int = 4  # her isabet için graf komşularından eklenecek aday sayısı
    max_context_docs: int = 8

    # Hibrit Arama (daha iyi geri getirme için)
    use_bm25: bool = True
    top_k_bm25: int = 12
    use_hyde: bool = True         # Hypothetical Document Embeddings (LLM ile)
    hyde_num: int = 2             # kaç sahte pasaj üretilecek
    rrf_k: int = 60               # Reciprocal Rank Fusion sabiti
    initial_topn: int = 24        # rerank öncesi aday havuzu

    # LLM
    llm_backend: str = "gemini"  # "gemini" | "openai" | "ollama"
    gemini_model: str = "gemini-1.5-pro"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0

    # Metin bölme
    chunk_chars: int = 1200
    chunk_overlap: int = 150

CFG = Config()

# --------------- DATA STRUCTURES ---------------
@dataclass
class Article:
    id: str         # ör: "Madde 101"
    section: str    # Bölüm/Başlık bilgisi (varsa)
    title: str      # Madde başlığı (yoksa id tekrar)
    text: str

# --------------- HELPERS ---------------
MADDE_PATTERNS = [
    r"(?i)\bmadde\s+(\d+)\b",  # Türkçe
    r"(?i)\barticle\s+(\d+)\b",  # İngilizce (olabilir)
]

MADDE_SPLIT = re.compile(r"(?is)(^|\n)\s*(madde\s+\d+\.?.*?)\n", re.MULTILINE)
SECTION_PAT = re.compile(r"(?i)^(bölüm|kısım|başlık)\s*[:\-]?\s*(.*)$")

REF_PAT = re.compile(r"(?i)madde\s+(\d+)")


def read_pdf(path: str) -> str:
    if not _HAVE_PDF:
        raise RuntimeError("PyPDF2 kurulu değil. `pip install PyPDF2`")
    reader = PdfReader(path)
    texts = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    return "\n".join(texts)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_sections_and_articles(raw: str) -> List[Article]:
    """Basit kural tabanlı: Bölüm/başlık ipuçlarını ve MADDE bloklarını yakalar.
    Metni makul parçalara böler; Anayasa formatlarında iyi çalışır.
    """
    # Normalize whitespace
    text = re.sub(r"\u00A0", " ", raw)
    text = re.sub(r"\r", "\n", text)

    # Satır satır gezinip en son görülen SECTION'u hatırla
    lines = text.splitlines()
    current_section = ""

    # Önce madde başlangıç indekslerini tespit edeceğiz
    # Strateji: madde başlığı satırlarında genelde "Madde X -" veya "MADDE X" geçer.
    madde_indices = []  # (line_idx, madde_id_text)
    for i, ln in enumerate(lines):
        ln_stripped = ln.strip()
        if SECTION_PAT.match(ln_stripped):
            current_section = ln_stripped
        # Madde yakala
        m = re.search(r"(?i)^(madde\s+\d+)\s*[:\-.]?\s*(.*)$", ln_stripped)
        if m:
            madde_indices.append((i, m.group(1).title()))  # "Madde 123"

    # Son indeksten dosya sonuna kadar uzanan bölümleri de alalım
    articles: List[Article] = []
    for idx, (start_i, madde_id) in enumerate(madde_indices):
        end_i = madde_indices[idx + 1][0] if idx + 1 < len(madde_indices) else len(lines)
        block = "\n".join(lines[start_i:end_i]).strip()

        # Başlık (ilk satırda madde tanımı sonrası kalan kısım)
        first_line = lines[start_i].strip()
        m_title = re.search(r"(?i)^madde\s+\d+\s*[:\-.]?\s*(.*)$", first_line)
        title = (m_title.group(1).strip() if (m_title and m_title.group(1)) else madde_id)

        # En yakın yukarıdaki SECTION'u bul (geriye doğru tarama)
        section = ""
        for j in range(start_i - 1, max(-1, start_i - 30), -1):
            sline = lines[j].strip()
            if SECTION_PAT.match(sline):
                section = sline
                break

        articles.append(Article(id=madde_id, section=section, title=title, text=block))

    # Eğer hiç madde yakalayamadıysak, tüm metni tek makale olarak ekleyelim
    if not articles:
        articles = [Article(id="Metin", section="", title="Genel", text=text)]

    return articles


def smart_chunk(text: str, max_chars: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # cümle sonlarına yakın kesmeye çalış
        window = text[start:end]
        # son nokta/virgül/semicolon arayalım
        cut = max(window.rfind("."), window.rfind("?"), window.rfind("!"))
        if cut == -1 or cut < max_chars * 0.4:
            cut = len(window)
        chunk = window[:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, start + cut - overlap)
        if start >= len(text):
            break
    return [c for c in chunks if c]


# --------------- INDEXING ---------------
class Indexer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.emb = HuggingFaceEmbeddings(model_name=cfg.embedding_model,
                                         model_kwargs={"device": cfg.embedding_device},
                                         encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings})

    def build(self, articles: List[Article]):
        os.makedirs(os.path.dirname(self.cfg.index_dir), exist_ok=True)

        docs: List[Document] = []
        manifest = []
        for a in articles:
            pieces = smart_chunk(a.text, self.cfg.chunk_chars, self.cfg.chunk_overlap)
            for i, p in enumerate(pieces):
                meta = {"article_id": a.id, "section": a.section, "title": a.title, "chunk_id": i}
                docs.append(Document(page_content=p, metadata=meta))
                manifest.append({"article_id": a.id, "section": a.section, "title": a.title, "chunk_id": i, "chars": len(p)})

        # FAISS oluştur
        vectordb = FAISS.from_documents(docs, self.emb)
        vectordb.save_local(self.cfg.index_dir)

        # Manifest kaydet
        with open(self.cfg.store_manifest, "w", encoding="utf-8") as f:
            json.dump({"count": len(docs), "docs": manifest}, f, ensure_ascii=False, indent=2)

        print(f"[OK] FAISS kaydedildi -> {self.cfg.index_dir}  (doc parçaları: {len(docs)})")


# --------------- GRAPH ---------------
class GraphBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, articles: List[Article]) -> nx.Graph:
        G = nx.Graph()
        # Düğümler: her madde
        for a in articles:
            G.add_node(a.id, section=a.section, title=a.title, text=a.text)

        # Kenarlar 1: art arda gelen maddeler (zayıf bağlam)
        for i in range(len(articles) - 1):
            id1, id2 = articles[i].id, articles[i + 1].id
            G.add_edge(id1, id2, type="sequence", weight=0.5)

        # Kenarlar 2: aynı bölümdeki maddeler (grup bağlamı)
        section_groups: Dict[str, List[str]] = {}
        for a in articles:
            key = a.section or ""
            section_groups.setdefault(key, []).append(a.id)
        for sec, ids in section_groups.items():
            for i in range(len(ids) - 1):
                G.add_edge(ids[i], ids[i + 1], type="section", weight=0.7)

        # Kenarlar 3: açık referanslar ("Madde X") -> güçlü bağ
        id_by_num: Dict[str, str] = {}  # "101" -> "Madde 101"
        for a in articles:
            m = re.search(r"(?i)madde\s+(\d+)", a.id)
            if m:
                id_by_num[m.group(1)] = a.id

        for a in articles:
            for hit in REF_PAT.findall(a.text):
                target = id_by_num.get(hit)
                if target and target != a.id:
                    # yönsüz kenar (graf basit tutuldu)
                    G.add_edge(a.id, target, type="xref", weight=1.0)

        # Kaydet
        os.makedirs(os.path.dirname(self.cfg.graph_path), exist_ok=True)
        nx.write_gpickle(G, self.cfg.graph_path)
        print(f"[OK] Graph kaydedildi -> {self.cfg.graph_path}  (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
        return G

    def load(self) -> nx.Graph:
        if not os.path.exists(self.cfg.graph_path):
            raise FileNotFoundError("Graph bulunamadı. Önce `ingest` çalıştırın.")
        return nx.read_gpickle(self.cfg.graph_path)


# --------------- RETRIEVAL + RERANK ---------------

def rrf_fuse(lists: List[List[Document]], k: int = 60, topn: int = 20) -> List[Tuple[Document, float]]:
    """Reciprocal Rank Fusion: birden fazla sıralı listeyi birleştirir.
    Skor = Σ 1/(k + rank). Aynı doküman birden çok listede yer alırsa toplanır.
    """
    scores: Dict[Tuple[str, int], float] = {}
    keep: Dict[Tuple[str, int], Document] = {}
    for lst in lists:
        for rank, d in enumerate(lst):
            key = (d.metadata.get("article_id"), d.metadata.get("chunk_id"))
            if key not in keep:
                keep[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [(keep[key], sc) for key, sc in items]
class Retriever:
    # ... mevcut yöntemler yukarıda ...

    def rerank(self, query: str, docs: List[Document], topk: int) -> List[Document]:
        if self.cross is None:
            q_emb = self.emb.embed_query(query)
            scored = []
            for d in docs:
                dv = self.emb.embed_documents([d.page_content])[0]
                score = sum(q * v for q, v in zip(q_emb, dv))
                scored.append((d, float(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [d for d, _ in scored[:topk]]

        pairs = [(query, d.page_content[:4096]) for d in docs]
        scores = self.cross.predict(pairs)
        tuples = list(zip(docs, scores))
        tuples.sort(key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in tuples[:topk]]


# --------------- LLM ANSWER ---------------
LEGAL_SYSTEM_PROMPT = (
    "Sen Türkiye Anayasası ve ilgili temel hukuk metinleri üzerinde eğitimli bir yardımcısın. "
    "Sadece sağlanan belgelerden alıntı yaparak yanıt ver. Madde numarası ve başlık belirterek kaynak göster. "
    "Belirsiz ise 'Metinde açık bir hüküm bulamadım' de ve en yakın ilgili maddeleri sun."
)

ANSWER_PROMPT = (
    "Kullanıcının sorusunu, aşağıdaki parçalardan doğrudan alıntılarla yanıtla.\n"
    "- Gerektiğinde madde numarası ver (örn. 'Madde 104').\n"
    "- Cevabın sonunda 'Kaynaklar' altında kullandığın maddeleri listele.\n"
    "- Varsayım yapma, metne sadık kal.\n\n"
    "Soru: {question}\n\n"
    "İlgili parçalar:\n{contexts}\n\n"
    "Yanıtını Türkçe, kısa ve net yaz."
)


def load_llm(cfg: Config):
    if cfg.llm_backend == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            print("[UYARI] GOOGLE_API_KEY bulunamadı. Lütfen ayarlayın.")
        return ChatGoogleGenerativeAI(model=cfg.gemini_model, temperature=cfg.temperature)

    if cfg.llm_backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("[UYARI] OPENAI_API_KEY bulunamadı. Ollama'ya düşülecek.")
        else:
            return ChatOpenAI(model=cfg.openai_model, temperature=cfg.temperature)

    if cfg.llm_backend == "ollama" or _HAVE_OLLAMA:
        try:
            return ChatOllama(model=cfg.ollama_model, temperature=cfg.temperature)
        except Exception as e:
            print(f"[UYARI] Ollama başlatılamadı: {e}")

    # Fallback
    return ChatGoogleGenerativeAI(model=cfg.gemini_model, temperature=cfg.temperature)
    if cfg.llm_backend == "ollama" or _HAVE_OLLAMA:
        try:
            return ChatOllama(model=cfg.ollama_model, temperature=cfg.temperature)
        except Exception as e:
            print(f"[UYARI] Ollama başlatılamadı: {e}")
    # Son çare: OpenAI'yı anahtar olsa da olmasa da zorla dener
    return ChatOpenAI(model=cfg.openai_model, temperature=cfg.temperature)


class QAEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ret = Retriever(cfg)
        self.llm = load_llm(cfg)

    def _hyde_passages(self, question: str, n: int) -> List[str]:
        if not self.cfg.use_hyde:
            return []
        prompt = (
            "Soruya dair, yasa/Anayasa üslubuna benzeyen kısa bilgi notları üret. "
            "Her not 3-5 cümle olsun, madde numarası uydurma. Soru: "
" + question"
        )
        passages = []
        try:
            for _ in range(n):
                resp = self.llm.invoke(prompt)
                txt = getattr(resp, "content", str(resp))
                passages.append(txt.strip())
        except Exception:
            pass
        return [p for p in passages if p]

    def answer(self, question: str) -> Tuple[str, List[Dict]]:
        # A) İlk adaylar: vektör + (opsiyonel) BM25 + (opsiyonel) HyDE
        lists: List[List[Document]] = []
        vec_docs = [d for (d, _s) in self.ret.vector_search(question, max(self.cfg.top_k_vector, self.cfg.initial_topn))]
        lists.append(vec_docs)

        if self.cfg.use_bm25:
            bm_docs = self.ret.bm25_search(question, self.cfg.top_k_bm25)
            if bm_docs:
                lists.append(bm_docs)

        if self.cfg.use_hyde:
            for hypo in self._hyde_passages(question, self.cfg.hyde_num):
                qv = self.ret.emb.embed_query(hypo)
                hy_docs = [d for (d, _s) in self.ret.vector_search_by_vec(qv, max(6, self.cfg.top_k_vector//2))]
                if hy_docs:
                    lists.append(hy_docs)

        fused = rrf_fuse(lists, k=self.cfg.rrf_k, topn=self.cfg.initial_topn)

        # B) Graf genişletme (seçili tohumlardan)
        expanded = self.ret.expand_with_graph(fused, self.cfg.graph_neighbor_k)

        # C) Rerank (CrossEncoder veya embedding-dot)
        reranked = self.ret.rerank(question, expanded, topk=self.cfg.max_context_docs)

        # D) Prompt + LLM
        ctx = self.make_context(reranked)
        prompt = LEGAL_SYSTEM_PROMPT + "\n\n" + ANSWER_PROMPT.format(question=question, contexts=ctx)
        resp = self.llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))

        # Kaynak listesi (benzersiz madde/başlık)
        sources = []
        seen = set()
        for d in reranked:
            aid = d.metadata.get("article_id")
            title = d.metadata.get("title")
            if (aid, title) in seen:
                continue
            seen.add((aid, title))
            sources.append({
                "article_id": aid,
                "title": title,
                "section": d.metadata.get("section"),
                "preview": d.page_content[:240] + ("…" if len(d.page_content) > 240 else "")
            })
        return text, sources


# --------------- COMMANDS ---------------

def cmd_ingest(args):
    path = args.path
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".pdf"):
        raw = read_pdf(path)
    else:
        raw = read_txt(path)

    articles = extract_sections_and_articles(raw)
    print(f"[INFO] Tespit edilen madde sayısı: {len(articles)}")

    # Graph
    gb = GraphBuilder(CFG)
    gb.build(articles)

    # Index
    ix = Indexer(CFG)
    ix.build(articles)


def cmd_ask(args):
    engine = QAEngine(CFG)
    ans, src = engine.answer(args.q)
    print("\n=== YANIT ===\n")
    print(ans)
    print("\n=== KAYNAKLAR ===\n")
    for i, s in enumerate(src, 1):
        print(f"{i}) {s['article_id']} — {s['title']} | {s['section']}")
        print(f"   {s['preview']}")


# --------------- UI (Gradio) ---------------

def ui_chat(history, message):
    engine = getattr(ui_chat, "_engine", None)
    if engine is None:
        engine = QAEngine(CFG)
        ui_chat._engine = engine
    answer, sources = engine.answer(message)

    # Kaynaklar bölümünü ekle
    src_md = "\n\n**Kaynaklar:**\n" + "\n".join(
        [f"- **{s['article_id']}** — {s['title']} ({s['section']})" for s in sources]
    )
    return answer + src_md


def cmd_ui(args):
    title = "GraphRAG • Hukuk Asistanı (Anayasa)"
    desc = (
        "Anayasa ve ilgili hukuk metinleri üzerinde çalışan bir yardımcı."

        "Hibrit geri getirme: FAISS + BM25 + HyDE + Graph genişletme + Reranker (LangChain)."
    )

    demo = gr.ChatInterface(
        fn=ui_chat,
        title=title,
        description=desc,
        theme=gr.themes.Soft(),
        retry_btn=None,
        undo_btn="Geri al",
        submit_btn="Sor",
        clear_btn="Temizle",
    )
    demo.launch()


# --------------- ENTRY ---------------

def main():
    parser = argparse.ArgumentParser(description="GraphRAG-Hukuk-Chatbot")
    sub = parser.add_subparsers(dest="cmd")

    p_ing = sub.add_parser("ingest", help="PDF/TXT hukuk metinlerini içe aktar ve indeksle")
    p_ing.add_argument("--path", required=True, help="PDF veya TXT dosya yolu")
    p_ing.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="CLI üzerinden soru sor")
    p_ask.add_argument("--q", required=True, help="Soru metni")
    p_ask.set_defaults(func=cmd_ask)

    p_ui = sub.add_parser("ui", help="Gradio arayüzünü başlat")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
