#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 Arayüzlü GraphRAG Hukuk Chatbot (Anayasa)
- FAISS (LangChain) vektör indeksi
- Bilgi grafı (NetworkX): ardışık/sektion/xref (Madde→Madde)
- Hibrit geri getirme: FAISS + BM25 + HyDE + Graph genişletme
- Reranking: Sentence-Transformers CrossEncoder (opsiyonel)
- LLM: Gemini (varsayılan) / OpenAI / Ollama

KULLANIM
--------
1) Kurulum (önerilen paketler):
   pip install -U pip
   pip install PyQt5 PyPDF2 networkx gradio faiss-cpu langchain langchain-community \
               langchain-openai langchain-google-genai google-generativeai \
               sentence-transformers rank-bm25
   # Windows'ta faiss sorun çıkarırsa: conda install -c conda-forge faiss-cpu -y

2) Gemini anahtarı (sohbet için):
   macOS/Linux: export GOOGLE_API_KEY=...
   Windows PS:  $env:GOOGLE_API_KEY="..."

3) Uygulama:
   python main_pyqt.py

Not: İndeksleme sırasında LLM gerekmez; yalnızca sohbet aşamasında kullanılır.
"""

import os
import re
import json
import pathlib
import pickle
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# --------- NLP / RAG bağımlılıkları ---------
import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from sentence_transformers import CrossEncoder
    _HAVE_CROSS = True
except Exception:
    _HAVE_CROSS = False

try:
    from PyPDF2 import PdfReader
    _HAVE_PDF = True
except Exception:
    _HAVE_PDF = False

# --------- PyQt5 ---------
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit,
    QTabWidget, QGroupBox, QFormLayout, QProgressBar,
    QComboBox, QSpinBox, QCheckBox, QSplitter, QSizePolicy,
    QToolButton, QScrollArea
)

# ---------------- CONFIG ----------------
@dataclass
class Config:
    data_dir: str = "./data"
    index_dir: str = "./indices/anayasa_faiss"
    graph_path: str = "./indices/anayasa_graph.gpickle"
    store_manifest: str = "./indices/anayasa_manifest.json"

    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval ayarları
    top_k_vector: int = 16
    graph_neighbor_k: int = 4
    max_context_docs: int = 8

    use_bm25: bool = True
    top_k_bm25: int = 12
    use_hyde: bool = True
    hyde_num: int = 2
    rrf_k: int = 60
    initial_topn: int = 24

    # LLM
    llm_backend: str = "gemini"  # "gemini" | "openai" | "ollama"
    gemini_model: str = "gemini-1.5-pro"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Parçalama
    chunk_chars: int = 1200
    chunk_overlap: int = 150

CFG = Config()

# --------------- YARDIMCI DESENLER ---------------
SECTION_PAT = re.compile(r"(?i)^(bölüm|kısım|başlık)\s*[:\-]?\s*(.*)$")
REF_PAT = re.compile(r"(?i)madde\s+(\d+)")

# --------------- IO ---------------

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

# --------------- MODEL YAPILARI ---------------
from dataclasses import dataclass
@dataclass
class Article:
    id: str
    section: str
    title: str
    text: str


def extract_sections_and_articles(raw: str) -> List[Article]:
    text = re.sub(r"\u00A0", " ", raw)
    text = re.sub(r"\r", "\n", text)
    lines = text.splitlines()

    # Madde başlangıçlarını tespit
    madde_indices: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        ln_stripped = ln.strip()
        if SECTION_PAT.match(ln_stripped):
            pass
        m = re.search(r"(?i)^(madde\s+\d+)\s*[:\-.]?\s*(.*)$", ln_stripped)
        if m:
            madde_indices.append((i, m.group(1).title()))

    articles: List[Article] = []
    for idx, (start_i, madde_id) in enumerate(madde_indices):
        end_i = madde_indices[idx + 1][0] if idx + 1 < len(madde_indices) else len(lines)
        block = "\n".join(lines[start_i:end_i]).strip()
        first_line = lines[start_i].strip()
        m_title = re.search(r"(?i)^madde\s+\d+\s*[:\-.]?\s*(.*)$", first_line)
        title = (m_title.group(1).strip() if (m_title and m_title.group(1)) else madde_id)
        section = ""
        for j in range(start_i - 1, max(-1, start_i - 30), -1):
            sline = lines[j].strip()
            if SECTION_PAT.match(sline):
                section = sline
                break
        articles.append(Article(id=madde_id, section=section, title=title, text=block))

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
        window = text[start:end]
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

# --------------- İNDEKS / GRAF ---------------
class Indexer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.emb = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": cfg.embedding_device},
            encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings},
        )

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

        vectordb = FAISS.from_documents(docs, self.emb)
        vectordb.save_local(self.cfg.index_dir)
        with open(self.cfg.store_manifest, "w", encoding="utf-8") as f:
            json.dump({"count": len(docs), "docs": manifest}, f, ensure_ascii=False, indent=2)
        return len(docs)


class GraphBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, articles: List[Article]) -> nx.Graph:
        G = nx.Graph()
        for a in articles:
            G.add_node(a.id, section=a.section, title=a.title, text=a.text)
        for i in range(len(articles) - 1):
            id1, id2 = articles[i].id, articles[i + 1].id
            G.add_edge(id1, id2, type="sequence", weight=0.5)
        section_groups: Dict[str, List[str]] = {}
        for a in articles:
            key = a.section or ""
            section_groups.setdefault(key, []).append(a.id)
        for _, ids in section_groups.items():
            for i in range(len(ids) - 1):
                G.add_edge(ids[i], ids[i + 1], type="section", weight=0.7)
        id_by_num: Dict[str, str] = {}
        for a in articles:
            m = re.search(r"(?i)madde\s+(\d+)", a.id)
            if m:
                id_by_num[m.group(1)] = a.id
        for a in articles:
            for hit in REF_PAT.findall(a.text):
                target = id_by_num.get(hit)
                if target and target != a.id:
                    G.add_edge(a.id, target, type="xref", weight=1.0)
        os.makedirs(os.path.dirname(self.cfg.graph_path), exist_ok=True)
        with open(self.cfg.graph_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        return G

# --------------- RETRIEVER ---------------
class Retriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.emb = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": cfg.embedding_device},
            encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings},
        )
        self.vectordb = FAISS.load_local(cfg.index_dir, self.emb, allow_dangerous_deserialization=True)
        with open(cfg.graph_path, "rb") as f:
            self.G = pickle.load(f)

        # BM25
        self.bm25 = None
        if cfg.use_bm25:
            all_docs = []
            for d in self.vectordb.docstore._dict.values():  # erişim: internal
                if isinstance(d, Document):
                    all_docs.append(d)
            if all_docs:
                self.bm25 = BM25Retriever.from_documents(all_docs)
                self.bm25.k = cfg.top_k_bm25

        # CrossEncoder
        self.cross = None
        if _HAVE_CROSS:
            try:
                self.cross = CrossEncoder(cfg.reranker_model)
            except Exception:
                self.cross = None

    def vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        results = self.vectordb.similarity_search_with_score(query, k=k)
        sims = []
        for doc, dist in results:
            sim = 1.0 / (1.0 + dist)
            sims.append((doc, sim))
        return sims

    def vector_search_by_vec(self, vec: List[float], k: int) -> List[Tuple[Document, float]]:
        results = self.vectordb.similarity_search_by_vector(vec, k=k)
        sims = []
        for rank, doc in enumerate(results):
            sims.append((doc, 1.0 / (1 + rank)))
        return sims

    def bm25_search(self, query: str, k: int) -> List[Document]:
        if self.bm25 is None:
            return []
        self.bm25.k = k
        return self.bm25.get_relevant_documents(query)

    def expand_with_graph(self, seeds: List[Tuple[Document, float]], k_neighbors: int) -> List[Document]:
        seed_ids = [d.metadata.get("article_id") for d, _ in seeds]
        candidates: Dict[Tuple[str, int], Document] = {}
        for d, _ in seeds:
            key = (d.metadata.get("article_id"), d.metadata.get("chunk_id"))
            candidates[key] = d
        for aid in seed_ids:
            if not aid or aid not in self.G:
                continue
            neighs = []
            for nb in self.G.neighbors(aid):
                w = self.G[aid][nb].get("weight", 0.5)
                neighs.append((nb, w))
            neighs.sort(key=lambda x: x[1], reverse=True)
            for nb, _w in neighs[:k_neighbors]:
                for d in self.vectordb.docstore._dict.values():  # internal
                    if isinstance(d, Document) and d.metadata.get("article_id") == nb:
                        key = (d.metadata.get("article_id"), d.metadata.get("chunk_id"))
                        candidates.setdefault(key, d)
        return list(candidates.values())

    def rerank(self, query: str, docs: List[Document], topk: int) -> List[Document]:
        if not docs:
            return []
        if self.cross is None:
            q_emb = self.emb.embed_query(query)
            scored = []
            for d in docs:
                dv = self.emb.embed_documents([d.page_content])[0]
                score = float(sum(q * v for q, v in zip(q_emb, dv)))
                scored.append((d, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [d for d, _ in scored[:topk]]
        pairs = [(query, d.page_content[:4096]) for d in docs]
        scores = self.cross.predict(pairs)
        tuples = list(zip(docs, scores))
        tuples.sort(key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in tuples[:topk]]

# --------------- LLM ---------------
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
        if not os.getenv("GOOGLE_API_KEY", ""):
            print("[UYARI] GOOGLE_API_KEY bulunamadı. Sohbet çağrıları başarısız olabilir.")
        return ChatGoogleGenerativeAI(model=cfg.gemini_model, temperature=cfg.temperature)
    if cfg.llm_backend == "openai":
        return ChatOpenAI(model=cfg.openai_model, temperature=cfg.temperature)
    # Ollama yolu eklemek isterseniz burada genişletilebilir
    return ChatGoogleGenerativeAI(model=cfg.gemini_model, temperature=cfg.temperature)


class QAEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ret = Retriever(cfg)
        self.llm = load_llm(cfg)

    def _hyde_passages(self, question: str, n: int) -> List[str]:
        if not self.cfg.use_hyde:
            return []
        prompt = f"""Soruya dair, yasa/Anayasa üslubuna benzeyen kısa bilgi notları üret.
Her not 3-5 cümle olsun, madde numarası uydurma.
Soru:
{question}"""
        passages = []
        try:
            for _ in range(n):
                resp = self.llm.invoke(prompt)
                txt = getattr(resp, "content", str(resp))
                passages.append(txt.strip())
        except Exception:
            pass
        return [p for p in passages if p]

    def make_context(self, docs: List[Document]) -> str:
        blocks = []
        seen = set()
        for d in docs:
            aid = d.metadata.get("article_id", "?")
            title = d.metadata.get("title", "")
            key = (aid, d.metadata.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            head = f"[{aid}] {title}" if title else f"[{aid}]"
            blocks.append(f"{head}\n{d.page_content}")
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str) -> Tuple[str, List[Dict]]:
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
        expanded = self.ret.expand_with_graph(fused, self.cfg.graph_neighbor_k)
        reranked = self.ret.rerank(question, expanded, topk=self.cfg.max_context_docs)
        ctx = self.make_context(reranked)
        prompt = LEGAL_SYSTEM_PROMPT + "\n\n" + ANSWER_PROMPT.format(question=question, contexts=ctx)
        resp = self.llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))
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

# --------------- RRF ---------------
from typing import Any

def rrf_fuse(lists: List[List[Document]], k: int = 60, topn: int = 20) -> List[Tuple[Document, float]]:
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

# --------------- UI: PyQt5 ---------------
APP_STYLE = """
* { font-family: 'Segoe UI', 'SF Pro Text', 'Inter', Arial; }
QMainWindow { background-color: #0B1220; }
QTabWidget::pane { border: 1px solid #1f2937; background: #0F172A; }
QTabBar::tab { background: #0F172A; color: #E5E7EB; padding: 10px 16px; border: 1px solid #1f2937; border-bottom: none; }
QTabBar::tab:selected { background: #111827; color: #FFFFFF; }
QGroupBox { color: #E5E7EB; border: 1px solid #1f2937; border-radius: 12px; margin-top: 12px; padding: 12px; background: #0E1627; }
QLabel { color: #CBD5E1; }
QLineEdit, QTextEdit, QSpinBox, QComboBox { color: #E5E7EB; background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 6px 8px; }
QPushButton { background: #10B981; color: #0B1220; border: none; border-radius: 12px; padding: 10px 14px; font-weight: 600; }
QPushButton:hover { background: #34D399; }
QPushButton:disabled { background: #1f2937; color: #94a3b8; }
QTextEdit#log, QTextEdit#chat { background: #0B1220; border: 1px solid #1f2937; color: #E5E7EB; border-radius: 12px; }
"""

class CollapsibleBox(QWidget):
    def __init__(self, title="Detaylar", parent=None, checked=True):
        super().__init__(parent)
        self.toggle = QToolButton(text=title, checkable=True, checked=checked)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggle.toggled.connect(self._on_toggled)

        self.content = QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content.setWidgetResizable(True)
        self.content.setFrameShape(self.content.NoFrame)
        self.content.setVisible(checked)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

    def setContentLayout(self, layout):
        w = QWidget(); w.setLayout(layout)
        self.content.setWidget(w)
        self.content.setMaximumHeight(w.sizeHint().height()+12)

    def _on_toggled(self, checked):
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)

# --------------- UI: PyQt5 ---------------
class IngestWorker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            if not os.path.exists(self.path):
                raise FileNotFoundError(self.path)
            raw = read_pdf(self.path) if self.path.lower().endswith('.pdf') else read_txt(self.path)
            self.progress.emit("Metin okundu, maddeler çıkarılıyor…")
            articles = extract_sections_and_articles(raw)
            self.progress.emit(f"Madde sayısı: {len(articles)} — Grafik oluşturuluyor…")
            gb = GraphBuilder(CFG)
            gb.build(articles)
            self.progress.emit("Vektör indeksi oluşturuluyor…")
            ix = Indexer(CFG)
            n = ix.build(articles)
            self.progress.emit("Tamamlandı.")
            self.done.emit(n)
        except Exception as e:
            self.error.emit(str(e))


class AskWorker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def run(self):
        try:
            self.progress.emit("Cevap hazırlanıyor…")
            engine = QAEngine(CFG)
            text, sources = engine.answer(self.question)
            self.done.emit(text, sources)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GraphRAG • Hukuk Asistanı (PyQt)")
        self.resize(980, 720)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # --- Tab 1: İndeks ---
        tab_idx = QWidget(); tabs.addTab(tab_idx, "İndeks")
        v = QVBoxLayout(tab_idx)

        form = QFormLayout()
        self.le_path = QLineEdit(); self.le_path.setPlaceholderText("Anayasa PDF/TXT yolu…")
        btn_browse = QPushButton("Dosya Seç")
        h = QHBoxLayout(); h.addWidget(self.le_path); h.addWidget(btn_browse)
        form.addRow("Kaynak dosya:", h)

        self.le_data = QLineEdit(CFG.data_dir)
        self.le_index = QLineEdit(CFG.index_dir)
        self.le_graph = QLineEdit(CFG.graph_path)
        form.addRow("data/ klasörü:", self.le_data)
        form.addRow("FAISS index:", self.le_index)
        form.addRow("Graph:", self.le_graph)

        v.addLayout(form)
        self.btn_ingest = QPushButton("İndeksle")
        v.addWidget(self.btn_ingest)

        self.te_log = QTextEdit(); self.te_log.setReadOnly(True)
        v.addWidget(self.te_log)

        btn_browse.clicked.connect(self.on_browse)
        self.btn_ingest.clicked.connect(self.on_ingest)

        # --- Tab 2: Sohbet ---
        tab_chat = QWidget(); tabs.addTab(tab_chat, "Sohbet")
        vc = QVBoxLayout(tab_chat)

        # LLM & API anahtarı
        g_api = QGroupBox("LLM ve Anahtar")
        fa = QFormLayout(g_api)
        self.cb_backend = QComboBox(); self.cb_backend.addItems(["gemini", "openai"])
        self.cb_backend.setCurrentText(CFG.llm_backend)
        default_key = os.getenv("GOOGLE_API_KEY", "") if CFG.llm_backend=="gemini" else os.getenv("OPENAI_API_KEY", "")
        self.le_api = QLineEdit(default_key)
        fa.addRow("LLM:", self.cb_backend)
        fa.addRow("API KEY:", self.le_api)
        btn_set = QPushButton("Anahtarı (Geçici) Ayarla")
        fa.addRow("", btn_set)
        vc.addWidget(g_api)
        btn_set.clicked.connect(self.on_set_api)

        # RAG ayarları (açılır/kapanır)
        rag_box = CollapsibleBox("RAG Ayarları", checked=False)
        rag_widget = QWidget(); fp = QFormLayout(rag_widget)
        self.cb_bm25 = QCheckBox("BM25 kullan"); self.cb_bm25.setChecked(CFG.use_bm25)
        self.cb_hyde = QCheckBox("HyDE kullan"); self.cb_hyde.setChecked(CFG.use_hyde)
        fp.addRow(self.cb_bm25, self.cb_hyde)
        def spin(val, lo, hi):
            s=QSpinBox(); s.setRange(lo,hi); s.setValue(val); return s
        self.sp_topk_vec = spin(CFG.top_k_vector, 1, 64)
        self.sp_topk_bm25 = spin(CFG.top_k_bm25, 1, 64)
        self.sp_init_topn = spin(CFG.initial_topn, 4, 128)
        self.sp_rrf = spin(CFG.rrf_k, 1, 200)
        self.sp_graph_k = spin(CFG.graph_neighbor_k, 0, 16)
        self.sp_ctx = spin(CFG.max_context_docs, 1, 24)
        self.sp_hyde_n = spin(CFG.hyde_num, 1, 8)
        fp.addRow("Top-K (vektör):", self.sp_topk_vec)
        fp.addRow("Top-K (BM25):", self.sp_topk_bm25)
        fp.addRow("Initial Top-N (fusion):", self.sp_init_topn)
        fp.addRow("RRF k:", self.sp_rrf)
        fp.addRow("Graph neighbor k:", self.sp_graph_k)
        fp.addRow("Max context docs:", self.sp_ctx)
        fp.addRow("HyDE sayısı:", self.sp_hyde_n)
        btn_apply = QPushButton("Ayarları Uygula")
        fp.addRow("", btn_apply)
        btn_apply.clicked.connect(self.on_apply_params)
        rag_box.setContentLayout(fp)
        vc.addWidget(rag_box)

        # Sohbet alanı (büyük) + altta yazma bölümü
        splitter = QSplitter(Qt.Vertical)
        self.te_chat = QTextEdit(); self.te_chat.setObjectName("chat"); self.te_chat.setReadOnly(True)
        self.te_chat.setPlaceholderText("Sohbet burada görünecek…")
        self.te_chat.setMinimumHeight(520)
        self.te_chat.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        composer = QWidget(); ch = QHBoxLayout(composer); ch.setContentsMargins(0,0,0,0)
        self.tx_compose = QTextEdit(); self.tx_compose.setPlaceholderText("Sorunuzu yazın… Shift+Enter: yeni satır")
        self.tx_compose.setFixedHeight(110)
        self.btn_ask = QPushButton("Gönder"); self.btn_ask.clicked.connect(self.on_ask)
        ch.addWidget(self.tx_compose); ch.addWidget(self.btn_ask)
        splitter.addWidget(self.te_chat)
        splitter.addWidget(composer)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 140])
        vc.addWidget(splitter)

        # tema
        self.setStyleSheet(APP_STYLE)

    # --------- UI Slots ---------
    def on_browse(self):
        pth, _ = QFileDialog.getOpenFileName(self, "PDF/TXT seç", "", "PDF/TXT (*.pdf *.txt)")
        if pth:
            self.le_path.setText(pth)

    def on_ingest(self):
        path = self.le_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir PDF/TXT dosyası seçin.")
            return
        # Klasörleri Config'e uygula
        CFG.data_dir = self.le_data.text().strip() or CFG.data_dir
        CFG.index_dir = self.le_index.text().strip() or CFG.index_dir
        CFG.graph_path = self.le_graph.text().strip() or CFG.graph_path
        os.makedirs(CFG.data_dir, exist_ok=True)

        self.btn_ingest.setEnabled(False)
        self.te_log.append("\n▶️ İndeksleme başlatıldı…")
        self.worker = IngestWorker(path)
        self.worker.progress.connect(lambda s: self.te_log.append(s))
        self.worker.error.connect(self.on_ingest_error)
        self.worker.done.connect(self.on_ingest_done)
        self.worker.start()

    def on_ingest_error(self, msg: str):
        self.btn_ingest.setEnabled(True)
        QMessageBox.critical(self, "Hata", msg)
        self.te_log.append(f"❌ Hata: {msg}")

    def on_ingest_done(self, n_parts: int):
        self.btn_ingest.setEnabled(True)
        self.te_log.append(f"✅ Tamam: {n_parts} parça indekslendi. Index: {CFG.index_dir}\nGraph: {CFG.graph_path}")

    def on_set_api(self):
        key = self.le_api.text().strip()
        backend = self.cb_backend.currentText().strip() if hasattr(self, 'cb_backend') else 'gemini'
        if not key:
            QMessageBox.warning(self, "Uyarı", "Geçerli bir API KEY girin.")
            return
        if backend == 'gemini':
            os.environ['GOOGLE_API_KEY'] = key
        else:
            os.environ['OPENAI_API_KEY'] = key
        CFG.llm_backend = backend
        QMessageBox.information(self, "Bilgi", f"{backend.upper()} anahtarı bu oturum için ayarlandı.")

    def on_apply_params(self):
        if hasattr(self, 'cb_bm25'):
            CFG.use_bm25 = self.cb_bm25.isChecked()
            CFG.use_hyde = self.cb_hyde.isChecked()
            CFG.top_k_vector = self.sp_topk_vec.value()
            CFG.top_k_bm25 = self.sp_topk_bm25.value()
            CFG.initial_topn = self.sp_init_topn.value()
            CFG.rrf_k = self.sp_rrf.value()
            CFG.graph_neighbor_k = self.sp_graph_k.value()
            CFG.max_context_docs = self.sp_ctx.value()
            CFG.hyde_num = self.sp_hyde_n.value()
        QMessageBox.information(self, "Ayarlar", "RAG ayarları güncellendi.")

    def on_ask(self):
        q = None
        if hasattr(self, 'tx_compose'):
            q = self.tx_compose.toPlainText().strip()
        else:
            q = getattr(self, 'le_q', QLineEdit()).text().strip()
        if not q:
            return
        if not (os.path.exists(CFG.index_dir) and os.path.exists(CFG.graph_path)):
            QMessageBox.warning(self, "Uyarı", "Önce indeks oluşturun (İndeks sekmesi).")
            return
        if hasattr(self, 'btn_ask'):
            self.btn_ask.setEnabled(False)
        self.te_chat.append(f"<b>Siz:</b> {q}")
        self.askw = AskWorker(q)
        self.askw.progress.connect(lambda s: self.te_chat.append(f"<i>{s}</i>"))
        self.askw.error.connect(self.on_ask_error)
        self.askw.done.connect(self.on_ask_done)
        self.askw.start()

    def on_ask_error(self, msg: str):
        self.btn_ask.setEnabled(True)
        self.te_chat.append(f"<span style='color:#c00'>Hata: {msg}</span>")

    def on_ask_done(self, text: str, sources: list):
        if hasattr(self, 'btn_ask'):
            self.btn_ask.setEnabled(True)
        src_html = "<br>".join([f"- <b>{s['article_id']}</b> — {s['title']} ({s['section']})" for s in sources])
        self.te_chat.append(f"<b>Asistan:</b> {text}<br><br><b>Kaynaklar:</b><br>{src_html}<hr>")
        if hasattr(self, 'tx_compose'):
            self.tx_compose.clear()
        elif hasattr(self, 'le_q'):
            self.le_q.clear()


# --------------- ENTRY ---------------
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
