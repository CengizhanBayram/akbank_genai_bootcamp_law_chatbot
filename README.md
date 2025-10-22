# GraphRAG Hukuk Chatbot (PyQt)

Türk Anayasası (ve benzeri hukuk metinleri) üzerinde **GraphRAG** mimarisiyle yerelde çalışan bir soru–cevap asistanı.

* **Vektör arama:** FAISS (LangChain)
* **Bilgi grafı:** NetworkX (Madde→Madde ardışık/section/xref bağları)
* **Hibrit geri getirme:** FAISS + **BM25** + **HyDE** + **Graph genişletme**
* **Reranking (opsiyonel):** Sentence-Transformers **CrossEncoder**
* **LLM:** varsayılan **Gemini** (Google Generative AI); **OpenAI** desteği opsiyonel
* **Arayüz:** **PyQt5** (masaüstü)



---

## İçindekiler

* [Ekran Görünümü](#ekran-görünümü)
* [Mimari Özeti](#mimari-özeti)
* [Gereksinimler](#gereksinimler)
* [Kurulum](#kurulum)
* [Hızlı Başlangıç](#hızlı-başlangıç)
* [Kullanım](#kullanım)
* [Yapılandırma (CFG)](#yapılandırma-cfg)
* [Dizin Yapısı](#dizin-yapısı)
* [Güvenlik / Sınırlamalar](#güvenlik--sınırlamalar)
* [Yol Haritası](#yol-haritası)

---

## Ekran Görünümü

PyQt arayüzü iki sekmeden oluşur:

* **İndeks**: PDF/TXT seç, **İndeksle** ➜ FAISS + Graf oluşturulur.
* **Sohbet**: API anahtarını gir, sorunu yaz ➜ yanıt + **Kaynaklar** (madde başlıkları) görünür.
<img width="1239" height="824" alt="image" src="https://github.com/user-attachments/assets/6bdc7513-0d03-461f-9b6e-cafb1e89a846" />

---

## Mimari Özeti

```
[PDF/TXT]
   │  extract_sections_and_articles()
   ▼
[Article list] ──► GraphBuilder (sequence/section/xref edges)
   │                                 │
   │                                 └─► [NetworkX Graph]
   ▼
Indexer (smart_chunk → embeddings)
   │
   └─► [FAISS]  ◄──── BM25 (opsiyonel, tüm dokümanlar)
                    ▲
HyDE (opsiyonel) ───┘   (LLM ile hipotetik notlar, ek vektör arama)

SORGULAMA:
User Q → FAISS/BM25/HyDE sonuçları → RRF fused → Graph genişletme →
Rerank (CrossEncoder opsiyonel) → En iyi parçalar → LLM (Gemini/OpenAI) yanıtı
```

---

## Gereksinimler

* **Python:** 3.9–3.11 (öneri: 3.10+)
* **İşletim Sistemi:** Windows / macOS / Linux
* **GPU:** Zorunlu değil (CPU yeterli). CrossEncoder/embeddings GPU ile hızlanır.

**Önerilen requirements (PyQt sürümü):**

```txt
# Core LangChain ekosistemi
langchain>=0.2.12
langchain-community>=0.2.12
langchain-openai
langchain-google-genai
google-generativeai

# Geri getirme + reranker
sentence-transformers
faiss-cpu
rank-bm25

# Yardımcılar ve masaüstü arayüz
PyPDF2
networkx
PyQt5

# (Opsiyonel) Bazı sistemlerde gerekebilir
# torch
# torchvision
```

> **Windows’ta FAISS** pip bazen sorun çıkarabilir. En sağlam yol:
>
> ```powershell
> conda install -c conda-forge faiss-cpu -y
> ```

---

## Kurulum

```powershell
# (Windows / Anaconda örneği)
"C:\Users\<USER>\anaconda3\python.exe" -m pip install --upgrade pip
"C:\Users\<USER>\anaconda3\python.exe" -m pip install -r requirements.txt
# FAISS pip’te hata verirse
a> conda install -c conda-forge faiss-cpu -y
```

> `ModuleNotFoundError: langchain_community` görürsen: `-r requirements` yerine **`-r requirements.txt`** kullandığından emin ol.

---

## Hızlı Başlangıç

1. **Belgeyi hazırla**: `data/` klasörüne Anayasa PDF/TXT koy veya indeks sekmesinde dosya seç.
2. **API anahtarı** (sohbet için):

   * **Gemini** (varsayılan):

     ```powershell
     $env:GOOGLE_API_KEY="YOUR_KEY"
     ```
   * **OpenAI** (opsiyonel; `CFG.llm_backend = "openai"` olacak şekilde kodda değiştir):

     ```powershell
     $env:OPENAI_API_KEY="YOUR_KEY"
     ```
3. **Uygulamayı çalıştır**:

   ```powershell
   python main.py

   ```

---

## Kullanım

### İndeks Sekmesi

1. **Dosya Seç** ile PDF/TXT dosyasını seç.
2. **İndeksle** düğmesine bas — log alanında:

   * metin çıkarımı → madde tespiti
   * **graf oluşturma** (ardışık/section/xref)
   * **FAISS** vektör indeksi + manifest yazımı

### Sohbet Sekmesi

1. **GOOGLE_API_KEY** alanına anahtarını gir → **(Geçici) Ayarla**.
2. Sorunu yaz ve **Sor** de. Yanıtın altında **Kaynaklar** listelenir.

---

## Yapılandırma (CFG)

`Config` dataclass’ında başlıca alanlar:

* **Embedding**: `embedding_model='intfloat/multilingual-e5-base'`, `embedding_device='cpu'`, `normalize_embeddings=True`
* **Retrieval**: `top_k_vector`, `use_bm25`, `top_k_bm25`, `use_hyde`, `hyde_num`, `rrf_k`, `initial_topn`, `graph_neighbor_k`, `max_context_docs`
* **LLM**: `llm_backend=('gemini'|'openai')`, `gemini_model`, `openai_model`, `temperature`
* **Parçalama**: `chunk_chars`, `chunk_overlap`
* **Dizinler**: `data_dir`, `index_dir`, `graph_path`, `store_manifest`

> Parametrelerle oynamak, geri getirme kalitesi/latensi üzerinde büyük etki yapar. Küçük belgelerde `top_k_vector` ve `initial_topn` değerlerini düşürmek hız kazandırır.

---

## Dizin Yapısı

```
project/
├─ data/                     # PDF/TXT kaynakları
├─ indices/
│  ├─ anayasa_faiss/         # FAISS dosyaları
│  ├─ anayasa_graph.gpickle 
│  └─ anayasa_manifest.json  # indekslenen parça bilgileri
├─ main.py              # klasik PyQt sürümü (nx 2.x)         
└─ requirements-pyqt.txt
```

---




## Güvenlik / Sınırlamalar

* Asistan, **yalnızca indekslenen metinlerden** alıntı yapacak şekilde tasarlanmıştır. Yanıtlar **hukukî görüş** değildir.
* **Pickle** dosyaları yalnızca **güvendiğiniz** kaynaklardan yüklenmelidir.

---

## Yol Haritası

* UI: model seçimi (Gemini/OpenAI) ve RAG parametrelerini PyQt arayüzünden değiştirilebilir hale getirme (v2’de kısmen hazır)
* İndeks durumu ve istatistik paneli (düğüm/kenar sayısı, parça adedi)
* Sohbeti **Markdown/HTML** olarak dışa aktarma, **kopyala** düğmesi
* Çoklu belge desteği ve **koleksiyon** yönetimi
* Hızlı “Madde atıf grafı” görselleştirme


