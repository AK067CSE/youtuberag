# YouTube AI Chatbot 🤖📺

**Created by Abhishek**

An intelligent chatbot that watches YouTube videos and answers questions about their content using advanced AI techniques.

---

## 🎯 Features

- **YouTube Video Analysis**: Automatically fetch and process transcripts from any YouTube video
- **Multi-Agentic RAG**: Specialized agents for retrieval, processing, and synthesis
- **BGE Embeddings**: State-of-the-art BGE (BAAI General Embedding) for semantic search
- **Advanced RAG Methods**: Multi-Query, HyDE, Agentic RAG, and more
- **Video RAG**: Specialized chunking and retrieval for video content
- **Vector Store**: Chroma-based embeddings for semantic search
- **Multi-language Support**: Automatic transcript language detection and translation
- **Modern UI**: Sleek dark-themed React frontend with smooth interactions

---

## 🧠 Advanced Multi-Agentic RAG Architecture

This project implements a sophisticated multi-agent RAG system using BGE embeddings and advanced retrieval methods.

### Mermaid Diagram: Multi-Agentic RAG System

```mermaid
flowchart TB
    subgraph "User Interface Layer"
        UI[React Frontend<br/>Chat + Video Input]
    end

    subgraph "Orchestration Layer"
        Router[Query Router<br/>Intent Classification]
        Memory[Memory Manager<br/>Conversation History]
        Supervisor[Agent Supervisor<br/>Tool Selection]
    end

    subgraph "Retrieval Agent"
        BGE[BGE Embedding Model<br/>bge-base-en-v1.5]
        HS[Hybrid Search<br/>Vector + Keyword]
        Exp[Query Expansion<br/>Multi-Query Generation]
        RRF[Reciprocal Rank<br/>Fusion]
    end

    subgraph "Processing Agent"
        Chunker[Smart Chunker<br/>Time-based + Semantic]
        Reranker[Cross-Encoder<br/>Re-ranking]
        Filter[Relevance Filter<br/>Threshold-based]
    end

    subgraph "Synthesis Agent"
        Prompt[Prompt Builder<br/>RAG Context Fusion]
        LLM[LLM Generator<br/>GPT-4o]
        Parser[Output Parser<br/>Structured Response]
    end

    subgraph "Vector Store Layer"
        Chroma[(Chroma DB<br/>BGE Embeddings)]
        FAISS[(FAISS Index<br/>Dense Retrieval)]
    end

    subgraph "Video Processing"
        YTFetcher[YouTube Transcript<br/>API Fetcher]
        LangDet[Language Detector<br/>Auto-translate]
    end

    UI --> Router
    Router --> Memory
    Router --> Supervisor
    
    Supervisor -->|Retrieve| RetrievalAgent
    Supervisor -->|Process| ProcessingAgent
    Supervisor -->|Synthesize| SynthesisAgent
    
    RetrievalAgent --> BGE
    BGE --> HS
    HS --> Chroma
    HS --> FAISS
    Exp --> RRF
    
    ProcessingAgent --> Chunker
    Chunker --> Reranker
    Reranker --> Filter
    
    SynthesisAgent --> Prompt
    Prompt --> LLM
    LLM --> Parser
    
    YTFetcher --> LangDet
    LangDet --> Chunker
    Chunker --> Chroma
    
    Parser --> UI
```

---

## 🔄 Advanced RAG Methods

### 1. Standard RAG (Naive RAG)

```mermaid
flowchart LR
    A[Query] --> B[Embed]
    B --> C[Retrieve]
    C --> D[Generate]
    D --> E[Response]
```

### 2. Advanced RAG Pipeline

```mermaid
flowchart LR
    subgraph "Pre-Retrieval"
        QP[Query Preprocessing]
        QE[Query Expansion]
        EE[Embedding Enhancement]
    end
    
    subgraph "Retrieval"
        HS[Hybrid Search]
        RR[Re-ranking]
    end
    
    subgraph "Post-Retrieval"
        FC[Filter & Cut]
        CB[Context Building]
    end
    
    QP --> QE --> EE --> HS --> RR --> FC --> CB --> Generate[Generate]
```

### 3. Multi-Query RAG

```mermaid
flowchart TB
    Q[Original Query] --> MQ[Multi-Query<br/>Generation]
    MQ --> Q1[Query v1]
    MQ --> Q2[Query v2]
    MQ --> Q3[Query v3]
    Q1 --> R1[Retrieve]
    Q2 --> R2[Retrieve]
    Q3 --> R3[Retrieve]
    R1 --> RRF[Reciprocal<br/>Rank Fusion]
    R2 --> RRF
    R3 --> RRF
    RRF --> F[Final Results]
```

### 4. HyDE (Hypothetical Document Embeddings)

```mermaid
flowchart LR
    Q[Query] --> HYDE[Generate<br/>Hypothetical<br/>Answer]
    HYDE --> E[Embed<br/>Hypothetical]
    E --> R[Retrieve<br/>Real Docs]
    R --> G[Generate<br/>Final Answer]
```

### 5. Agentic RAG

```mermaid
flowchart TB
    Start[User Query] --> Agent[LLM Agent]
    
    Agent --> Decide{Decide Action}
    
    Decide -->|Search| Search[Web Search]
    Decide -->|Retrieve| Ret[Vector Search]
    Decide -->|Clarify| Clar[Ask Follow-up]
    Decide -->|Answer| Ans[Generate Answer]
    
    Search --> Agent
    Ret --> Agent
    Clar --> Agent
    
    Ans --> End[Response]
    
    Agent -.-> Loop[Iterate if needed]
    Loop -.-> Agent
```

---

## 📊 RAG Method Comparison

```mermaid
flowchart LR
    subgraph "Comparison"
        A[Naive RAG] --> |Fast, Low Acc| A1[Simple QA]
        B[Advanced RAG] --> |Medium Speed, High Acc| B1[Production]
        C[Multi-Query] --> |Slow, High Acc| C1[Complex Docs]
        D[HyDE] --> |Medium, Very High Acc| D1[Ambiguous Q]
        E[Agentic RAG] --> |Slow, Highest Acc| E1[Research]
    end
```

---

## 🎬 Video RAG Specific Implementation

### Mermaid: Video RAG Pipeline

```mermaid
flowchart TB
    subgraph "Video Input"
        URL[YouTube URL]
        ID[Video ID Extract]
    end

    subgraph "Transcript Processing"
        TF[Transcript Fetcher]
        LD[Language Detection]
        Tr[Translator<br/>if needed]
    end

    subgraph "Chunking Strategies"
        TB[Time-Based<br/>30-sec windows]
        SB[Semantic<br/>Topic boundaries]
        HB[Hybrid<br/>Time + Semantic]
        SW[Sliding Window<br/>with overlap]
    end

    subgraph "Embedding Layer"
        BGE[BGE-m3 Embeddings<br/>Multi-lingual]
        Vec[Vector Store<br/>Chroma]
    end

    subgraph "Retrieval"
        HS[Hybrid Search<br/>BM25 + Dense]
        RR[Re-ranking<br/>BGE-Reranker]
    end

    subgraph "Generation"
        CB[Context Builder<br/>with timestamps]
        LLM[LLM Response<br/>with citations]
    end

    URL --> ID --> TF --> LD --> Tr
    Tr --> TB & SB & HB & SW
    TB & SB & HB & SW --> BGE
    BGE --> Vec --> HS --> RR --> CB --> LLM
```

### Video Chunking Strategies

```mermaid
flowchart LR
    subgraph "Chunking Methods"
        T1[Time-based<br/>Fixed intervals]
        S1[Semantic<br/>Topic detection]
        H1[Hybrid<br/>Combined approach]
        W1[Sliding Window<br/>With overlap]
    end
    
    T1 --> V1[Good for lectures]
    S1 --> V2[Good for varied content]
    H1 --> V3[Best for YouTube]
    W1 --> V4[Best for continuity]
```

---

## 🏗️ Complete System Architecture

```mermaid
flowchart TB
    subgraph "Frontend"
        React[React + Vite]
        Chat[Chat Interface]
        Video[Video Input]
    end

    subgraph "Backend API"
        Fast[FastAPI Server]
        Auth[Auth Handler]
        CORS[CORS Middleware]
    end

    subgraph "Agent Orchestrator"
        Router[Query Router]
        Memory[Conversation Memory]
        Planner[Execution Planner]
    end

    subgraph "Retrieval Pipeline"
        QE[Query Expander]
        BGE[BGE-bge-base-en-v1.5]
        HS[Hybrid Search]
        RRF[Reciprocal Rank Fusion]
        ReRank[BGE-Reranker-v1-m3]
    end

    subgraph "Video Pipeline"
        YTD[YouTube Transcript]
        Det[Language Detector]
        Trans[Translator]
        Chunk[Smart Chunker]
    end

    subgraph "Generation"
        Prompts[Prompt Templates]
        LLM[GPT-4o]
        Output[Output Parser]
    end

    subgraph "Storage"
        Chroma[(Chroma DB)]
        Cache[(Redis Cache)]
        History[(Chat History)]
    end

    React --> Fast
    Fast --> Router
    Router --> Planner
    Planner --> QE
    QE --> BGE
    BGE --> HS
    HS --> ReRank
    ReRank --> Prompts
    Prompts --> LLM
    LLM --> Output
    
    YTD --> Det
    Det --> Trans
    Trans --> Chunk
    Chunk --> Chroma
    
    Chroma --> HS
    History --> Memory
```

---

## 📝 Option B Explanation - Golden Dataset for RAG

Based on the assignment in [`1.txt`](1.txt), this project implements Option B - a Golden Dataset for RAG evaluation:

### Task Overview
Build a thoughtful evaluation set from 4 neural network/deep learning videos to test RAG system performance.

### Videos Used
1. **3Blue1Brown** - But what is a Neural Network? - `aircAruvnKk`
2. **3Blue1Brown** - Transformers, the tech behind LLMs - `wjZofJX0v4M`
3. **CampusX** - What is Deep Learning? (Hindi) - `fHF22Wxuyw4`
4. **CodeWithHarry** - All About ML & Deep Learning (Hindi) - `C6YtPJxNULA`

### Implementation in This Project
The project includes [`golden_dataset_option_b.py`](Youtube-Chatbot-main/Backend/golden_dataset_option_b.py) which:

1. **Fetches Transcripts** using YouTube Transcript API
2. **Processes Content** with text chunking strategies
3. **Generates QA Pairs** covering:
   - Neural network fundamentals
   - Deep learning concepts
   - Transformer architecture
   - Practical ML applications

### Methodology Notes

**Question Selection Criteria:**
- Cover different difficulty levels (basic, intermediate, advanced)
- Test various retrieval scenarios (factual, conceptual, procedural)
- Ensure answerable from transcript content

**Retrieval Testing:**
- What makes wrong retrieval: context from wrong video, outdated info, missing key details
- Good retrieval: precise, relevant, complete context

**Evaluation Focus:**
- Semantic similarity between query and retrieved chunks
- Factual accuracy of generated answers
- Source attribution correctness

---

## 🚀 Tech Stack

### Backend
- **FastAPI** - High-performance web framework
- **LangChain** - LLM application framework
- **BGE Embeddings** - State-of-the-art embeddings from BAAI
- **Chroma** - Vector database for embeddings
- **OpenAI** - GPT models for text generation
- **YouTube Transcript API** - Fetch video transcripts

### Frontend
- **React** - UI library
- **Vite** - Build tool
- **Axios** - HTTP client
- **React Hook Form** - Form handling
- **Tailwind CSS** - Styling

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 18+
- OpenAI API Key

### Backend Setup
```bash
cd Backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Create .env file with your OPENAI_API_KEY
python main.py
```

### Frontend Setup
```bash
cd Frontend
npm install
npm run dev
```

---

## 🔧 Environment Variables

Create a `.env` file in the Backend directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/url/upload` | POST | Upload YouTube URL and process transcript |
| `/api/chat` | POST | Send chat query and get AI response |

---

## 💡 Usage

1. **Start the backend server** on port 8080
2. **Start the frontend** on port 5173
3. **Paste a YouTube URL** in the input field
4. **Click "Watch Video"** to process the transcript
5. **Ask questions** about the video content

---

## 🔒 Security Notes

- Never expose your OpenAI API key in frontend code
- Use environment variables for sensitive data
- Consider rate limiting for production use
- Implement proper CORS policies

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) - LLM framework
- [BAAI](https://github.com/FlagOpen/FlagEmbedding) - BGE Embeddings
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- [Chroma](https://www.trychroma.com/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [3Blue1Brown](https://www.3blue1brown.com/) - Excellent math visualizations
- [Livo AI](https://livoassistant.com/) - AI products and automation