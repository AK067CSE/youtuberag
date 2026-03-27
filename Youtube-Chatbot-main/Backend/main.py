from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptAvailable,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
    VideoUnavailable,
    TooManyRequests,
    InvalidVideoId,
    YouTubeRequestFailed,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from translate import Translator
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from golden_dataset_option_b import generate_option_b_dataset

load_dotenv(dotenv_path=f"{os.path.dirname(__file__)}/.env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


class user_entered_query(BaseModel):
    query: str


class uploaded_url(BaseModel):
    url: str


class OptionBGenerateRequest(BaseModel):
    num_questions: int = 5
    # Optional: default is all 4 required videos in the assignment prompt.
    video_ids: Optional[List[str]] = None


vector_store = None


@app.post("/url/upload")
def loading_transcripts(data: uploaded_url):
    global vector_store
    url = data.url.strip()

    # Extract video id from common YouTube URL shapes.
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0].split("/")[0]
    elif "youtube.com" in url and "v=" in url:
        video_id = url.split("v=")[-1].split("&")[0].split("?")[0]
    elif "youtube.com" in url and "/shorts/" in url:
        video_id = url.split("/shorts/")[-1].split("?")[0].split("/")[0]
    else:
        raise HTTPException(status_code=400, detail="Unsupported YouTube URL format")

    text = ""
    vector_store = None
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        available_langs = sorted({t.language_code for t in transcript_list})
        preferred = [lang for lang in ["en", "en-US", "en-GB"] if lang in available_langs]
        if not preferred and available_langs:
            # Fall back to whatever language exists.
            preferred = [available_langs[0]]

        # Try to load a manual transcript first; if that fails, try generated captions.
        try:
            transcript = transcript_list.find_transcript(preferred)
        except (NoTranscriptAvailable, NoTranscriptFound):
            if hasattr(transcript_list, "find_generated_transcript"):
                transcript = transcript_list.find_generated_transcript(preferred)
            else:
                raise

        chunks = transcript.fetch()

        # Some error cases may return an empty response instead of raising.
        text = " ".join(getattr(chunk, "text", "") for chunk in chunks).strip()
        if not text:
            raise CouldNotRetrieveTranscript("Transcript text is empty.")

    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcripts are disabled for this video")
    except (NoTranscriptAvailable, NoTranscriptFound):
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except TooManyRequests:
        raise HTTPException(
            status_code=429,
            detail="YouTube rate limit hit. Please wait a bit and try again.",
        )
    except YouTubeRequestFailed as e:
        # Check if it's a rate limit error
        if "429" in str(e):
            raise HTTPException(
                status_code=429,
                detail="YouTube rate limit hit. Please wait a bit and try again.",
            )
        raise HTTPException(
            status_code=502,
            detail="YouTube request failed while fetching transcript",
        )
    except (VideoUnavailable, InvalidVideoId):
        raise HTTPException(status_code=400, detail="Invalid or unavailable video id")
    except CouldNotRetrieveTranscript:
        raise HTTPException(status_code=502, detail="Could not retrieve transcript for this video")
    except Exception as e:
        # Covers XML parse errors like: `no element found: line 1, column 0`
        if isinstance(e, ET.ParseError):
            raise HTTPException(
                status_code=502,
                detail="YouTube returned an empty/invalid transcript response",
            )
        raise HTTPException(status_code=500, detail=f"Transcript processing failed: {type(e).__name__}")

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

    chunks = splitter.create_documents([text or ""])
    if not chunks:
        raise HTTPException(status_code=404, detail="Transcript splitting produced no chunks")

    # Vector Store

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)
    return {"message": "Transcript processed successfully"}


@app.post("/api/chat")
def youtube_chatbot(data: user_entered_query):
    user_query = data.query
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Upload a YouTube URL first via /url/upload")
    # Retriever
    retriever = vector_store.as_retriever(search_type="similarity", kwargs={"k": 5})
    prompt = PromptTemplate(
        template="""
You are a helpful assistant for answering questions about a YouTube video.

Follow these rules strictly:
1. If the question is a greeting or casual message (like "ok", "hi", "thanks", "cool", etc.), respond naturally and conversationally. Do NOT say you don't know.
2. If the question is about the video content AND the context is sufficient, answer in detail using ONLY the provided context.
3. If the question is about the video content BUT the context is insufficient, say you don't have enough information from the video to answer that.

The context can be in any language but always answer in English only.

Context:
{context}

Question:
{query}

Answer:
""",
        input_variables=["query", "context"],
    )

    def translate_to_english(chunks):
        translated_chunks = []
        translator = Translator(to_lang="en")
        for chunk in chunks:
            try:

                translated_text = translator.translate(chunk.page_content)

            except Exception:
                # Fallback: keep original text if translation fails
                translated_text = chunk.page_content

            translated_chunk = chunk.__class__(
                page_content=translated_text, metadata=chunk.metadata
            )
            translated_chunks.append(translated_chunk)

        return translated_chunks

    def format_docs(result):
        context_text = "\n\n".join(c.page_content for c in result)
        return context_text

    parallel_chain = RunnableParallel(
        {
            "context": retriever
            | RunnableLambda(translate_to_english)
            | RunnableLambda(format_docs),
            "query": RunnablePassthrough(),
        }
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()
    sequence_chain = prompt | model | parser

    final_chain = parallel_chain | sequence_chain

    result = final_chain.invoke(user_query)
    return result


@app.post("/api/option-b/generate")
async def option_b_generate(req: OptionBGenerateRequest) -> Dict[str, Any]:
    # Heavy pipeline: run it off the event loop.
    result = await asyncio.to_thread(
        generate_option_b_dataset,
        num_questions=req.num_questions,
        selected_video_ids=req.video_ids,
    )
    return result

