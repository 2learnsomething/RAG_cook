"""
Utility module for RAG operations including encoding, file loading, and visualization.
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sqlite3

# Langchain imports
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    BiliBiliLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain.chains.query_constructor.base import AttributerInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LlamaIndex imports
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

# Custom imports
from C3.visual_bge.visual_bge.modeling import Visualized_BGE
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import logging

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnablePassthrough,
)
from langchain_community.utils.math import cosine_similarity

warnings.filterwarnings("ignore")

# Constants
DEFAULT_IMG_HEIGHT = 300
DEFAULT_IMG_WIDTH = 300
DEFAULT_ROW_COUNT = 3
DEFAULT_DEVICE = "cpu"
SUPPORTED_FILE_TYPES = {"pdf", "md", "txt"}


class Encoder:
    """
    Encoder class for converting images and text into vector embeddings.

    Attributes:
        model: The Visualized_BGE model for encoding.
    """

    def __init__(self, model_name: str, model_path: str) -> None:
        """
        Initialize the encoder with a specific model.

        Args:
            model_name: Name of the BGE model to use.
            model_path: Path to the model weights.
        """
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> List[float]:
        """
        Encode a query with both image and text.

        Args:
            image_path: Path to the query image.
            text: Query text.

        Returns:
            Vector embedding as a list of floats.
        """
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> List[float]:
        """
        Encode an image into a vector embedding.

        Args:
            image_path: Path to the image.

        Returns:
            Vector embedding as a list of floats.
        """
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def dir_walk(directory: Union[str, Path]) -> List[str]:
    """
    Recursively walk through a directory and collect all file paths.

    Args:
        directory: Path to the directory to walk through.

    Returns:
        List of absolute file paths found in the directory and its subdirectories.

    Raises:
        ValueError: If the directory doesn't exist.
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    return all_files


def file_loaders(
    files: List[str],
    mode_pdf: str = "single",
    mode_md: str = "single",
) -> List[Union[PyMuPDFLoader, UnstructuredMarkdownLoader, TextLoader]]:
    """
    Create appropriate loaders for a list of files based on their extensions.

    Args:
        files: List of file paths to create loaders for.
        mode_pdf: Mode for PDF loading (default: "single").
        mode_md: Mode for Markdown loading (default: "single").

    Returns:
        List of document loaders.

    Raises:
        ValueError: If an unsupported file type is encountered.
    """
    loaders = []

    for file_path in tqdm(files, desc="Processing files"):
        file_ext = Path(file_path).suffix.lstrip(".").lower()

        if file_ext == "pdf":
            loaders.append(PyMuPDFLoader(file_path, mode=mode_pdf))
        elif file_ext == "md":
            loaders.append(UnstructuredMarkdownLoader(file_path, mode=mode_md))
        elif file_ext == "txt":
            loaders.append(TextLoader(file_path))
        else:
            raise ValueError(
                f"Unsupported file type: {file_ext}. "
                f"Supported types: {SUPPORTED_FILE_TYPES}"
            )

    return loaders


def load_document_from_loader(
    loaders: List[Union[PyMuPDFLoader, UnstructuredMarkdownLoader, TextLoader]],
) -> List[Document]:
    """
    Load documents from a list of loaders.

    Args:
        loaders: List of document loaders.

    Returns:
        List of loaded documents.
    """
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def load_embeddings(
    model_name: str,
    device: str = DEFAULT_DEVICE,
    normalize: bool = True,
    cache_folder: str = "embeddings",
) -> Optional[HuggingFaceEmbeddings]:
    """
    Load HuggingFace embeddings model.

    Args:
        model_name: Name of the HuggingFace model.
        device: Device to run the model on (default: "cpu").
        normalize: Whether to normalize embeddings (default: True).
        cache_folder: Folder to cache the model (default: "embeddings").

    Returns:
        HuggingFaceEmbeddings instance if model is supported, None otherwise.
    """
    supported_models = ["BAAI/bge-small-zh-v1.5", "BAAI/bge-small-en-v1.5"]

    if model_name not in supported_models:
        warnings.warn(
            f"Model {model_name} may not be tested. "
            f"Recommended models: {supported_models}"
        )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
        cache_folder=cache_folder,
    )


def visualize_results(
    query_image_path: str,
    retrieved_images: List[str],
    img_height: int = DEFAULT_IMG_HEIGHT,
    img_width: int = DEFAULT_IMG_WIDTH,
    row_count: int = DEFAULT_ROW_COUNT,
) -> np.ndarray:
    """
    Create a panoramic visualization showing query image and retrieved results.

    Args:
        query_image_path: Path to the query image.
        retrieved_images: List of paths to retrieved images.
        img_height: Height of each image tile (default: 300).
        img_width: Width of each image tile (default: 300).
        row_count: Number of images per row (default: 3).

    Returns:
        Panoramic image as a numpy array (BGR format).

    Raises:
        FileNotFoundError: If query image or any retrieved image is not found.
    """
    if not Path(query_image_path).exists():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")

    # Initialize panoramic images
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full(
        (panoramic_height, panoramic_width, 3), 255, dtype=np.uint8
    )
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # Process query image
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]  # RGB to BGR
    resized_query = cv2.resize(query_cv, (img_width, img_height))

    # Add blue border to query image
    bordered_query = cv2.copyMakeBorder(
        resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0)
    )

    # Place query image at bottom of display area
    query_display_area[img_height * (row_count - 1) :, :] = cv2.resize(
        bordered_query, (img_width, img_height)
    )

    # Add "Query" text
    cv2.putText(
        query_display_area,
        "Query",
        (10, panoramic_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    # Process retrieved images
    for i, img_path in enumerate(retrieved_images):
        if not Path(img_path).exists():
            warnings.warn(f"Retrieved image not found: {img_path}, skipping...")
            continue

        row, col = divmod(i, row_count)
        start_row = row * img_height
        start_col = col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]  # RGB to BGR
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))

        # Add black border
        bordered_retrieved = cv2.copyMakeBorder(
            resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        panoramic_image[
            start_row : start_row + img_height, start_col : start_col + img_width
        ] = bordered_retrieved

        # Add index number
        cv2.putText(
            panoramic_image,
            str(i),
            (start_col + 10, start_row + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    return np.hstack([query_display_area, panoramic_image])


def query_safe_recursive(
    query_str: str,
    summary_index,
    content_index,
    similarity_top_k: int = 1,
) -> str:
    """
    Perform a safe recursive query using summary-based routing.

    This function first routes the query through a summary index to find the most
    relevant category/sheet, then performs detailed retrieval in the content index
    filtered by that category.

    Args:
        query_str: The query string to search for.
        summary_index: VectorStoreIndex containing summaries for routing.
        content_index: VectorStoreIndex containing detailed content.
        similarity_top_k: Number of top results to retrieve (default: 1).

    Returns:
        Query response as a string.
    """
    print("--- 开始执行查询 ---")
    print(f"查询: {query_str}")

    # Step 1: Route using summary index
    print("\n第一步：在摘要索引中进行路由...")
    summary_retriever = VectorIndexRetriever(
        index=summary_index, similarity_top_k=similarity_top_k
    )
    retrieved_nodes = summary_retriever.retrieve(query_str)

    if not retrieved_nodes:
        return "抱歉，未能找到相关的电影年份信息。"

    matched_sheet_name = retrieved_nodes[0].node.metadata.get("sheet_name")
    if not matched_sheet_name:
        return "抱歉，检索节点缺少必要的元数据信息。"

    print(f"路由结果：匹配到工作表 -> {matched_sheet_name}")

    # Step 2: Retrieve from content index with filtering
    print("\n第二步：在内容索引中检索具体信息...")
    content_retriever = VectorIndexRetriever(
        index=content_index,
        similarity_top_k=similarity_top_k,
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="sheet_name", value=matched_sheet_name)]
        ),
    )
    query_engine = RetrieverQueryEngine.from_args(content_retriever)
    response = query_engine.query(query_str)
    return response


class SigLIPEmbeddingFunction:
    def __init__(
        self, model_name="google/siglip-base-patch16-256-multilingual", device="cpu"
    ):
        """
        初始化SigLIP嵌入函数
        Args:
            model_name: SigLIP模型名称
            device: 设备类型 ("cpu" 或 "cuda")
        """
        self.model_name = model_name
        self.device = device

        print(f"正在加载SigLIP模型:{model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # 限制词汇表大小
            stop_words="english",
            ngram_range=(1, 2),  # 考虑单词以及bigram
        )
        self.tfidf_fitted = False
        with torch.no_grad():
            dummy_text = ["test"]
            inputs = self.processor(
                text=dummy_text, padding="max_length", return_tensors="pt"
            )
            outputs = self.model.text_model(
                **{"k": v.to(device) for k, v in inputs.items() if k != "pixel_values"}
            )
            self.dense_dim = outputs.pooler_output.shape[-1]
        print(f"--> SigLIP 模型加载完成。密集向量维度: {self.dense_dim}")

    @property
    def dim(self):
        """返回维度信息，兼容原BGE-M3接口"""
        return {
            "dense": self.dense_dim,
            "sparse": (
                self.tfidf_vectorizer.max_features if self.tfidf_fitted else 10000
            ),
        }

    def fit_sparse(self, docs):
        print("--> 正在拟合 TF-IDF 模型...")
        self.tfidf_vectorizer.fit(docs)
        self.tfidf_fitted = True
        print(
            f"--> TF-IDF 模型拟合完成。词汇表大小: {len(self.tfidf_vectorizer.vocabulary_)}"
        )

    def encode_text_dense(self, texts):
        if isinstance(texts, str):
            text = [text]

        dense_vectors = []
        batch_size = 8  # 减小批次大小以节省内存

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts,
                    padding="max_padding",
                    trunction=True,
                    return_tenors="pt",
                )
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                    if k != "pixel_values"
                }
                outputs = self.model.text_model(**inputs)
                embeddings = outputs.pooler_output

                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                dense_vectors.extend(embeddings.cpu().numpy())
        return np.array(dense_vectors)

    def encode_text_sparse(self, texts):
        if not self.tfidf_fitted:
            raise ValueError("请先调用 fit_sparse() 方法拟合TF-IDF模型")

        if isinstance(texts, str):
            text = [texts]

        sparse_maxtrix = self.tfidf_vectorizer.transform(texts)
        return sparse_maxtrix

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if not self.tfidf_fitted:
            self.fit_sparse(texts)
        dense_vectors = self.encode_text_sparse(texts)
        sparse_vectors = self.encode_text_sparse(texts)
        return {"dense": dense_vectors, "sparse": sparse_vectors}


class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT重排器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = "bert-base-uncased"

        # 加载模型和分词器
        object.__setattr__(self, "tokenizer", AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, "model", AutoModel.from_pretrained(model_name))
        self.model.eval()
        print(f"ColBERT模型加载完成")

    def encode_text(self, texts):
        """colbert文本编码"""
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i : i + 1]

            # 计算相似度矩阵
            similarity_matrix = torch.matmul(
                query_emb, doc_emb.unsqueeze(0).transpose(-2, -1)
            )

            # 应用文档mask
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(
                ~doc_mask_expanded.bool(), -1e9
            )

            # maxsim操作
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 应用查询mask
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(
                ~query_mask_expanded.boll(), 0
            )

            # 求和得到最终的分数
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)
        return scores

    def compress_documents(self, documents, query, callbacks=None):
        """对文档进行ColBERT重排序"""
        if len(documents) == 0:
            return documents
        query_inputs = self.tokenizer(
            [query], return_tensor="pt", padding=True, trunction=True, max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_inputs.last_hidden_state, p=2, dim=-1)

        # 编码文档
        doc_texts = [doc.page_content for doc in documents]
        doc_inputs = self.tokenizer(
            doc_texts, return_tensors="pt", padding=True, trunction=True, max_length=128
        )
        with torch.no_grad():
            doc_outputs = self.model(**doc_inputs)
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)
        # 计算Colbert相似度
        scores = self.calculate_colbert_similarity(
            query_embeddings,
            doc_embeddings,
            query_inputs["attention_mask"],
            doc_inputs["attention_mask"],
        )

        scores_docs = list(zip(documents, scores))
        scores_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scores_docs[:5]]

        return reranked_docs
