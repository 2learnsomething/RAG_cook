from typing import List, Dict
from langchain_core.documents import Document
from pathlib import Path
import uuid
from langchain_text_splitters import MarkdownHeaderTextSplitter


class DataPreparationModule:
    """数据准备模块-负责数据加载、清洗和预处理"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档，完整食谱
        self.chunks: List[Document] = []  # 子文档(按照标题分割的小块)
        self.parent_child_map: Dict[str, str] = {}  # 子模块ID ->父文档ID的映射

    def load_documents(self) -> List[Document]:
        """加载文档数据"""
        documents = []
        data_path_obj = Path(self.data_path)
        # 遍历所有文档格式为md的文件
        # rglob("*.md"): 递归查找所有Markdown文件
        for md_file in data_path_obj.rglob("*.md"):
            # 读取文档内容,保存Markdown格式
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            # 生成一个唯一的父ID
            parent_id = str(uuid.uuid4())
            # 创建document对象
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                },
            )
            documents.append(doc)

        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)
        self.documents = documents
        return documents

    def _enhance_metadata(self, doc: Document):
        """增强文档元素数据,直接修改"""
        file_path = Path(doc.metadata.get("source", ""))
        path_parts = file_path.parts

        category_mapping = {
            "meat_dish": "荤菜",
            "vegetable": "素菜",
            "soup": "汤品",
            "desert": "甜品",
            "breakfast": "早餐",
            "staple": "主食",
            "squatic": "水产",
            "condiment": "调料",
            "drink": "饮品",
        }
        doc.metadata["category"] = "其他"
        # 从路径分类上给予标签
        for key, value in category_mapping.items():
            if key in path_parts:
                doc.metadata["category"] = value
                break
        # 提取菜品名称
        doc.metadata["dish_name"] = file_path.stem

        # 分析菜品难度等级
        content = doc.page_content
        if "★★★★★" in content:
            doc.metadata["difficulty"] = "非常困难"
        elif "★★★★" in content:
            doc.metadata["difficulty"] = "困难"
        elif "★★★" in content:
            doc.metadata["difficulty"] = "中等"
        elif "★★" in content:
            doc.metadata["difficulty"] = "比较简单"
        elif "★" in content:
            doc.metadata["difficulty"] = "简单"
        else:
            doc.metadata["difficulty"] = "难度未知"

    def chunk_documents(self) -> List[Document]:
        """Markdown结构感知分块"""
        if not self.documents:
            raise ValueError("先加载文档")

        # 使用Markdown标题分割器
        chunks = self._markdown_header_split()

        # 为每一个chunk添加元数据
        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i  # 按照顺序分块的索引
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        self.chunks = chunks
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """使用Markdown标题分割器进行结构化分割"""
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),  # 菜品名称
            ("##", "二级标题"),  # 必备原料、计算、操作等
            ("###", "三级标题"),  # 建议版本、复杂版本等
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # 保留标题便于理解上下文
        )
        all_chunks = []
        for doc in self.documents:
            # 对于每一个文档进行Markdown分割
            md_chunks = markdown_splitter.split_text(doc.page_content)
            # 为每一个子块建立和父文档的关系
            parent_id = doc.metadata["parent_id"]

            for i, chunk in enumerate(md_chunks):
                # 为子块分配id并建立父子关系
                child_id = str(uuid.uuid4())
                # 继承父辈的metadata
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update(
                    {
                        "chunk_id": child_id,
                        "parent_id": parent_id,  # 这一行可不要
                        "doc_type": "child",  # 标记为子文档
                        "chunk_index": i,  # 在父文档中的位置
                    }
                )
                self.parent_child_map[child_id] = parent_id

            all_chunks.extend(md_chunks)
        return all_chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """根据子块获取对应的父文档"""

        # 统计每个父文档被匹配的次数
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档避免重复查找
                if parent_id not in parent_docs_map:
                    for doc in self.parent_child_map:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break
                # 按照相关性排序并构建去重后的父文档指标
                sorted_parent_ids = sorted(
                    parent_relevance.keys(),
                    key=lambda x: parent_relevance[x],
                    reverse=True,
                )

                # 构建去重后的父文档列表
                parent_doc = []
                for parent_id in sorted_parent_ids:
                    if parent_id in parent_docs_map:
                        parent_doc.append(parent_docs_map[parent_id])
                return parent_doc
