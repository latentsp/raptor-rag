import logging
import pickle

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .embedding_models import BaseEmbeddingModel
from .qa_models import BaseQAModel, LiteLLMQAModel
from .reranker import BaseReRanker
from .summarization_models import BaseSummarizationModel
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Tree
from .utils import BaseTextSplitter

supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logger = logging.getLogger(__name__)


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,
        qa_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model="default",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        tr_reranker=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="default",
        tb_text_splitter=None,
    ):
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(f"tree_builder_type must be one of {list(supported_tree_builders.keys())}")

        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        if embedding_model is not None and not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError("embedding_model must be an instance of BaseEmbeddingModel")
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError("Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both.")
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        if summarization_model is not None and not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError("summarization_model must be an instance of BaseSummarizationModel")
        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        if tr_reranker is not None and not isinstance(tr_reranker, BaseReRanker):
            raise ValueError("tr_reranker must be an instance of BaseReRanker")

        if tb_text_splitter is not None and not isinstance(tb_text_splitter, BaseTextSplitter):
            raise ValueError("tb_text_splitter must be an instance of BaseTextSplitter")

        _tree_builder_class, tree_builder_config_class = supported_tree_builders[tree_builder_type]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
                text_splitter=tb_text_splitter,
            )

        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of "
                f"{tree_builder_config_class} for tree_builder_type "
                f"'{tree_builder_type}'"
            )

        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
                reranker=tr_reranker,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError("tree_retriever_config must be an instance of TreeRetrieverConfig")

        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or LiteLLMQAModel()
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = f"""
        RetrievalAugmentationConfig:
            {self.tree_builder_config.log_config()}

            {self.tree_retriever_config.log_config()}

            QA Model: {self.qa_model}
            Tree Builder Type: {self.tree_builder_type}
        """
        return config_summary


class RetrievalAugmentation:
    def __init__(self, config=None, tree=None):
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError("config must be an instance of RetrievalAugmentationConfig")

        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}") from e
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError("tree must be an instance of Tree, a path to a pickled Tree, or None")

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logger.info("Successfully initialized RetrievalAugmentation with Config %s", config.log_config())

    def add_documents(self, docs):
        if self.tree is not None:
            logger.warning("Overwriting existing tree.")

        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(
        self,
        question,
        start_layer: int | None = None,
        num_layers: int | None = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        if self.retriever is None:
            raise ValueError("The TreeRetriever instance has not been initialized. Call 'add_documents' first.")

        return self.retriever.retrieve(
            question,
            start_layer=start_layer,
            num_layers=num_layers,
            top_k=top_k,
            max_tokens=max_tokens,
            collapse_tree=collapse_tree,
            return_layer_information=return_layer_information,
        )

    def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int | None = None,
        num_layers: int | None = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        if return_layer_information:
            context, layer_information = self.retrieve(
                question,
                start_layer=start_layer,
                num_layers=num_layers,
                top_k=top_k,
                max_tokens=max_tokens,
                collapse_tree=collapse_tree,
                return_layer_information=True,
            )
            answer = self.qa_model.answer_question(context, question)
            return answer, layer_information

        context = self.retrieve(
            question,
            start_layer=start_layer,
            num_layers=num_layers,
            top_k=top_k,
            max_tokens=max_tokens,
            collapse_tree=collapse_tree,
            return_layer_information=False,
        )
        return self.qa_model.answer_question(context, question)

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logger.info("Tree successfully saved to %s", path)
