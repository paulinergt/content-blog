"""
title: Llama Index Ollama Github Pipeline
author: paulinergt
date: 2025-03-04
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-github, pydantic
"""

# This code is based on a Python example pipeline from Open WebUI Pipelines. The original implementation can be found at:   
# https://github.com/open-webui/pipelines/blob/main/examples/pipelines/rag/llamaindex_ollama_github_pipeline.py  

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import os
import asyncio


class Pipeline:
    class Valves(BaseModel):
        EMBEDDING_MODEL_NAME: str
        OLLAMA_BASE_URL: str
        MODEL_NAME: str
        GITHUB_TOKEN: str
        GITHUB_REPO_OWNER: str
        GITHUB_REPO: str
        GITHUB_REPO_BRANCH: str


    def __init__(self):
        self.documents = None
        self.index = None
        self.valves = self.Valves(**{"EMBEDDING_MODEL_NAME":os.getenv("EMBEDDING_MODEL_NAME","nomic-embed-text"),
                                     "OLLAMA_BASE_URL":os.getenv("OLLAMA_BASE_URL","http://ollama:80"),
                                     "MODEL_NAME":os.getenv("MODEL_NAME","deepseek-r1:7b"),
                                     "GITHUB_TOKEN":os.getenv("GITHUB_TOKEN"),
                                     "GITHUB_REPO_OWNER":os.getenv("GITHUB_REPO_OWNER","open-webui"),
                                     "GITHUB_REPO":os.getenv("GITHUB_REPO","docs"),
                                     "GITHUB_REPO_BRANCH":os.getenv("GITHUB_REPO_BRANCH","main"),
                                     })

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.readers.github import GithubRepositoryReader, GithubClient
        from llama_index.core import SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.EMBEDDING_MODEL_NAME,
            base_url=self.valves.OLLAMA_BASE_URL,
        )
        print("ACCES AU MODÈLE D'EMBEDDING OK")

        Settings.llm = Ollama(model=self.valves.MODEL_NAME,
                             base_url=self.valves.OLLAMA_BASE_URL,)
        print("ACCES AU LLM OK")

        global index, documents

        # github_token = self.valves.GITHUB_TOKEN
        # owner = self.valves.GITHUB_REPO_OWNER
        # repo = self.valves.GITHUB_REPO
        # branch = self.valves.GITHUB_REPO_BRANCH

        # github_client = GithubClient(github_token=github_token, verbose=True)

        # reader = GithubRepositoryReader(
        #     github_client=github_client,
        #     owner=owner,
        #     repo=repo,
        #     use_parser=False,
        #     verbose=False,
        #     filter_file_extensions=(
        #         [
        #             ".png",
        #             ".jpg",
        #             ".jpeg",
        #             ".gif",
        #             ".svg",
        #             ".ico",
        #             "json",
        #             ".ipynb",
        #         ],
        #         GithubRepositoryReader.FilterType.EXCLUDE,
        #     ),
        # )

        # loop = asyncio.new_event_loop()

        # reader._loop = loop

        # try:
        #     # Load data from the branch
        #     self.documents = await asyncio.to_thread(reader.load_data, branch=branch)
        #     self.index = VectorStoreIndex.from_documents(self.documents)
        # finally:
        #     loop.close()

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)

        print(self.documents)
        print(self.index)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen