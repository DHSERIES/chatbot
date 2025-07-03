from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import OPENAI_API_KEY, LLM_MODEL
import os


class CustomLLMChatbot:
    def __init__(self, knowledge_base_path=None, tools_enabled=True):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt = PromptTemplate.from_template(
            "You are a helpful assistant. Maintain context. Respond concisely.\n\n{chat_history}\nHuman: {input}\nAI:"
        )

        self.knowledge_base = self._load_knowledge_base(knowledge_base_path) if knowledge_base_path else None
        self.tools = self._setup_tools() if tools_enabled else []
        self.agent_chain = self._build_agent_chain()

    def _load_knowledge_base(self, file_path):
        loader = TextLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)

    def _setup_tools(self):
        tools = []

        if self.knowledge_base:
            tools.append(
                Tool(
                    name="KnowledgeBaseQA",
                    func=self.knowledge_base.run,
                    description="Useful for answering questions about internal documents."
                )
            )

        return tools

    def _build_agent_chain(self):
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def chat(self, user_input):
        return self.agent_chain.run(user_input)