# Text Summarization

## Summarization Strategies
### 1. Stuff Method
In the Stuff method, all the input text is “stuffed” into the prompt in one go. The model processes the entire text at once to generate a summary. This is simple but only works well with shorter texts due to token limits.

**Pros:**
- Fast and simple.
- Works for shorter texts that fit within token limits.

**Cons:**
- Limited by token size (can’t handle large texts).
---

Example:

 ```python
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Initialize OpenAI LLM
llm = OpenAI()

# Sample long text (here, we wrap the text in a Document object)
docs = [Document(page_content="LangChain is a powerful tool for developing applications with language models.")]

# Load the 'stuff' summarization chain
stuff_chain = load_summarize_chain(llm, chain_type="stuff")

# Generate summary
summary = stuff_chain.run(docs)
print(summary)


```

### 2. Map-Reduce Method
The Map-Reduce method breaks the input text into smaller chunks. Each chunk is summarized individually (Map step), and the individual summaries are then combined to produce a final summary (Reduce step).

**Pros:**
- Can handle large documents.
- Breaks down tasks into smaller chunks, which makes it scalable for longer inputs.

**Cons:**
- The final summary might lose context or coherence across chunks.

Example:
```python
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Initialize OpenAI LLM
llm = OpenAI()

# Sample long text split into chunks
docs = [
    Document(page_content="LangChain is a framework for building applications powered by LLMs."),
    Document(page_content="It provides several components such as Chains, Agents, Prompts, and Memory.")
]

# Define custom prompts
map_prompt = "Summarize the following text: {text}"
combine_prompt = "Combine the following summaries into a cohesive paragraph: {summaries}"

# Load the map-reduce summarization chain with custom prompts
map_reduce_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt
)

# Generate summary
summary = map_reduce_chain.run(docs)
print(summary)


```

### 3. Refine Method
The Refine method also processes text in chunks. However, after summarizing each chunk, it refines the previous summary by adding new information from the next chunk. This iterative approach allows for a more coherent and context-aware summary.

**Pros:**
- Creates a more coherent final summary.
- Preserves context by refining the previous summary with each new chunk.

**Cons:**
- Slower than Map-Reduce since it processes text iterativel


Example with prompt template:
```python
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Define a prompt template for summarization
prompt_template = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="""
You are refining a summary. Here is what you have so far: {existing_answer}
Now, refine it with the following additional text:
{text}
""",
)

# Initialize OpenAI LLM
llm = OpenAI()

# Sample long text split into chunks (using Document objects)
docs = [
    Document(page_content="LangChain helps you build applications with language models."),
    Document(page_content="It supports modular components such as Chains, Agents, and Memory.")
]

# Load the refine summarization chain with the custom prompt template
refine_chain = load_summarize_chain(
    llm,
    chain_type="refine",
    prompt=prompt_template
)

# Generate summary
summary = refine_chain.run(docs)
print(summary)


```



