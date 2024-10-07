# Text Summarization


## 5 Levels Of Summarization: Novice to Expert
Summarization is a fundamental building block of many LLM tasks. You'll frequently run into use cases where you would like to distill a large body of text into a succinct set of points.

Depending on the length of the text you'd like to summarize, you have different summarization methods to choose from.

We're going to run through 5 methods for summarization that start with Novice and end up expert. These aren't the only options, feel free to make up your own. If you find another one you like please share it with the community.

### 5 Levels Of Summarization:

- Summarize a couple sentences - Basic Prompt
- Summarize a couple paragraphs - Prompt Templates, Stuff-Method
- Summarize a couple pages - Map Reduce/ Refine
- Summarize an entire book - Best Representation Vectors
- Summarize an unknown amount of text - Agents

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

### Level 4: Best Representation Vectors - Summarize an entire book

**The BRV Steps:**

- Load your book into a single text file
- Split your text into large-ish chunks
- Embed your chunks to get vectors
- Cluster the vectors to see which are similar to each other and likely talk about the same parts of the book
- Pick embeddings that represent the cluster the most (method: closest to each cluster centroid)
- Summarize the documents that these embeddings represent


Example

``` python
from langchain.document_loaders import PyPDFLoader

# Load the book
loader = PyPDFLoader("../data/IntoThinAirBook.pdf")
pages = loader.load()

# Cut out the open and closing parts
pages = pages[26:277]

# Combine the pages, and replace the tabs with spaces
text = ""
for page in pages:
    text += page.page_content   
text = text.replace('\t', ' ')

num_tokens = llm.get_num_tokens(text)
print (f"This book has {num_tokens} tokens in it")
```

``` python

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.cluster import KMeans
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Taking out the warnings
import warnings
from warnings import simplefilter

# Filter out FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)

docs = text_splitter.create_documents([text])

num_documents = len(docs)

print (f"Now our book is split up into {num_documents} documents")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectors = embeddings.embed_documents([x.page_content for x in docs])

# Assuming 'embeddings' is a list or array of 1536-dimensional embeddings

# Choose the number of clusters, this can be adjusted based on the book's content.
# I played around and found ~10 was the best.
# Usually if you have 10 passages from a book you can tell what it's about
num_clusters = 11

# Perform t-SNE and reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
reduced_data_tsne = tsne.fit_transform(vectors)

# Plot the reduced data
plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Book Embeddings Clustered')
plt.show()

# Find the closest embeddings to the centroids

# Create an empty list that will hold your closest points
closest_indices = []

# Loop through the number of clusters you have
for i in range(num_clusters):
    # Get the list of distances from that particular cluster center
    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

    # Find the list position of the closest one (using argmin to find the smallest distance)
    closest_index = np.argmin(distances)
    # Append that position to your closest indices list
    closest_indices.append(closest_index)

selected_indices = sorted(closest_indices)
selected_indices




```



