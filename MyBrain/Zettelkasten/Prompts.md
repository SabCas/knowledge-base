# Prompts

A PromptTemplate in LangChain allows you to define a reusable template for generating prompts, with placeholders (variables) for dynamic input, along with optional prefixes and suffixes for the prompt structure.

## Components:
- **Variables (Placeholders)**: These are the parts of the prompt that are dynamic, allowing you to insert specific information as needed.
- **Prefix:** The part of the prompt that appears before the placeholders. This could provide instructions or context.
- **Suffix:** The part of the prompt that appears after the placeholders. This often contains the final question or command for the language model.








## Code Example
In this example, we’ll use a PromptTemplate with a prefix, placeholders for variables, and a suffix to generate a well-structured prompt dynamically.

## Example: Question-Answer Prompt with Prefix and Suffix

```python
from langchain import PromptTemplate

# Define the template with a prefix, placeholders, and a suffix
template = """
You are an expert assistant. {prefix}

Here is some information: {context}

Using the above information, {suffix}:
{question}
"""

# Create a PromptTemplate object with input variables
prompt_template = PromptTemplate(
    input_variables=["prefix", "context", "suffix", "question"], 
    template=template
)

# Define dynamic inputs for the placeholders
prefix = "Please focus on accuracy."
context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
suffix = "answer the following question"
question = "Where is the Eiffel Tower located?"

# Fill the template with the inputs
filled_prompt = prompt_template.format(prefix=prefix, context=context, suffix=suffix, question=question)

# Output the filled prompt
print(filled_prompt)
```

