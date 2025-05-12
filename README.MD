# Website Summarizer

## A Python-based Tool to Summarize Website Content 🌐💬

The **Website Summarizer** is a Python-based tool designed to scrape website content and generate a concise, readable summary. Leveraging the power of the **Ollama** model, it filters out unnecessary elements such as images and ads, and summarizes the core content in a specified format, such as **Markdown**.

![Website Summarizer](./visuals/website_summarizer_visual.png)  <!-- Optional image path -->

## Features

✨ **Key Features**:
- **Scrapes websites** and extracts only relevant text for summarization.
- **Uses the Ollama AI model** for generating accurate, concise summaries.
- **Supports multiple output formats** (Markdown by default).
- **Easy integration**: Simply provide any website URL to get the summary.

## Requirements

To use the **Website Summarizer**, make sure you have the following:

- **Python 3.x**
- **requests**: For making HTTP requests to retrieve webpage content.
- **BeautifulSoup**: For parsing and cleaning HTML content.
- **IPython**: For rendering the summary in Markdown format (optional, for Jupyter environments).
- **ollama**: For interacting with the Ollama AI model.

Install the necessary libraries using pip:

```bash
pip install requests beautifulsoup4 ipython ollama
```
## Usage
### Initialization
You can initialize the `WebsiteSummarizer` class by specifying the model and the output language:

```python

summarizer = WebsiteSummarizer(model="llama3.2", language="markdown")
```

### Summarizing a Website
To summarize a website, simply call the `display_summary` method with the website URL:

```python

summarizer.display_summary("https://CNN.com")
```
This will:

1. Scrape the website content.

2. Send the content to the Ollama model for summarization.

3. Display the summarized content in the specified format (Markdown in this case).

### Customizing
You can change the AI model or output language by initializing `WebsiteSummarizer` with the desired settings:

```python

summarizer = WebsiteSummarizer(model="llama3.2", language="text")  # For plain text output
```

## How it Works 🔍
1. Website Scraping: The `Website` class scrapes content from the provided URL, cleaning up unnecessary elements such as images, scripts, and styles.

2. Prompt Generation: The `WebsiteSummarizer` class creates a message format for the Ollama model, prompting it to generate a summary of the website content.

Summary Display: The summary is returned by the model and displayed in the specified format (e.g., Markdown).

## Example Code
```python

summarizer = WebsiteSummarizer()
summarizer.display_summary("https://CNN.com")
```
## Contribution 🤝
Feel free to fork this repository, open issues, or submit pull requests to improve the project!
