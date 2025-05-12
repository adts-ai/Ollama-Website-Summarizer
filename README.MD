# Website Summarizer

A Python-based tool for summarizing the content of websites using the Ollama model. The tool scrapes relevant text from websites, filters out unnecessary elements (like images and ads), and generates a concise summary in a specified format (such as Markdown).

## Features

- Scrapes websites and extracts relevant text.
- Uses the Ollama model for content summarization.
- Allows output in different formats (currently supports Markdown).
- Easily integrateable with any URL to summarize website content.

## Requirements

- Python 3.x
- `requests`: For making HTTP requests to retrieve the webpage content.
- `BeautifulSoup`: For parsing and cleaning up the HTML content.
- `IPython`: For rendering the summary in Markdown format (optional, for Jupyter environments).
- `ollama`: For accessing the Ollama AI model.

Install the necessary libraries using pip:

```bash
pip install requests beautifulsoup4 ipython ollama

Usage
Initialization
You can initialize the WebsiteSummarizer class by specifying the model and the output language:

python
Copy
Edit
summarizer = WebsiteSummarizer(model="llama3.2", language="markdown")
Summarizing a Website
To summarize a website, simply call the display_summary method with the website URL:

python
Copy
Edit
summarizer.display_summary("https://CNN.com")
This will:

Scrape the website content.

Send the content to the Ollama model for summarization.

Display the summarized content in the specified format (Markdown in this case).

Customizing
You can change the AI model or output language by initializing WebsiteSummarizer with the desired settings:

python
Copy
Edit
summarizer = WebsiteSummarizer(model="llama3.2", language="text")  # For plain text output
How it Works
Website Scraping: The Website class scrapes content from the provided URL, cleaning up unnecessary elements such as images, scripts, and styles.

Prompt Generation: The WebsiteSummarizer class creates a message format for the Ollama model, prompting it to generate a summary of the website content.

Summary Display: The summary is returned by the model and displayed in the specified format (e.g., Markdown).

Example Code
python
Copy
Edit
summarizer = WebsiteSummarizer()
summarizer.display_summary("https://CNN.com")