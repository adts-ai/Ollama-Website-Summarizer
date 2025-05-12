import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
import ollama

class WebsiteSummarizer:
    """Master class that handles website scraping, summarization, and prompt generation."""
    
    def __init__(self, model="llama3.2", language="markdown"):
        """
        Initialize the summarizer with a specific model and output language.
        
        Args:
            model (str): The AI model to be used for summarization.
            language (str): The format in which the summary will be presented (e.g., markdown).
        """
        self.model = model
        self.language = language
        self.system_prompt = self._generate_system_prompt(language)

    def _generate_system_prompt(self, language):
        """
        Generates the system prompt to guide the model's response.
        
        Args:
            language (str): The format for the output summary.
        
        Returns:
            str: The system prompt for the summarizer.
        """
        return f"You are an assistant that analyzes the contents of a website and provides a short summary. "\
               f"Ignore text related to navigation, ads, or images. Respond in {language}."

    class Website:
        """Nested class representing a scraped website and its content."""
        
        def __init__(self, url):
            """
            Scrape website content from the provided URL.
            
            Args:
                url (str): The URL of the website to scrape.
            """
            self.url = url
            self.title, self.text = self._scrape_website(url)

        def _scrape_website(self, url):
            """Scrape and extract relevant text from the website."""
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title and clean up the HTML by removing irrelevant tags
            title = soup.title.string if soup.title else "No title found"
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            text = soup.body.get_text(separator="\n", strip=True)
            
            return title, text

    def _create_messages(self, website):
        """
        Create the message format required by the Ollama model to generate a summary.
        
        Args:
            website (Website): The Website object containing the scraped content.
        
        Returns:
            list: The list of messages formatted for the model.
        """
        user_prompt = f"You are looking at a website titled '{website.title}'. The contents of this website are as follows:"\
                      f"\nPlease summarize the content in {self.language}. If there are any news or announcements, include them as well.\n\n"\
                      f"{website.text}"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def summarize(self, url):
        """
        Scrape the website content and summarize it using the Ollama model.
        
        Args:
            url (str): The URL of the website to summarize.
        
        Returns:
            str: The summarized content of the website.
        """
        website = self.Website(url)
        messages = self._create_messages(website)
        response = ollama.chat(model=self.model, messages=messages)
        return response['message']['content']

    def display_summary(self, url):
        """
        Display the summary of the website in a markdown format.
        
        Args:
            url (str): The URL of the website to display the summary of.
        """
        summary = self.summarize(url)
        display(Markdown(summary))

# Usage example:

summarizer = WebsiteSummarizer()
summarizer.display_summary("https://CNN.com")
