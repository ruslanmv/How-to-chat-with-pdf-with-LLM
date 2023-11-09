# ArXiv Research with Artificial Intelligence using IBM WatsonX

Today, we are going to build an interesting application that allows you to search files in ArXiv using WatsonX.

## Introduction

In the world of scientific research, finding relevant information from a vast pool of academic papers can be a daunting task. Traditional search engines often fall short in effectively retrieving the most pertinent articles, hindering progress in finding potential cures and treatments for critical health issues. However, with the advent of AI-powered technologies like WatsonX.ai and Streamlit, researchers now have a powerful tool at their disposal to navigate the wealth of knowledge stored in ArXiv.

In this blog, we will explore how to build an application that utilizes these cutting-edge technologies to answer scientific questions.

![demo](../assets/images/posts/readme/demo.gif)

The high-level structure of the program is as follows:

1. Question Analysis: Analyze your question using the Artificial Intelligence of WatsonX
2. Searching on ArXiv: Search for relevant papers on ArXiv
3. Download multiple papers and extract their text content.
4. Text Chunking: Divide the extracted text into smaller chunks that can be processed effectively.
5. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
6. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
7. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant contents.

## Step 1: Environment Creation

There are several ways to create an environment in Python. In this tutorial, we will show two options.

1. Conda method:

First, you need to install Anaconda from this [link](https://www.anaconda.com/products/individual). Install it in the location **C:\\Anaconda3** and then check if your terminal recognizes **conda** by typing the following command:

```
C:\\conda --version
conda 23.1.0
```

The environment we will use is Python 3.10.11. You can create an environment called **watsonx** (or any other name you prefer) by running the following command:

```
conda create -n watsonx python==3.10.11
```

After creating the environment, activate it:

```
conda activate watsonx
```

Next, install the necessary packages by running the following command:

```
conda install ipykernel notebook
```

2. Python native method:

First, install Python 3.10.11 from [here](https://www.python.org/downloads/). Then, create a virtual environment by running the following command:

```
python -m venv .venv
```

You will notice a new directory in your current working directory with the same name as your virtual environment. Activate the virtual environment:

```
.venv\\Scripts\\activate.bat
```

Upgrade pip:

```
python -m pip install --upgrade pip
```

Install the notebook package:

```
pip install ipykernel notebook
```

## Step 2: Setup Libraries

Once we have our running environment, we need to install additional libraries. Install the necessary libraries by running the following command:

```
pip install streamlit python-dotenv PyPDF2 arxiv langchain htmlTemplates ibm_watson_machine_learning requests pandas
```

Change the directory to the ArXiv-Chat-App:

```
cd ArXiv-Chat-App
```

Install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

Step 2: Retrieving an API Key from WatsonX.ai:

1. Visit the WatsonX.ai website (https://www.watsonx.ai) and sign up for an account.
2. Once signed in, navigate to the API section and generate an API key.
3. Copy the API key, as we will need it to authenticate our requests later.
4. Add the API key to the `.env` file in the project directory.

If you have a high-end NVIDIA GPU card, you can install the pytorch capability with CUDA:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## Step 3: Running your program

To use the ArXiv Chatter App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.
2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
```
streamlit run app.py
```
3. The application will launch in your default web browser, displaying the user interface.
4. Load multiple PDF documents into the app by following the provided instructions.
5. Ask questions in natural language about the loaded PDFs using the chat interface.

Conclusion:
By harnessing the power of AI, specifically WatsonX.ai and Streamlit, we have created an innovative application that revolutionizes the way researchers search in ArXiv. This technology empowers scientists to find solutions to critical health problems efficiently, potentially leading to groundbreaking discoveries and advancements in medical research. With AI as our ally, we can pave the way for a healthier future.

## Troubleshooting

You can get a list of existing Conda environments using the command below:

### Delete an Environment in Conda

```
conda env list
```

Before you delete an environment in Conda, you should first deactivate it. You can do that using this command:

```
conda deactivate
```

Once you've deactivated the environment, you will be switched back to the `base` environment. To delete an environment, run the command below:

```
conda remove --name ENV_NAME --all
```

Faiss issues:

If you encounter the following error:

```
INFO:faiss.loader:Loading faiss with AVX2 support.
INFO:faiss.loader:Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
INFO:faiss.loader:Loading faiss.
INFO:faiss.loader:Successfully loaded faiss.
```

Using Command Prompt (cmd):

1. Open Command Prompt as an administrator.
2. Navigate to the directory where you want to create the symbolic link using the `cd` command. For example, if you want to create the link in your user folder, you can use:

   ```
   cd your_python_path/site-packages/faiss
   ```

   You can retrieve your Python path by typing `conda info`.

3. Create the symbolic link using the `mklink` command as follows:

   ```
   mklink swigfaiss_avx2.py swigfaiss.py
   ```

   This command creates a symbolic link named `swigfaiss_avx2.py` that points to `swigfaiss.py`.

Using Linux:

```
cd your_python_path/site-packages/faiss
ln -s swigfaiss.py swigfaiss_avx2.py
```

## Contributing

This repository is intended for educational purposes.

## License

The ArXiv Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).