
# ArXiv Research with Artificial Intelligence with IBM WatsonX

Today we are going to build an interesting application that allow search files in ArXiv by using WatsonX.

## Introduction
In the world of scientific research, finding relevant information from a vast pool of academic papers can be a daunting task. Traditional search engines often fall short in effectively retrieving the most pertinent articles, for example hindering progress in finding potential cures and treatments for critical health issues. However, with the advent of AI-powered technologies like WatsonX.ai and Streamlit, researchers now have a powerful tool at their disposal to navigate the wealth of knowledge stored in ArXiv.

 In this blog, we will explore how to build an application that utilizes these cutting-edge technologies to answer scientific questions.

![demo](../assets/images/posts/readme/demo.gif)

The high level of the structure of the program is the follow

1. Question Analysis:Analize your question with Artificial Intelligence of WatsonX
2. Searching on Arxiv: Search the relevant papers on ArXiv
3. Download multiple  papers and extracts their text content.
4. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.
4. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
6. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant contents.


## Step 1. Enviroment creation

There are several ways to create an envirment in python, in this tutorial will show two options.
1. Conda method

First you need to install anaconda at this [link](https://www.anaconda.com/products/individual)


in this location **C:\Anaconda3** , then you, check that your terminal , recognize **conda**

```
C:\conda --version
conda 23.1.0
```

The environments supported that I will consider is Python 3.10.11

I will create an environment called **watsonx**, but you can put the name that you like.

```
conda create -n watsonx python==3.10.11
```

then we activate

```
conda activate watsonx
```
then in your terminal type the following commands:

```
conda install ipykernel notebook
```


2. Python native 

First we install python 3.10.11 [here](https://www.python.org/downloads/)


```
python -m venv .venv
```

You’ll notice a new directory in your current working directory with the same name as your virtual environment.

Activate the virtual environment.

```
.venv\Scripts\activate.bat
```
then


```
python -m pip install --upgrade pip
```

then we install our notebook, because also you can use Jupyter Notebook
```
pip install ipykernel notebook
```
# Step 2 . Setup libraries


Once we have our running environment  we install our kernel
```
python -m ipykernel install --user --name watsonx --display-name "Python (watsonx)"
```


 Install Required Libraries: Install the necessary libraries by running the following command:

```
pip install streamlit python-dotenv PyPDF2 arxiv langchain htmlTemplates ibm_watson_machine_learning requests pandas
   
```

```
cd ArXiv Chatter App
```
and  we install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

Step 2: Retrieving an API Key from WatsonX.ai:
1. Visit the WatsonX.ai website (https://www.watsonx.ai) and sign up for an account.

2. Once signed in, navigate to the API section and generate an API key.

3. Copy the API key, as we will need it to authenticate our requests later.


 and add it to the `.env` file in the project directory.


```

If you have high end GPU  NVIDIA card you can install the pytorch capability with cuda

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```



## Step 3 Run you program

-----
To use the ArxiV Chatter App, follow these steps:

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

## Troubleshoting

You can get a list of existing Conda environments using the command below:

### Delete an Environment in Conda

```bash
conda env list
```



Before you delete an environment in Conda, you should first deactivate it. You can do that using this command:

```bash
conda deactivate
```

Once you've deactivated the environment, you'd be switched back to the `base` environment.

To delete an environment, run the command below:

```bash
conda remove --name ENV_NAME --all
```



Faiss issues

if you have

```
INFO:faiss.loader:Loading faiss with AVX2 support.
INFO:faiss.loader:Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
INFO:faiss.loader:Loading faiss.
INFO:faiss.loader:Successfully loaded faiss.
```



Using Command Prompt (cmd):

1. Open Command Prompt as an administrator.

   - You can do this by searching for "Command Prompt" in the Start menu, right-clicking it, and selecting "Run as administrator."

2. Navigate to the directory where you want to create the symbolic link using the `cd` command. For example, if you want to create the link in your user folder, you can use:

   ```
   cd your_python_path/site-packages/faiss
   ```

   where the you could retrieve  your path env by typing `conda info`

3. Create the symbolic link using the `mklink` command as follows:

   ```
   mklink swigfaiss_avx2.py swigfaiss.py
   ```

   This command creates a symbolic link named `swigfaiss_avx2.py` that points to `swigfaiss.py`.

Using  Linux

```
cd your_python_path/site-packages/faiss
ln -s swigfaiss.py swigfaiss_avx2.py
```



## Contributing

------------
This repository is intended for educational purposes.

## License
-------
The ArXivChat App is released under the [MIT License](https://opensource.org/licenses/MIT).