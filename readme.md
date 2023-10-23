# How to chat with a PDF by using LLM in Streamlit

Hello, today we are going to build a simple application that where we load  a PDF  

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Step 1. Installation of Conda

First you need to install anaconda at this [link](https://www.anaconda.com/products/individual)

![image-20231021101716486](assets/images/posts/readme/image-20231021101716486.png)

in this location **C:\Anaconda3** , then you, check that your terminal , recognize **conda**

```
C:\conda --version
conda 23.1.0
```

## Step 2. Environment creation

The environments supported that I will consider is Python 3.10,

I will create an environment called **chatpdf**, but you can put the name that you like.

```
conda create -n chatpdf python==3.9.4
```

then we activate

```
conda activate chatpdf
```

then in your terminal type the following commands:

```
conda install ipykernel notebook
```

then

```
python -m ipykernel install --user --name chatpdf --display-name "Python (chatpdf)"
```



Clone the repository to your local machine.



Install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

1. Obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys) and add it to the `.env` file in the project directory.

for example

```
echo OPENAI_API_KEY=sk-hJAldsflhsdflsdhflsldfhsdlfhlsdfhXLWZygAKn > .env
```

and if you will use [HuggingFace](https://huggingface.co/settings/tokens) models   you also add

```
echo HUGGINGFACEHUB_API_TOKEN=hf_xqPNsdlfjlsdjflsdjfjfkpAgVk >>.env
```



Support GPU(optionall)

If you have high end GPU  NVIDIA card you can install the pytorch capability with cuda

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```



## How to run you program

-----
To use the Chat PDF  App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.



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
This repository is intended for educational purposes and does not accept further contributions. It serves as supporting material for a YouTube tutorial that demonstrates how to build this project. Feel free to utilize and enhance the app based on your own requirements.

## License
-------
The ChatPDF App is released under the [MIT License](https://opensource.org/licenses/MIT).