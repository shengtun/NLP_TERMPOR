# NLP PROJECT TEAM 4
Our project focuses on integrating a new large language model, **Qwen-7B-Chat**, into the **MiniGPT-4** framework to enhance its performance on multimodal tasks. Given the challenge of limited training data, we introduced the **RAG (Retrieval-Augmented Generation)** module to effectively reduce hallucinations during the generation process, ensuring more accurate and reliable outputs.
![image name](https://raw.githubusercontent.com/shengtun/NLP_TERMPOR/refs/heads/master/architecture.png)

Our main objectives are as follows:

1.  **Model Integration**: Successfully integrate Qwen-7B-Chat with the MiniGPT-4 framework for seamless deployment and operation.
2.  **Hallucination Mitigation**: Utilize the Multimodel RAG module with a retrieval mechanism to address hallucinations caused by insufficient training data. We will use **Wikipedia** as the retrieval source in the **RAG framework**.
4.  **Performance Enhancement**: Ensure high-quality outputs in multimodal tasks, such as image captioning and question answering, even under data constraints.
## 
**Download the related weights**
1. Image encoder weights
`wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth`
2. Q-former weights
(a)[hugging face](https://huggingface.co/google-bert/bert-base-uncased/tree/main)
(b)`wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth`

3. LLM
[Qwen/Qwen-7B-Chat · Hugging Face](https://huggingface.co/Qwen/Qwen-7B-Chat)

**Directory structure**
```text
├── cache
│   ├── ckpt
│   │   ├── bert-base-uncased
│   │   ├── blip2
│   │   │   ├── blip2_pretrained_flant5xxl.pth
│   │   ├── eva
│   │   │   ├── eva_vit_g.pth
│   │   ├── Qwen7B-chat               
```
**Demo without RAG**
`python cli_demo.py --checkpoint-path xxxxxx`
**Demo without RAG**
`python cli_demo2.py --checkpoint-path xxxxxx`
 ## Results
 1. **Result without RAG**
![image with rag](https://raw.githubusercontent.com/shengtun/NLP_TERMPOR/refs/heads/master/vis/image/greatwall_wo_rag.jpg)
 2. **Result withRAG**
 ![image with rag](https://raw.githubusercontent.com/shengtun/NLP_TERMPOR/refs/heads/master/vis/image/greatwall_w_rag.png)
