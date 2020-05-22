# Knowldege-Grounded-Conversation
﻿# Paper-Reading
Paper reading list in natural language processing.


- [Paper-Reading](#paper-reading)
  - [Deep Learning in NLP](#deep-learning-in-nlp)
  - [Pre-trained Language Models](#pre-trained-language-models)
  - [Dialogue System](#dialogue-system)
  - [Knowledge Representation & Reasoning](#knowledge-representation-and-reasoning)
  - [Text Summarization](#text-summarization)
  - [Topic Modeling](#topic-modeling)
  - [Machine Translation](#machine-translation)
  - [Question Answering](#question-answering)
  - [Reading Comprehension](#reading-comprehension)
  - [Image Captioning](#image-captioning)

***

* **Pretrained Seq2Seq**: "Unsupervised Pretraining for Sequence to Sequence Learning". EMNLP(2017) [[PDF]](https://www.aclweb.org/anthology/D17-1039)
* **Multi-task Learning**: "An Overview of Multi-Task Learning in Deep Neural Networks". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1706.05098.pdf)
* **Gradient Descent**: "An Overview of Gradient Descent Optimization Algorithms". arXiv(2016) [[PDF]](https://arxiv.org/pdf/1609.04747.pdf)

## Dialogue System

### PTMs for Dialogue
* **ToD-BERT**: "ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2004.06871.pdf) [[code]](https://github.com/jasonwu0731/ToD-BERT) :star::star::star::star:
* **PLATO**: "PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable". ACL(2020) [[PDF]](https://arxiv.org/pdf/1910.07931.pdf) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO) :star::star::star::star:
* **Guyu**: "An Empirical Investigation of Pre-Trained Transformer Language Models for Open-Domain Dialogue Generation". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2003.04195.pdf) [[code]](https://github.com/lipiji/Guyu) :star::star::star:


### Knowledge-driven Conversation
* **DuRecDial**: "Towards Conversational Recommendation over Multi-Type Dialogs". ACL(2020) [[PDF]](https://arxiv.org/pdf/2005.03954.pdf) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-DuRecDial) :star::star::star:
* **KdConv**: "KdConv: A Chinese Multi-domain Dialogue Dataset Towards Multi-turn Knowledge-driven Conversation". ACL(2020) [[PDF]](https://arxiv.org/pdf/2004.04100.pdf) [[data]](https://github.com/thu-coai/KdConv) :star::star::star:
* **DuConv**: "Proactive Human-Machine Conversation with Explicit Conversation Goals". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1369) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-DuConv) :star::star::star:
* **KBRD**: "Towards Knowledge-Based Recommender Dialog System". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1189.pdf) [[code]](https://github.com/THUDM/KBRD) :star::star::star::star:
* **ReDial**: "Towards Deep Conversational Recommendations". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/8180-towards-deep-conversational-recommendations.pdf) [[data]](https://github.com/ReDialData/website) :star::star:
* **Dual Fusion**: "Smarter Response with Proactive Suggestion: A New Generative Neural Conversation Paradigm". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0629.pdf) :star::star::star:

### Open-domain Dialogue
* **RefNet**: "RefNet: A Reference-aware Network for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.06449.pdf) [[code]](https://github.com/ChuanMeng/RefNet) :star::star::star:
* **GLKS**: "Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.09528.pdf) [[code]](https://github.com/PengjieRen/GLKS) :star::star::star:
* **HDSA**: "Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1360) [[code]](https://github.com/wenhuchen/HDSA-Dialog) :star::star::star::star:
* **PostKS**: "Learning to Select Knowledge for Response Generation in Dialog Systems". IJCAI(2019) [[PDF]](https://www.ijcai.org/proceedings/2019/0706.pdf) :star::star:
* **Two-Stage-Transformer**: "Wizard of Wikipedia: Knowledge-Powered Conversational agents". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1811.01241.pdf) :star::star:
* **CAS**: "Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1124) [[code]](https://github.com/jcyk/Skeleton-to-Response) :star::star::star:
* **Edit-N-Rerank**: "Response Generation by Context-aware Prototype Editing". AAAI(2019) [[PDF]](https://arxiv.org/pdf/1806.07042.pdf) [[code]](https://github.com/MarkWuNLP/ResponseEdit) :star::star::star:
* **HVMN**: "Hierarchical Variational Memory Network for Dialogue Generation". WWW(2018) [[PDF]](https://dl.acm.org/citation.cfm?doid=3178876.3186077) [[code]](https://github.com/chenhongshen/HVMN) :star::star::star:
* **XiaoIce**: "The Design and Implementation of XiaoIce, an Empathetic Social Chatbot". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1812.08989.pdf) :star::star::star:
* **D2A**: "Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7558-dialog-to-action-conversational-question-answering-over-a-large-scale-knowledge-base.pdf) [[code]](https://github.com/guoday/Dialog-to-Action) :star::star::star:
* **DAIM**: "Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7452-generating-informative-and-diverse-conversational-responses-via-adversarial-information-maximization.pdf) :star::star:
* **MTask**: "A Knowledge-Grounded Neural Conversation Model". AAAI(2018)  [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16710/16057) :star:
* **GenDS**: "Flexible End-to-End Dialogue System for Knowledge Grounded Conversation". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1709.04264.pdf) :star::star:
* **Time-Decay-SLU**: "How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1194) [[code]](https://github.com/MiuLab/Time-Decay-SLU) :star::star::star::star:
* **REASON**: "Dialog Generation Using Multi-turn Reasoning Neural Networks". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1186) :star::star::star:
* **STD/HTD**: "Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1204) [[code]](https://github.com/victorywys/Learning2Ask_TypedDecoder) :star::star::star:
* **CSF**: "Generating Informative Responses with Controlled Sentence Function". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1139) [[code]](https://github.com/kepei1106/SentenceFunction) :star::star::star:
* **NKD**: "Knowledge Diffusion for Neural Dialogue Generation". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1138) [[data]](https://github.com/liushuman/neural-knowledge-diffusion) :star::star:
* **DAWnet**: "Chat More: Deepening and Widening the Chatting Topic via A Deep Model". SIGIR(2018) [[PDF]](https://dl.acm.org/citation.cfm?doid=3209978.3210061) [[code]](https://sigirdawnet.wixsite.com/dawnet) :star::star::star:
* **ZSDG**: "Zero-Shot Dialog Generation with Cross-Domain Latent Actions". SIGDIAL(2018) [[PDF]](https://www.aclweb.org/anthology/W18-5001) [[code]](https://github.com/snakeztc/NeuralDialog-ZSDG) :star::star::star:
* **DUA**: "Modeling Multi-turn Conversation with Deep Utterance Aggregation". COLING(2018) [[PDF]](https://www.aclweb.org/anthology/C18-1317) [[code]](https://github.com/cooelf/DeepUtteranceAggregation) :star::star:
* **Data-Aug**: "Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding". COLING(2018) [[PDF]](https://www.aclweb.org/anthology/C18-1105) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) :star::star:
* **DC-MMI**: "Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1431) [[code]](https://github.com/abaheti95/DC-NeuralConversation) :star::star:
* **cVAE-XGate/CGate**: "Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1432) [[code]](https://github.com/XinnuoXu/CVAE_Dial) :star::star::star:
* **DAM**: "Multi-Turn Response Selection for Chatbots with Deep Attention
Matching Network". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1103) [[code]](https://github.com/baidu/Dialogue/tree/master/DAM) :star::star::star::star:
* **SMN**: "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1046)  [[code]](https://github.com/MarkWuNLP/MultiTurnResponseSelection) :star::star::star::star:
* **MMI**: "A Diversity-Promoting Objective Function for Neural Conversation Models". NAACL-HLT(2016)  [[PDF]](https://www.aclweb.org/anthology/N16-1014) [[code]](https://github.com/jiweil/Neural-Dialogue-Generation) :star::star:
* **RL-Dialogue**: "Deep Reinforcement Learning for Dialogue Generation". EMNLP(2016) [[PDF]](https://www.aclweb.org/anthology/D16-1127) :star:
* **TA-Seq2Seq**: "Topic Aware Neural Response Generation". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14563/14260) [[code]](https://github.com/LynetteXing1991/TA-Seq2Seq) :star::star:
* **MA**: "Mechanism-Aware Neural Machine for Dialogue Response Generation". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14471/14267) :star::star:
* **HRED**: "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models". AAAI(2016) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11957/12160) [[code]](https://github.com/julianser/hed-dlg) :star::star:
* **VHRED**: "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14567/14219) [[code]](https://github.com/julianser/hed-dlg-truncated) :star::star:
* **CVAE/KgCVAE**: "Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1061) [[code]](https://github.com/snakeztc/NeuralDialog-CVAE) :star::star::star:
* **ERM**: "Elastic Responding Machine for Dialog Generation with Dynamically Mechanism Selecting". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16316/16134) :star::star:
* **Tri-LSTM**: "Augmenting End-to-End Dialogue Systems With Commonsense Knowledge". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16573/16030) :star::star:
* **CCM**: "Commonsense Knowledge Aware Conversation Generation with Graph Attention". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0643.pdf) [[code]](https://github.com/tuxchow/ccm) :star::star::star::star::star:
* **Retrieval+multi-seq2seq**: "An Ensemble of Retrieval-Based and Generation-Based Human-Computer Conversation Systems". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0609.pdf) :star::star::star:

