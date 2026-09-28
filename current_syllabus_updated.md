HLTH 667M Syllabus, Dr. Helen Chen, University of Waterloo, Fall Term, 2026 1
HLTH 667M
GENERATIVE AI CORE TECHNIQUES, ARCHITECTURES AND
ALGORITHMS FOR DATA REPRESENTATION AND GENERATION
FALL TERM, 2026
Lecture: Thursdays 4:00-6:50pm, BMH 1016 and online
Dr. Helen Chen,
School of Public Health Sciences
helen.chen@uwaterloo.ca
Dr. Bing Hu,
Cheriton School of Computer Science
bing.hu@uwaterloo.ca
DESCRIPTION
This six-week modular course introduces the computational and mathematical foundations of Generative Artificial
Intelligence (GenAI), with emphasis on the models and algorithms that underpin modern generative systems. Students
learn the principles behind core generative model families, including autoencoders and variational autoencoders (VAEs),
generative adversarial networks (GANs), autoregressive models, normalizing flows, diffusion models, and transformerbased large language models (LLMs). The course also introduces retrieval-augmented generation (RAG), ontologies and
knowledge graphs as mechanisms for incorporating health domain and operational knowledge into knowledge-grounded
GenAI systems.
COURSE OVERVIEW
The course is designed primarily for graduate health informatics students and health researchers who need sufficient
technical understanding to evaluate, configure, collaborate on, and govern GenAI solutions in healthcare and public health
practices and research. The emphasis is on GenAI technical literacy: understanding how models represent data, how
generative outputs are produced, why models fail, and how health knowledge can be represented and connected to vendor
or locally developed GenAI systems. Lectures are integrated with guided tutorials using small, interpretable examples.
The lectures cover the following topics:
• Foundations of Generative AI and Representation Learning
• Latent representations, autoencoders, VAEs, GANs and synthetic health data
• Autoregressive models, tokenization, embeddings, attention and transformer architectures
• Foundation models and LLMs: pretraining, inference, decoding, adaptation and alignment
• Retrieval-augmented generation (RAG), vector retrieval, health terminologies, ontologies and knowledge graphs
• Diffusion and multimodal generative models; evaluation, failure modes, privacy, bias and health GenAI system
design
LEARNING OUTCOMES
At the end of the course, students should be able to:
1. Explain the fundamental principles of generative AI, including probability, representation learning, latent spaces,
embeddings, and neural-network-based generation.
2. Describe and compare major generative AI architectures, including VAEs, GANs, autoregressive models,
transformers/LLMs, and diffusion models, and relate their characteristics to health applications.
HLTH 667M Syllabus, Dr. Helen Chen, University of Waterloo, Fall Term, 2026 2
3. Apply approaches for agentic programming for exploratory data analysis and experiment with LLM chatbots
including RAG and the use of health terminologies, ontologies, and knowledge graphs to incorporate domain and
operational knowledge
MATERIALS AND RESOURCES
TEXTBOOK(S) REQUIRED
• No required textbook.Ge
RECOMMENDED READING AND RESOURCES
• Selected instructor-provided readings on generative modelling, transformers, LLMs, RAG, ontologies and knowledge
graphs.
• PyTorch documentation and selected tutorials for neural networks and generative models.
• Hugging Face documentation and selected educational materials on transformers, tokenization and foundation models.
• Selected health AI literature and case studies will be posted in LEARN.
TECHNICAL EXPECTATIONS
Students should have basic familiarity with Python or be prepared to work through guided notebooks. The course does not
assume advanced calculus or machine-learning theory. Mathematical expressions are used to develop conceptual
understanding of model objectives and behaviour rather than to emphasize formal derivations. Tutorials will provide
partially completed code so that students can focus on changing model parameters, observing behaviour, interpreting
results and explaining technical choices.
ASSESSMENT
There are five graded components in this six-week module:
Assessment Weight Description
Concept quizzes 15% Short quizzes assessing core concepts, terminology, architectures and
interpretation of model behaviour.
Technical Lab 1 - Representation and
Generative Models
10% Guided VAE/GAN experiment using health-related data; students
interpret latent representations, generated outputs and training
behaviour.
Technical Lab 2 - Transformers, LLMs
and Knowledge-Grounded RAG
20% Experiments with tokenization, attention/decoding and a small RAG
workflow; students examine how retrieval and structured domain
knowledge affect responses.
GenAI Project Proposal 35% Individual or small-team propose a GenAI solution addressing a
healthcare or public-health problem. Students justify model family,
knowledge source, RAG/fine-tuning strategy.
Individual Oral Defense 20% A short individual defence assessing understanding of the student's
design, code, model behaviour, limitations and technical choices.
HLTH 667M Syllabus, Dr. Helen Chen, University of Waterloo, Fall Term, 2026 3
COURSE SCHEDULE
Week Title Topics Activities
Week 0 Course Introduction Content, Assessment, Lab, Assessment Setting up computing
environment
Week 1 GenAI Foundations and
Representation Learning
Generative vs. discriminative models; probability intuition; neural-network
refresher; embeddings; latent space; autoencoders and VAEs.
Quiz 1; Lab 1 starts
Week 2 Generative Architectures and
Synthetic Health Data
VAEs, GANs, adversarial learning, training objectives, sampling, mode
collapse; overview of normalizing flows; fidelity/utility concepts.
Lab 1 due
Week 3 Autoregressive Models,
Attention and Transformers
Sequence probability; tokenization; embeddings; positional encoding; selfattention; Q/K/V; transformer architecture; next-token prediction.
Quiz 2; guided
transformer tutorial
Week 4 Foundation Models and Large
Language Models
Pretraining; foundation models; context windows; inference;
temperature/top-k/top-p; instruction tuning; SFT; PEFT/LoRA; alignment;
hallucination.
Project proposal starts
Week 5 Knowledge-Grounded GenAI:
RAG, Ontologies and
Knowledge Graphs
Chunking and embeddings; vector search; RAG; health terminologies and
concept models; ontology and knowledge-graph fundamentals; ontologyenhanced retrieval; GraphRAG; provenance.
Lab 2 due; project
consultation
Week 6 Multimodal GenAI, Evaluation of
GenAI model
Diffusion model intuition; multimodal models; RAG vs. fine-tuning;
evaluation of fidelity, utility and robustness; privacy, bias, safety.
Project
presentation/proposal
submission; oral technical
defence
Week 7 Defense
GENERATIVE AI USE IN THIS COURSE
Because GenAI is the subject of this course, students are expected to use GenAI tools critically and transparently. GenAI
may be used for brainstorming, code support, debugging, explanation and experimentation where permitted by the
instructor. Students remain responsible for understanding, validating and defending all submitted work. They must not
submit model-generated analysis or code that they cannot explain. Course-specific requirements for disclosure or
documentation of GenAI use will be provided with each assessment.
COURSE DESIGN NOTE
The course emphasizes the role of health informatics professionals not only as users of GenAI, but also as domain experts
who can translate clinical, public-health and operational knowledge into computable representations and knowledgegrounding strategies. In particular, students will learn how documents, terminologies, ontologies, knowledge graphs and
organizational knowledge can contribute to RAG-based GenAI solutions.