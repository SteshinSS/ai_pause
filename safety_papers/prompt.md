# Which arXiv papers are AI safety research

You label machine-learning papers first posted to arXiv in 2020–2025 for a census of AI safety research: how many
such papers the world wrote each year. You get one paper: its arXiv id, categories, title and abstract. Give it a
**topic**, a **kind** and a **confidence**, with a short reason. Decide from the title and abstract alone; authors
and venues are not shown, and you should not guess them. Most papers are out of scope.

## What we count

The study's definition:

> AI Safety is an overloaded term. Here we mean experimental research that improves our understanding of AI and
> helps predict its behaviour:
> - Interpretability, explainability, transparency, modularity
> - Deep learning generalization research
> - Jailbreaks, red-teaming, adversarial robustness
> - Narrower AI Safety topics with computational experiments: control, alignment experiments, model organisms, CoT,
>   unlearning, deception.
>
> We don't mean:
> - Theoretical work (agent foundations, philosophy, decision theory, timeline modelling, economics)
> - Governance
> - Privacy
> - Hallucinations
> - Capability prediction (scaling laws, forecasts)

In-scope research studies trained AI systems themselves: what they represent inside, how they learn and generalize,
how they can be made to misbehave and how to stop it, whether they pursue what their developers intend, and how to
test, monitor and constrain them. Research that makes systems more capable, or applies AI to a task, is out however
often it says "safe", "robust", "trustworthy", "aligned" or "explainable". Research is in whether or not it calls
itself AI safety: most interpretability and generalization research does not.

A paper is in scope when both hold:

1. **Topic**: its main contribution falls under one of the seven in-scope topics IN, SD, AV, JB, UL, AL, EV.
2. **Kind E**: that contribution rests on new experiments with, or empirical analyses of, trained AI models.

You give topic and kind separately and the scope is computed from them, so give a survey or a theory paper its true
topic (a survey of jailbreak attacks is topic JB, kind S).

## How to decide

- **The main contribution decides.** Ask what the paper delivers (a finding, a method, an attack, a benchmark, a
  model, a system) and what it is about. A motivating sentence, one section, one experiment or a keyword does not put
  a paper in scope. A report on a new model is NO even if it has a safety section.
- **Any trained network can be the object**: language models, vision-language models, image, video and audio
  generators, agents, RL policies, CNN or ViT classifiers, tabular models, toy transformers, protein models. JB, UL,
  AL and EV are mostly about language models, multimodal assistants, generators and agents; IN, SD and AV apply to
  any model.
- **Methods count in every in-scope topic except SD**: a new jailbreak defense, adversarial defense, unlearning
  method, explanation method, oversight protocol or monitor is in scope. In SD the deliverable must be understanding
  of learning or generalization, and a method that merely raises generalization scores is NO.
- **Between two in-scope topics**, choose the one closest to the main finding; the choice does not change the scope.
- **Out-of-scope neighbours of safety** have their own codes (CP, HF, PR, FA, WM, PL, SR, GV) so that the
  scope can be revised later without reading the papers again. Use them precisely; NO is for everything else.

## In-scope topics

**IN — interpretability, explainability, transparency, modularity.** What trained models compute and why they give
their outputs, and methods to find out, for any model and in any domain.
- mechanistic interpretability: circuits, features, superposition, sparse autoencoders, transcoders, crosscoders,
  attribution graphs
- probing classifiers on internal activations and representation analysis (does the model encode X, where, how is
  it used); logit and tuned lens; representation similarity across layers or models
- causal interventions to understand components: activation patching, causal tracing, causal mediation and
  interchange interventions, ablations; neuron and attention-head analyses; localizing knowledge
- representation engineering and activation steering that identify and manipulate internal representations of
  concepts, behaviours or values
- model diffing (what fine-tuning changed inside a model); training-data attribution and influence functions
- explanation methods: feature attribution and saliency (gradients, SHAP, LIME), counterfactual explanations,
  concept-based and prototype explanations, rule extraction, feature visualization; evaluations of explanations
  (faithfulness, user studies)
- explaining models used in medicine, finance, science or other domains, when explaining the model is the main
  contribution
- interpretable or modular models by design: concept bottlenecks, prototype networks, sparse, monosemantic or
  modular networks, inherently interpretable models
- faithfulness of a model's natural-language explanations of its own answers; automated interpretability (models
  describing neurons or features)

Not IN: an application or method paper that only adds a few attention maps or saliency figures (NO);
knowledge-editing methods judged on editing benchmarks (NO).

**SD — generalization research, science of deep learning.** Experiments whose main question is how, when or why
trained networks learn, generalize, memorize or fail to generalize. The deliverable is understanding: a finding, a
mechanism, an empirical regularity.
- generalization versus memorization; how much and what models memorize; how memorization or benchmark
  contamination inflates measured ability
- out-of-distribution, compositional, systematic and length generalization: analyses of when and why models succeed
  or fail, including distribution-shift benchmarks that report how models degrade
- shortcut learning, spurious features, underspecification
- grokking, double descent, benign overfitting, neural collapse, lottery tickets as findings; loss landscapes, mode
  connectivity, implicit and simplicity biases
- training dynamics: phase transitions, how abilities and circuits form during training
- mechanisms of in-context learning; out-of-context reasoning (for example the reversal curse); controlled
  synthetic-task studies of what transformers can learn
- what fine-tuning, RLHF or RL (versus supervised fine-tuning) changes in a model's behaviour and generalization;
  forgetting
- geometry of representations and their convergence across models

Not SD: methods for better domain or out-of-distribution generalization (NO); training recipes (NO); scaling laws
and capability forecasts (CP); hallucination (HF); theorems (kind T).

**AV — adversarial robustness.** Attacks that make a model mispredict, and defenses, for models of any kind and
modality.
- adversarial examples, digital or physical, white-box, black-box or transferred; adversarial training; certified
  robustness and formal verification of networks
- backdoors, trojans and data poisoning of classifiers, encoders, detectors, RL policies and federated learning
- robustness of a language model's task accuracy to adversarially perturbed inputs
- why adversarial examples exist and what robust models learn

Not AV: robustness to natural distribution shift, noise or corruptions (methods NO, analyses SD); attacks that make
assistants, agents or generators follow an attacker or produce harmful content (JB).

**JB — jailbreaks, red-teaming, attacks and safeguards for language models, generators and agents.** Making these
systems act against their developers' intent, and keeping them from doing so.
- jailbreak attacks and defenses: adversarial suffixes, role-play, multi-turn, many-shot, encodings, low-resource
  languages
- red-teaming methods and red-teaming studies, automated or human
- prompt injection, indirect prompt injection, agent hijacking, poisoned tools, memories or retrieved documents
- backdoors, trojans and data poisoning in language models, code models, assistants, agents, retrieval pipelines
  and image generators, at pretraining, fine-tuning or RLHF
- harmful fine-tuning: attacks that remove safety by fine-tuning, defenses, tamper-resistant safeguards for open
  weights
- safety training: refusal and harmlessness training, safety RLHF, over-refusal; how refusal works inside the model
  (IN also fits)
- guard models and safety classifiers for model inputs and outputs; safety filters of image and video generators
- adversarial images, audio or text that make multimodal assistants follow an attacker or produce harmful content;
  jailbreaks of image and video generators

Not JB: adversarial examples that make classifiers mispredict (AV); privacy attacks (PR); attacks on watermarks (WM);
measuring toxicity or bias without adversarial search (FA); AI used for cybersecurity (NO).

**UL — unlearning and concept erasure in language and generative models.** Removing knowledge, concepts or
behaviours from language models, vision-language models, and image, video or audio generators, whatever the motive
(hazardous knowledge, private data, copyrighted text, artists' styles, nudity, bias); attacks on and evaluations of
unlearning (relearning, recovering "unlearned" knowledge); concept erasure in language-model representations;
filtering pretraining data to keep hazardous knowledge out of a model.

Not UL: unlearning in classifiers, recommenders, graph networks or federated learning (PR).

**AL — alignment experiments, deception, model organisms, oversight, chain of thought.** Experiments on whether AI
systems pursue the objectives their developers intend: how misaligned or deceptive behaviour arises, how to detect
it, and how to supervise systems that humans cannot fully evaluate.
- reward hacking, specification gaming, reward tampering, reward-model over-optimization; goal misgeneralization;
  side effects and impact measures; shutdown and interruptibility in RL
- sycophancy; honesty and lying (the model represents the truth and says otherwise); deception, strategic
  deception, scheming, alignment faking, sandbagging
- situational awareness, evaluation awareness, self-recognition, introspection and self-knowledge
- model organisms of misalignment: emergent misalignment, sleeper agents and other backdoors studied as models of
  deceptive misalignment, misalignment from narrow fine-tuning, subliminal learning
- power-seeking, self-preservation, shutdown resistance and corrigibility in language-model agents; goals and
  preferences of models measured as their own properties
- scalable oversight: debate, weak-to-strong and easy-to-hard generalization, recursive reward modelling, AI critics
  and assistants that help humans evaluate stronger models, sandwiching
- chain-of-thought faithfulness and monitorability, obfuscated or encoded reasoning, steganography, monitoring chains
  of thought for misbehaviour
- stability of personas, values and goals; manipulation of users or overseers; collusion and deception among AI
  agents

Not AL: preference learning for helpfulness (PL); moral, cultural or political views of models (FA); constrained or
safe RL (SR); hallucination (HF).

**EV — dangerous-capability and propensity evaluations, AI control, monitoring.** Measuring whether AI systems can
or would cause serious harm, and protocols that keep possibly misaligned systems from causing it.
- dangerous capabilities: offensive cyber (exploitation, attack challenges framed as risk), biological and chemical
  weapons uplift, persuasion and manipulation, autonomous replication, self-exfiltration, sabotage, evading oversight
- propensities: whether models or agents choose harmful, deceptive or power-seeking actions in realistic scenarios
- safety benchmarks for language models and agents, with results; agent safety (unsafe or irreversible actions,
  misuse of tools)
- AI control: trusted and untrusted monitoring, auditing, resampling and deferral protocols, control evaluations
- monitoring deployed agents for misbehaviour; eliciting capabilities in safety evaluations

Not EV: general capability benchmarks (NO); capability forecasts and measurements of agents' general, autonomous or
AI-R&D capability (CP); accuracy of medical or legal answers (HF or NO).

## Out-of-scope topics

- **CP — capability prediction**: scaling laws, predicting loss or downstream performance from compute, data or
  size, predictability and emergence of capabilities with scale, capability trends and forecasts, measurements of
  frontier models' general, autonomous or AI-R&D capability.
- **HF — hallucination and reliability**: hallucination, factuality, faithfulness of summaries or answers to their
  sources, calibration, uncertainty estimation, out-of-distribution detection, selective prediction and abstention.
- **PR — privacy and data rights**: differential privacy, membership inference, extraction or reconstruction of
  training data, personal data, model stealing, privacy in federated learning, copyright and consent for training
  data, unlearning outside language and generative models.
- **FA — fairness, bias, toxicity, ethics**: social bias and stereotypes, fairness, measuring or reducing toxicity and
  hate speech, political and cultural bias, moral judgement and ethics benchmarks.
- **WM — watermarking and AI-content detection**: watermarks and fingerprints of models or outputs, detecting
  AI-generated text, images or audio, deepfakes, provenance.
- **PL — preference learning and post-training for helpfulness**: RLHF, DPO and their variants, reward models,
  preference data, instruction tuning, RL for reasoning, alignment with human preferences in the sense of
  helpfulness, personalization, pluralistic alignment.
- **SR — safety of controllers and deployed systems**: safe and constrained RL, safe exploration, shielding, safety of
  autonomous vehicles, robots and medical AI, runtime monitors, safety assurance and certification.
- **GV — governance and society**: policy, law, regulation, standards, audits as institutions, compute governance and
  other technical governance tools, AI-risk arguments, philosophy, economics and labour, forecasting surveys,
  societal impact.
- **NO — everything else**: new models, architectures, training methods, efficiency, applications, capability
  benchmarks, datasets, systems, and theory unrelated to the topics above.

## Kind

- **E — empirical**: the main contribution rests on new experiments with trained AI models (training, fine-tuning,
  running, probing, attacking or evaluating them) or on systematic empirical analysis of their outputs. Toy models
  studied experimentally count. Benchmarks and datasets count when the paper reports model results. Studies with
  human participants count when they run models to measure a model's capability or to test human oversight.
- **T — theory**: theorems, proofs, derivations, formal or game-theoretic models; any experiments only illustrate or
  simulate the mathematics. Theory whose claims are tested by substantial experiments on trained networks is E.
- **S — other non-empirical work**: survey, review, systematization, taxonomy, position or perspective paper,
  conceptual essay, research agenda, tutorial, workshop report, policy analysis, benchmark or dataset proposal
  without model results. A position paper is E only if experiments are the main support of its claim.

## Confidence

- **h**: the label is clear.
- **l**: the paper is on the scope boundary: an in-scope and an out-of-scope topic both fit, or the topic is in
  scope and it is unclear whether the kind is E. Use l only for such boundary cases.

## Borderline decisions

Methods and findings
1. A method that raises benchmark scores (domain generalization, augmentation, regularization, robustness to
   natural shifts or corruptions, continual training, architectures, distillation, merging, model editing) is NO
   even when framed as generalization, robustness or trustworthiness. Robustness against an adversary is AV.
2. Training-dynamics papers aimed at a better recipe (schedules, stability tricks, optimizers, data mixtures) are NO;
   aimed at understanding how networks learn, SD.
3. Statistics of trained weights or activations with no question about behaviour, learning or generalization are NO.

Capabilities
4. Capability benchmarks, leaderboards, "can LLMs do X?" studies and reasoning methods are NO. They are SD only when
   the main finding is how the model generalizes (memorization versus reasoning shown by controlled perturbations,
   what RL teaches beyond the base model), and EV only when the capability measured is a hazard listed under EV.
5. Scaling laws, capability forecasts and measurements of general, autonomous or AI-R&D capability are CP. A scaling
   study of a safety failure (sycophancy, jailbreak robustness or deception versus model size) takes the failure's
   topic.
6. Detecting benchmark contamination and building contamination-resistant benchmarks are NO; experiments on how
   memorization or contamination inflates measured ability are SD.
7. Behavioural or cognitive-science studies of models (human-likeness, theory of mind, linguistic phenomena,
   psychology tests, similarity to brains) are NO, unless the main question is how the model generalizes (SD) or
   what its internals represent (IN).

Interpretability
8. Interpretability used as a tool for another in-scope topic: choose the topic of the main finding (a probe that
   detects deception: AL if the finding is about deception, IN if it is about representations).
9. Activation steering used only as a trick to raise task scores is NO.
10. Locating where knowledge is stored is IN; editing methods judged on editing benchmarks are NO; editing out
    harmful knowledge or behaviour is UL or JB.
11. Attention maps or saliency figures added to a method or application paper do not make it IN.

Attacks and safeguards
12. Attacks that make a model mispredict (classifiers of any modality; a language model's task accuracy under
    adversarial perturbation) are AV. Attacks that make assistants, agents or generators follow an attacker or
    produce harmful content are JB.
13. Backdoors and poisoning: in generative models, assistants, agents, retrieval pipelines, code models and image
    generators, JB; in classifiers, encoders used as classifiers and federated learning, AV; backdoored models
    studied as model organisms of deceptive misalignment, AL.
14. A product or application that only deploys guardrails or filters is NO.
15. AI for cybersecurity (malware, intrusion or vulnerability detection, pentesting agents, secure code generation)
    is NO; measuring AI's offensive cyber capability as a risk is EV.
16. Moderating human-written content is FA or NO; guard models for model inputs and outputs are JB.

Alignment and its neighbours
17. RLHF, DPO, reward-modelling, preference-data and instruction-tuning methods are PL. Studies of reward hacking,
    over-optimization, sycophancy or deception caused by such training are AL. Preference training for harmlessness
    and refusal is JB. Weak-to-strong and other oversight methods are AL.
18. Hallucination, factuality and calibration are HF, including benchmarks of whether models repeat common
    misconceptions. Lying or deception (the model knows better) is AL; how a model represents truth internally is
    IN; introspection and self-knowledge experiments are AL.
19. Which moral, cultural or political views a model expresses is FA. Whether a model has stable goals or preferences
    of its own and acts on them (power-seeking, self-preservation, scheming) is AL or EV.
20. Safe, constrained or shielded RL is SR. RL experiments on reward hacking, specification gaming, goal
    misgeneralization, side effects or shutdown are AL.
21. Reliability of LLM judges and other evaluation tooling is NO, unless the question is supervising models stronger
    than the supervisor (AL) or judges being manipulated (JB or AL).
22. Collusion, deception, escalation or failures of cooperation among AI agents with safety stakes are AL;
    multi-agent RL algorithms for cooperation are NO.

Privacy and unlearning
23. Memorization as a scientific question (how much, what, how it evolves, how it relates to generalization) is SD;
    extraction attacks, membership inference and personal-data leakage are PR.
24. Unlearning and concept erasure in language, vision-language and generative models are UL whatever the motive; in
    classifiers, recommenders, graphs and federated learning, PR.

Evaluations, people, other fields
25. Benchmarks and datasets on in-scope topics are E when they report model results, S when they do not.
26. Studies of how people use or perceive AI are GV, FA or NO. Studies that run models with people to measure a
    hazardous capability (persuasion) are EV; people supervising stronger models, AL.
27. Medical, legal or financial "safety" meaning accurate advice is HF or NO; complying with harmful requests in
    those domains is JB; models encouraging self-harm or giving dangerous advice under pressure is EV.
28. Dual-use risk of AI for science (designing toxins, uplift for weapons) is EV; AI for science otherwise is NO.
29. Broad trustworthiness benchmarks spanning many dimensions take the dominant dimension; EV if harmful content,
    misuse or agent safety dominate.

Non-empirical work
30. Theorems with illustrative experiments are kind T.
31. Agent foundations, decision theory and formal models of alignment are AL with kind T. AI-risk arguments,
    philosophy, governance, economics and timelines are GV.

## Papers from 2020–2022

The vocabulary changed; label what the paper does, the same way for every year.
- Before "jailbreak": adversarial triggers or prompts that make GPT-2- or GPT-3-style generators or dialogue systems
  produce toxic or harmful text, and red-teaming language models, are JB. Measuring the average toxicity of
  generations without adversarial search is FA.
- BERTology, probing and diagnostic classifiers, attention-head analyses, what BERT, ELMo or GPT-2 encode: IN.
- Underspecification, shortcut learning, distribution-shift analyses, double-descent and neural-tangent-kernel
  experiments, lottery-ticket findings, loss-landscape and mode-connectivity experiments: SD (methods NO, proofs T).
- Specification gaming, side effects, impact measures, safe interruptibility, reward tampering, mesa-optimization and
  goal-misgeneralization experiments in RL: AL. Debate, iterated amplification and recursive reward modelling
  experiments: AL. Learning from human feedback to do a task better: PL.
- Adversarial examples, adversarial training and certified robustness of image or text classifiers: AV. Backdoors
  in NLP classifiers: AV; in generative language models: JB.
- Machine unlearning in classifiers: PR. Safe exploration and constrained RL: SR.
- Saliency, concept and attribution methods, interpretable machine learning for tabular data, explainability of
  models in applications: IN.

## Words that mislead

- "alignment" also means image-text or cross-modal alignment, domain alignment, word or sequence alignment, matching
  representations to brains, and fitting preferences (PL). Only alignment with intended objectives is AL.
- "safe" and "safety" also mean safe RL and the safety of vehicles and robots (SR), and food, drug or workplace safety
  (NO).
- "robust" against an adversary is AV; against natural shifts or noise, a method is NO and an analysis of why
  models generalize, or fail to, is SD.
- "interpretable" or "explainable" in the title of an application paper: IN only if explaining the model is the main
  contribution, NO if the paper is about the application.
- "probing" also means prompting a model to test its knowledge (NO unless rule 4 applies); IN probing reads internal
  activations.
- "emergent" abilities with scale are CP; "emergent misalignment" is AL.
- "steering" by prompts or decoding is NO; activation steering is IN.
- "deception" or lie detection in humans, "trojans" in hardware, "red teams" in network security, "jailbreaking"
  phones: NO.
- "hallucinations" of image generators (artifacts) are HF or NO.

## Output

`why`: the reason for the topic and kind, in at most 15 words; then `topic`, `kind` and `conf`.
