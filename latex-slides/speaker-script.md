# HLTH 667M · Lab 2 deck · Speaker script

One entry per slide of `lab2-slides.pdf`, in deck order. The number in each heading is the PDF page.

- **Say** is wording you can speak as written. Each entry runs 30 to 60 seconds unless a time is given.
- **Point at** names the part of the slide to indicate while you speak.
- **Notebook** marks a hand-off: open the named section, run it, then return to the deck.
- **Ask** is a question for the room. Take one or two answers and move on.

`python list_slides.py --check` confirms that this script still matches the deck after an edit.

## The story in one paragraph

Tutorial 1 teaches how to judge a result: know the data, split first, put every learning step in a pipeline, compare with a baseline, evaluate once, and treat a surprisingly good number as a leak until shown otherwise. Tutorial 2 teaches how a language model produces a result: text becomes tokens, tokens become vectors, attention lets tokens read each other, a mask decides who may read whom and so defines the training objective, decoding turns probabilities into words, and retrieval decides which text reaches the model. The two tutorials meet at the end. A RAG chatbot is a system that produces fluent answers, and the habits from Tutorial 1 are how you decide whether to believe them. Lab 2 asks students to build that chatbot and judge it.

## Time plan

| Block | Slides | Talk time | Notebook time |
|---|---|---:|---:|
| Opening | 1–3 | 4 min | |
| Tutorial 1 · framing | 4–6 | 3 min | |
| Tutorial 1 · Part 0 data checks | 7–16 | 10 min | 25 min |
| Tutorial 1 · Part 1A classification | 17–24 | 9 min | 30 min |
| Tutorial 1 · Part 1B regression and wrap-up | 25–34 | 10 min | 30 min |
| Tutorial 2 · framing | 35–36 | 2 min | |
| Tutorial 2 · Part 0 tokenization | 37–42 | 6 min | 20 min |
| Tutorial 2 · Part 1 Word2Vec | 43–50 | 8 min | 25 min |
| Tutorial 2 · Part 2 attention and masks | 51–75 | 25 min | 30 min |
| Decoding (slides only) | 76–81 | 7 min | |
| Tutorial 2 · Part 3 RAG, GraphRAG and chat app | 82–104 | 27 min | 50 min |
| Close and Lab 2 | 105–106 | 4 min | |

---

# Opening

### 1 · Title

**Say:** Welcome to Lab 2. Today has two tutorials that answer two questions. The first question is how you judge a result that a model gives you. The second is how a language model produces a result in the first place. By the end you will have built a small system that answers questions from documents, and you will have the tools to decide how far to trust it.

### 2 · Seven notebooks, one path

**Say:** Here is the whole path. Three notebooks on machine-learning pipelines, then four on language models. Each tutorial numbers its notebooks from zero, so when I say Part 0 I will name the tutorial as well. Read the line at the bottom: Tutorial 1 teaches you to judge a result, and Tutorial 2 teaches you how a language model produces one. Keep that pairing in mind, because the last notebook needs both halves.

**Point at:** the two braces, then the takeaway.

### 3 · Ground rules for both tutorials

**Say:** Three rules hold all day. We use teaching data only: a synthetic hospital CSV, two benchmark datasets that ship with scikit-learn, and fictional clinic documents. Patient data, personal health information and API keys stay out of every notebook, prompt, screenshot and log. And we study mechanisms. A good number today shows that a method works the way we said it does. Using a model in care needs its own evidence, and we will say what that evidence is.

---

# Tutorial 1 · Machine-learning pipelines

### 4 · Tutorial 1 — Machine-learning pipelines

**Say:** Tutorial 1. The subtitle is the whole workflow in one line: audit, split, build, compare, evaluate once, and state the limits of the number. Everything in the next thirty slides is one of those verbs.

### 5 · Two kinds of prediction

**Say:** Supervised learning predicts a target. When the target is a category, such as benign or malignant, the task is classification. When the target is a number, such as a disease-progression score of 152, the task is regression. This choice comes first because it decides the model family, the baseline you compare against, how you split the data, and which metrics make sense.

**Ask:** Length of stay in days: which kind is it? What about "readmitted within 30 days"?

### 6 · One workflow, every time

**Say:** Both kinds of prediction follow the same seven steps. Audit the data. Define the target and the moment of prediction. Split before fitting anything. Put preprocessing and the model in one pipeline. Compare candidates against a baseline. Evaluate once on the held-out test set. State the limits of the result. The green box and the red box are the two steps people skip, and they are the two that protect you most.

**Transition:** We start with the first box, the audit.

## Part 0 · Healthcare data processing

### 7 · Part 0 · Healthcare data processing — Know the data before modelling it

**Say:** The guiding question for this notebook: what should you check in any health dataset before anyone builds a model on it? We will use the course CSV, which you met in Lab 1, as the worked example.

### 8 · Five checks before any model

**Say:** Five checks, in order. Structure: types, missing values, duplicates. Identifiers: columns that are unique to each row. Odd values: anything outside a plausible range. Relationships: do patterns you expect clinically show up? And timing: when does each value become known? Each check ends in one of two things, a decision you write down or a question for whoever owns the data.

### 9 · Checks 1 and 2: structure and identifiers

**Say:** The course CSV has 55,500 rows. It has zero missing values, 534 exact duplicate rows, and a Name column where 90 percent of values are unique. Remove the duplicates before you split, so that the same row cannot land in both training and test. Keep Name out of the features, because a model given an identifier memorises rows. And notice the zero. Real hospital data always has gaps, so a file with none is telling you it was generated.

**Notebook:** Tutorial 1 Part 0, sections 2 and 3.

### 10 · Check 3: treat an odd value as a question

**Say:** The file has 106 bills at or below zero. The tempting move is to delete them. Look at the four explanations on the right: a refund, a reversed charge, an adjustment, or an entry error. Each one calls for a different response. A refund is real information and should stay. An entry error should be corrected at the source. So list the odd rows, write down every explanation that fits, and keep the rows until the data owner confirms one.

### 11 · Plot the same two variables

**Say:** This is the notebook's own output for billing amount and length of stay. A summary table told us the minimum bill was negative. The plot adds how many values sit near that minimum and whether the overall shape is believable. Billing is almost perfectly flat from zero to fifty thousand dollars, and length of stay is flat from one to thirty days. Real billing and real stays are heavily skewed, with many small values and a long tail. That flat shape is a second sign that this file is synthetic.

**Notebook:** Tutorial 1 Part 0, section 4.

### 12 · Check 4: do known relationships appear?

**Say:** Pick a relationship you would bet on clinically and plot it. We expect emergency admissions to cost more than elective ones. In this file the mean bill is 25.6, 25.5 and 25.5 thousand dollars across the three admission types, and test results split into exact thirds. A flat result means the signal is weak or the data were generated. Either way the conclusion is the same: use this file to practise the workflow, and benchmark on other data.

### 13 · Billing by admission type

**Say:** Here is the same comparison as the notebook draws it, with the full distribution in each box. The three boxes are nearly identical in median, spread and range. A model asked to predict billing from admission type has nothing to learn here, and that is why the next two notebooks switch to datasets with real signal.

**Notebook:** Tutorial 1 Part 0, section 5.

### 14 · Check 5: prediction time decides what is usable

**Say:** This is the check that matters most in health data. Draw the patient's timeline and mark the moment the prediction is made. Here we predict at admission. Age, gender, condition and admission type exist at that moment, so they are usable. Discharge date, length of stay and the bill exist only later. Medication sits in between, and the file does not say when it was recorded. A column is usable only if its value exists at the moment of prediction.

**Ask:** If we moved the prediction to the day of discharge, which boxes change colour?

### 15 · Give every column a role

**Say:** Turn that timeline into a table with one row per column and a role for each. Available columns are used once their timing is confirmed. Columns with unclear timing become a question for the data owner. Future values and outcomes are excluded, and so are identifiers. Operational columns such as doctor, hospital and room are excluded, or kept with a clear statement that the model then depends on one site. Write this table before you choose features. In the student notebook, filling it in is your first exercise.

**Notebook:** Tutorial 1 Part 0, section 6.

### 16 · Leakage takes two forms

**Say:** Leakage means the model saw information at training time that it will lack when it is used. It comes in two forms. The first is a value from the future, such as discharge date in a prediction made at admission. The timeline on the last slide guards against that. The second is quieter: a preprocessing step, such as scaling, imputing or feature selection, that was fitted on every row before the split. Both forms run without any error message, and both make your result look better. The next two notebooks build the defence against the second form.

## Part 1A · Classification

### 17 · Part 1A · Classification — Predicting a category

**Say:** The question for this notebook: can tumour measurements classify a recorded diagnosis, and how would we know? The second half of that question is the real subject. We use the breast cancer dataset that ships with scikit-learn: 569 cases, 30 numeric measurements, and a label of benign or malignant.

### 18 · Split first. Evaluate once.

**Say:** Set aside 20 percent of the data as a test set before doing anything else, and leave it alone. All model building happens in the other 80 percent, using five-fold cross-validation. Each row of this picture is one fold: fit on the blue blocks and validate on the gold one, then rotate. Cross-validation tells you which candidate to pick. The test set grades that one choice, once. A test set that you look at twice has become part of your training process.

### 19 · Everything that learns goes inside the pipeline

**Say:** Imputing with a median learns a median. Scaling learns a mean and a standard deviation. One-hot encoding learns a list of categories. Each of those is a small model, so each belongs inside the dashed box with the classifier. scikit-learn then refits the whole box inside every training fold, and the validation fold stays unseen. This one structural habit removes the second form of leakage from the previous section.

### 20 · Encode each category as its own column

**Say:** A model needs numbers, and categories need care. If you code emergency, elective and urgent as 1, 2 and 3, you tell the model that urgent is three times emergency and that the gaps are equal. One-hot encoding gives each category its own zero-or-one column and makes no such claim. It has a cost. Six categorical columns in the course CSV become 25, and the Doctor column alone has 40,000 distinct values, which would add 40,000 columns. High-cardinality columns need a different treatment or exclusion.

**Notebook:** Tutorial 1 Part 1A, section 7.

### 21 · The baseline sets the bar

**Say:** Before trusting a model, ask what a model with no intelligence would score. The classes are unbalanced: 63 percent benign, 37 percent malignant. A dummy that always answers "benign" reaches 0.63 accuracy and misses every malignant case. Macro F1 exposes it at 0.39, because it averages performance over both classes. Logistic regression reaches 0.98 on both. Choose your metric before you see results, so that the choice reflects the clinical question and stays independent of which number happens to look best.

**Notebook:** Tutorial 1 Part 1A, sections 3 and 4.

### 22 · Read errors by direction

**Say:** This is the held-out test set, 114 cases, evaluated once. The model got 112 right. Read the two off-diagonal cells first. One malignant tumour was called benign: a false negative, a missed cancer. One benign tumour was called malignant: a false positive, an unnecessary follow-up. Accuracy counts those two errors as equal, and a patient does not. Also keep the takeaway in view: 0.98 on 114 cases is one estimate from one split, and a different split would give a slightly different number.

**Notebook:** Tutorial 1 Part 1A, section 5.

### 23 · Which mistake costs more?

**Say:** Two metrics separate the two directions of error. Precision asks: of the cases the model flagged, how many were real? Lead with it when a false alarm is costly, for example an invasive biopsy or alert fatigue on a ward. Recall, which clinicians call sensitivity, asks: of the real cases, how many did the model find? Lead with it when a miss is costly. The model outputs a probability, and the threshold that turns it into a decision is yours to set according to those costs.

**Ask:** For a screening test that feeds into a confirmatory test, which one do you lead with?

### 24 · AUC measures ranking

**Say:** The ROC curve shows every possible threshold at once, and the area under it is 0.995 here. AUC has a plain meaning: draw one malignant case and one benign case at random, and AUC is how often the malignant one receives the higher score. It measures ranking. Before a clinic could use this model it would still need a chosen threshold, a calibration check that a predicted 80 percent means 80 percent, and evidence that acting on the prediction helps patients.

**Notebook:** Tutorial 1 Part 1A, section 6.

## Part 1B · Regression

### 25 · Part 1B · Regression — Predicting a number

**Say:** Now the target is a number. The question: can baseline measurements estimate a disease-progression score one year later, and by how much do we miss? The dataset is the scikit-learn diabetes set, 442 patients and ten baseline measurements. The workflow is the same seven steps, so I will spend the time on what differs: the metrics, and a demonstration of leakage.

### 26 · Three metrics, three questions

**Say:** Mean absolute error answers "how large is a typical miss?" in the target's own units, which makes it the easiest to explain to a clinician. Root mean squared error asks the same question and punishes large misses, because a ten-unit error counts four times a five-unit error. R-squared asks how much better the model is than always predicting the mean. It has no units, and zero means no better than the mean. Report MAE for meaning and R-squared for context.

### 27 · Better than baseline — and still mostly unexplained

**Say:** The dummy that predicts the mean misses by about 67 points on average. Ridge regression misses by 45. That is a clear improvement. Now read R-squared: 0.48 in cross-validation and 0.45 on the held-out test set, so more than half of the variation between patients is still unexplained. Hold both facts together. A model can beat its baseline and still be too weak to base a decision on.

**Notebook:** Tutorial 1 Part 1B, sections 4 and 5.

### 28 · Reading R² correctly

**Say:** A number line helps. One is a perfect model. Zero is what you get by predicting the mean for everyone. Our model sits at 0.45. R-squared can also go below zero, and that surprises people. A negative value means the model did worse than predicting the mean, which usually signals overfitting or a bug. We will see a negative R-squared in three slides, and it will be the correct answer.

### 29 · What the metrics hide: compressed predictions

**Say:** The actual progression scores have a standard deviation of 73. The model's predictions have a standard deviation of 54. The model rarely predicts a very high or a very low score. Think about where a clinician would act: at the extremes. So the model is least informative exactly where decisions are made, and MAE, RMSE and R-squared all stay silent about it. The curves here are illustrative, drawn from the two measured standard deviations.

### 30 · Actual against predicted, and the residuals

**Say:** This is the notebook's own plot. On the left, a perfect model would put every point on the diagonal. Our points form a flatter cloud: patients with high actual scores sit below the line, so they are under-predicted, and patients with low scores sit above it. On the right, the residuals slope for the same reason. Always draw this plot. It is the only place this behaviour shows up.

**Notebook:** Tutorial 1 Part 1B, section 6.

### 31 · Leakage can manufacture a result

**Say:** Here is an experiment where we know the right answer in advance. Take 200 rows, 2,000 features and a target, all pure random noise. The true R-squared is zero. Now pick the 20 features that correlate best with the target, using all rows, and then cross-validate a model on them. The result is plus 0.32. We have manufactured a finding from noise. Move the selection step inside the pipeline, so that it sees only each training fold, and the result is minus 0.28: the honest answer, that there is nothing here.

**Notebook:** Tutorial 1 Part 1B, section 7.

### 32 · The defence is structural

**Say:** The two rows differ only in order. In the leaking version, feature selection saw every row, including the rows later used for validation, so the chosen features already carried information about the answers. In the safe version the data are split first, and selection and fitting live together in one Pipeline object. Careful intentions are a weak defence against leakage. Code structure that makes the leak impossible is a strong one.

### 33 · Before you report any number

**Say:** Six questions to ask before any result leaves your hands. Does every feature exist at prediction time? Did the split come before any fitting? Is every fitted step inside the pipeline? Was the metric chosen before the results? Was the model compared with a dummy baseline? Was the test set used exactly once? And one rule for the day a result looks wonderful: check for leakage first.

### 34 · What still needs separate evidence

**Say:** Suppose all six answers are yes. You have shown that the workflow was executed correctly, and that is all a high score shows. Five things still need their own evidence. Performance at another site or in a later year. Calibration. Performance in subgroups, which tells you who carries the errors. Causation, which needs a causal study design. And usefulness: whether acting on the prediction helps anyone. Keep this slide in mind during Tutorial 2, where the outputs are fluent sentences and the temptation to trust them is stronger.

---

# Tutorial 2 · From tokens to grounded answers

### 35 · Tutorial 2 — From tokens to grounded answers

**Say:** Tutorial 2. We now open up the language model. The subtitle lists the four stops: how text becomes numbers, how a model combines them, how it picks words, and how evidence gets in.

### 36 · Five mechanisms, five questions

**Say:** Five mechanisms, each answering one question. Tokens: how does text become numbers? Vectors: how does a token get a meaning? Attention: how do tokens use each other? Decoding: how is the next word chosen? Retrieval: how does evidence get in? Four of them have a notebook. Decoding is the grey box, which we cover from the slides. Each stop creates the problem that the next one solves.

## Part 0 · Tokenization

### 37 · Part 0 · Tokenization — A model reads tokens

**Say:** The question: how does a sentence of clinical text become the numbers a language model actually receives? The short answer is that the model never sees words. It sees tokens, and clinical language fares badly in that translation.

### 38 · Clinical words arrive in pieces

**Say:** On the left is what we write, in the middle what the model receives, using the tokenizer behind the GPT-4 family. "Portal" is one token. "Metformin" is three. "Myocardial infarction" is five. "HbA1c" is five single characters. The model handles everyday words whole and clinical words as fragments, and it has to reassemble the meaning from the pieces.

**Ask:** Why would "portal" get its own token when "HbA1c" does not?

### 39 · Byte pair encoding: merge what is frequent

**Say:** The answer is in how the tokenizer is built. Byte pair encoding starts with single characters. It finds the most frequent adjacent pair in a large body of text and merges it into a new symbol. Then it repeats, tens of thousands of times. "Portal" is common in web text, so its pieces merge all the way up into one token. The ordered list of merges is the tokenizer. In the notebook you will build one by hand in about twenty lines.

**Notebook:** Tutorial 2 Part 0, sections 3 and 4.

### 40 · Frequency decides the cost

**Say:** This is the hand-built tokenizer from the notebook, trained on fourteen fictional clinic sentences. The grey number is how often each word appeared in training. "The" appeared twenty times and costs one token. "Tachycardia" and "myocardial" never appeared and cost nine each. Look at "metformin": it appeared twice and still costs eight. Seeing a word once or twice does little. Token cost follows frequency, and clinical terms are rare in the general text these tokenizers learn from.

**Notebook:** Tutorial 2 Part 0, section 5.

### 41 · Clinical text costs more

**Say:** Here is the production tokenizer on five kinds of text. The dashed line marks one token per word. Plain English and clinic administrative text sit just above it at 1.08. A clinical narrative costs 2.0 tokens per word, a medication list 2.3, and a lab panel 4.0. Two things are counted in tokens: the price you pay, and the context limit that decides how much text fits in a prompt. Clinical text uses up both two to four times faster.

**Notebook:** Tutorial 2 Part 0, section 7.

### 42 · Same meaning, different tokens

**Say:** One more consequence, and it sets up the rest of the day. "Laboratory result" and "bloodwork" mean the same thing and share zero tokens. "Appointment" and "visit" share zero. Tokens come from character frequency and carry no meaning of their own. So meaning has to come from somewhere else. One source is vectors, which is the next notebook. The other is an explicit terminology, which we add in Part 3.

**Notebook:** Tutorial 2 Part 0, section 8.

## Part 1 · Word2Vec and t-SNE

### 43 · Part 1 · Word2Vec and t-SNE — From identity to meaning

**Say:** A token ID is only a row number in a table. Token 4,821 is no closer to token 4,822 than to any other. The question for this notebook: how does a token acquire a meaning?

### 44 · Meaning from neighbours

**Say:** The idea is sixty years old: you know a word by the company it keeps. Take a centre word, here "followup", and look at the words within a window of two on each side. Train a small model to connect centre words with their context words. Words that appear in similar contexts are pushed toward similar vectors. Note the qualifier in the takeaway. The vectors describe how words are used in this corpus, and a different corpus would give different neighbours.

### 45 · Two training directions

**Say:** Word2Vec can be trained in two directions. CBOW uses the context to predict the centre word. Skip-gram uses the centre word to predict its context, and it is the one we use because it behaves better on small corpora. Notice the pattern in both: hide part of the text and predict it from the rest. We will meet that pattern again in the attention notebook, at much larger scale.

### 46 · Comparing vectors: cosine similarity

**Say:** Each word is now a vector of 50 numbers, and we compare two words by the angle between their vectors. Cosine similarity is 1 when they point the same way and 0 when they are unrelated. It ignores length, so a frequent word gains no advantage. Hold on to this formula. Embedding retrieval in Part 3 uses exactly the same calculation on whole passages.

### 47 · Variety of context beats more rows

**Say:** The notebook trains two models. The first corpus is 240 sentences stamped from one template. The median similarity between our selected words is 0.75, which means nearly every word looks like every other word, and the vectors are useless. The second corpus is 364 sentences with varied structures. The median drops to 0.28 and words separate into neighbourhoods. The lesson carries over to every representation model: quality comes from variety of context, and piling up near-identical rows adds little.

**Notebook:** Tutorial 2 Part 1, sections 1 to 5.

### 48 · Similarity in the varied corpus

**Say:** This is the notebook's similarity heat map for the varied corpus. Read one row from left to right. Bright cells are words used in similar sentences. The clearest block is at the bottom right: "privacy", "consent", "auditor", "access" and "audit" are all similar to one another and dark toward the scheduling words. Elsewhere you see bright pairs, such as "nurse" with "scheduler", "appointment" with "followup", and "laboratory" with "technician". One surprise is worth pointing out: "record" is closer to "portal" than to the other governance words, because this corpus talks about records in the portal. The vectors report usage in these sentences.

**Notebook:** Tutorial 2 Part 1, section 5.

### 49 · Reading a t-SNE map

**Say:** Fifty dimensions cannot be drawn, so t-SNE squeezes them into two. The squeeze keeps one thing faithfully: which points are close neighbours. It distorts the rest. So the reasonable reading of a t-SNE map is "which words sit in the same local neighbourhood in this run". The axis directions, the distance between far-apart groups, the size of a cluster and even the number of clusters are products of the method. People over-read these maps constantly, in papers as well as in class.

### 50 · The map from one run

**Say:** Here is the map the notebook produces. The colours are labels we assigned by hand for teaching, and the algorithm never saw them. The governance words form a tight neighbourhood at the top left, which matches the bright block in the heat map. "Record" sits apart from them, next to "portal", which also matches. The scheduling words are spread across the map, so read nothing into a colour being scattered. Now imagine changing the random seed: the groups would move, and the gaps between them would change. So trust who sits next to whom, and treat the rest as decoration. For any actual claim, go back to the cosine similarities, which were measured in the full 50 dimensions.

**Notebook:** Tutorial 2 Part 1, section 6.

**Transition:** Word2Vec gives each word one fixed vector. "Discharge" gets the same vector on a ward and in a wound-care note. The next mechanism lets a word's representation depend on the sentence around it.

## Part 2 · Attention, masks and training objectives

### 51 · Part 2 · Attention, masks and training objectives — How tokens use each other

**Say:** This is the longest section, and it has one question: how does one attention operation support two different ways of training a language model? We will look at the two training styles first, then the operation they share, and then the single setting that separates them, which is the mask.

### 52 · The problem: learning language from raw text

**Say:** Labelled health text is scarce and expensive, because every label needs a clinician's time. Unlabelled text is almost unlimited. So we let the text label itself. Hide a piece of it, ask the model to predict the hidden piece, compare the prediction with the real text, and adjust. This is called self-supervised learning. It is the same idea as Word2Vec, applied to a far larger model. The only design question is which piece to hide.

### 53 · Autoregression: predict the next token

**Say:** The first answer: hide the next token. The model reads "An urgent result is communicated by the" and predicts what follows. It sees only what came before. The formula says the probability of a whole sentence is a chain of next-token predictions multiplied together. The loop underneath is how these models write: predict one token, append it to the text, repeat. Every chat model you have used works this way, one token at a time from left to right.

### 54 · Masked language modelling: fill in the blank

**Say:** The second answer: hide random tokens anywhere in the text, about 15 percent of them in BERT, and replace each with a special mask token. The model predicts each hidden word using the words on both sides. "The nurse blank the laboratory result": the word "result" on the right helps you guess "reviewed". A model trained this way becomes good at reading. It turns text into representations that are useful for classifying notes and for search.

### 55 · Same goal, different strengths

**Say:** Side by side. The autoregressive model hides the next token, may look only at earlier tokens, and is a natural writer: answers, summaries, drafts. That is the GPT family. The masked model hides random tokens, may look in both directions, and is a natural reader: classification, clinical coding, and the embedding models we will use for retrieval. That is the BERT family. Both learn what words mean in context from unlabelled text. In the notebook you will train one small model each way.

### 56 · Both depend on one operation

**Say:** Here is the link between them. To predict a hidden token, a position has to gather information from the positions it is allowed to see. Each grid shows that permission: the row is the position doing the predicting, and green cells are positions it may use. The autoregressive grid is a triangle, because each position sees only the past. The masked grid is full. The gathering step is called attention, and these two grids are the only difference between the training styles. Let us first see how the gathering works, and then come back to the grids.

### 57 · Attention ends in a weighted average

**Say:** Start from the end. Attention produces a weighted average. Take weights of 0.75 and 0.25 and two value vectors. Multiply, add, and you get 7.5 and 5.0. The output is a blend of both values in proportion to the weights. Attention never picks one source and discards the others. Everything else in the mechanism exists to decide what those weights should be.

**Notebook:** Tutorial 2 Part 2, section 1.

### 58 · Three roles: query, key, value

**Say:** Each position plays three roles, and each role gets its own vector. The query says what this position is looking for. The key says how this position can be found by others. The value is the information it hands over when it is chosen. Think of a library. Your query is your question, the keys are the labels on the spines, and the values are the contents of the books. Queries are compared with keys to get weights, and the weights are applied to the values.

### 59 · Scaled dot-product attention

**Say:** Here is the whole formula, and underneath it the five steps. Match: multiply queries by keys, which compares every query with every key. Scale: divide by the square root of the key dimension. Mask: set the score of every blocked pair to minus infinity. Normalise: apply softmax across each row, so each row of weights sums to one and the blocked pairs become exactly zero. Combine: multiply the weights by the values, giving one output row per query. The red box, the mask, is the step this section is really about. The other four are identical in every transformer.

**Notebook:** Tutorial 2 Part 2, section 2.

### 60 · Why divide by √d_k?

**Say:** A short aside on the scale step, which the notebook does not cover. Dot products grow as vectors get longer. Feed softmax moderate scores such as 2 and 0, and the weights come out 0.88 and 0.12, so both keys contribute. Feed it extreme scores such as 8 and 0, and the weights collapse to 0.9997 and 0.0003. Attention has turned into picking a single key, and learning stalls. Dividing by the square root of the dimension keeps the scores moderate, so attention stays a blend.

### 61 · One row, slowly

**Say:** Follow one query through. The query "clinic" scores 0.707 against its own key and 0 against "portal". Softmax turns those into weights of 0.670 and 0.330. Multiply by the values: 0.670 times 10, 0 plus 0.330 times 0, 20 gives 6.70 and 6.60. Look at the second number. Its weight was the smaller one, yet it contributes almost as much, because its value was 20. A position's contribution depends on its weight and on the size of its value together. That is one reason an attention weight alone is a weak explanation of what a model did.

### 62 · Scores, weights, output

**Say:** This is the same calculation as the notebook shows it: scores on the left, weights in the middle, outputs on the right. Check that each row of weights sums to one. The function that produces all three tables is five lines long, and the notebook confirms it gives the same answer as PyTorch's built-in attention. That is all the arithmetic there is. From here on we change what goes into the function.

**Notebook:** Tutorial 2 Part 2, section 3. In the student version, completing this function is the first exercise.

### 63 · Self-attention and cross-attention

**Say:** Where do queries, keys and values come from? In self-attention, all three are projections of the same sequence. Every token is a query, and every token is also a key and a value, so the weight matrix is square. In cross-attention, the queries come from one sequence and the keys and values from another, which is how a decoder reads an encoder's output in a translation model. The models we discuss today use self-attention.

### 64 · Beyond the notebook: attention is blind to word order

**Say:** A second aside that the notebook leaves out. Look back at the formula: nothing in it refers to a token's position. Shuffle the input tokens and the outputs are simply shuffled the same way. For health text that is serious. "Result normal, not abnormal" and "result abnormal, not normal" contain the same tokens and mean opposite things. So models add a position signal to each token before computing queries, keys and values. With token content alone the output change after a shuffle is 0.00, and with a positional encoding added it is 0.89. Once positions exist, we can talk about which positions a token may read.

### 65 · A causal mask hides the future

**Say:** Here is the triangle again, now with its mechanism. Each position may use itself and earlier positions. Every later position has its score set to minus infinity before the softmax, so its weight becomes exactly zero. In the two-token example, the first row changes from 0.670 and 0.330 to 1 and 0, because the first token can no longer read the second. This is the autoregressive rule, enforced inside attention. The timing matters: masking happens before softmax, so it changes what information is available. It is more than hiding a cell afterwards.

### 66 · Same Q, K and V; only the mask differs

**Say:** The notebook runs attention twice on a four-token sentence with identical queries, keys and values. On the left there is no mask and every token reads every token. On the right is the causal mask: the upper triangle is exactly zero. Every row still sums to one, so the weight that would have gone to later tokens has moved onto the positions that remain visible. "Clinic", the first token, can read only itself, and its row is 1, 0, 0, 0.

**Notebook:** Tutorial 2 Part 2, section 4.

### 67 · A mask is a table of who may read whom

**Say:** A mask is a table of true and false with one row per reader. Model families differ largely in how they fill it in. Bidirectional: read every token, used by BERT-style encoders. Causal: read yourself and earlier tokens, used by GPT-style models. Padding: nobody reads the filler tokens that even out a batch. Prefix: the prompt is read both ways and the continuation is causal, as in T5. Sliding window: read only the last few tokens, which long-context models use to control cost. The arithmetic of attention stays the same in all of them.

### 68 · Six masks on one sentence

**Say:** Here are six masks drawn on one sentence, straight from the notebook. Green means the row token may read the column token. Follow the row for "cough". Under the full mask it reads the whole sentence, including "today", which comes after it. Under the causal mask it reads "the patient reports cough". The padding mask removes a whole column. The sliding window allows the fewest pairs, 14 of 36, and that is its purpose: full attention on n tokens costs n squared comparisons, and a window of width w costs about n times w.

**Notebook:** Tutorial 2 Part 2, section 5. In the student version you write the causal and sliding-window masks yourself.

### 69 · Each training objective is a mask plus a target

**Say:** Now we can close the loop from the start of this section. A training objective is two choices: a mask and a target. For autoregression, the target is the input shifted by one, every position is scored, and the mask must be causal. Without it, position 3 could read position 4, which is its own answer. For masked language modelling, the target is the original token at each mask position, about 15 percent of positions are scored, and the attention mask is full because the clues sit on both sides. Note the vocabulary trap in the takeaway. The attention mask is a table of permissions. The mask token is a change to the input. They are two separate things that share a name.

### 70 · Both objectives on one sentence

**Say:** The notebook draws both objectives on one sentence. Each row label reads "input, arrow, target". On the left, autoregression scores 11 positions out of 11, so every token is a training example. On the right, masked language modelling scores 2 positions out of 12, the outlined rows. Autoregression gets far more training signal per sentence, which is one reason the largest generative models use it. Now look at what the masked model gets in return. The row for the first mask token is green all the way across, so it may read "chest xray" to its right. The autoregressive row for that position stops at "reports".

**Notebook:** Tutorial 2 Part 2, section 6.

### 71 · From an attention row to a prediction

**Say:** How does an attention output become a predicted word? Three steps. Attend: the position gets one row of weights over the positions its mask allows. Combine: those weights blend the value vectors into one output vector. Predict: a final linear layer, called the prediction head, turns that vector into one score for every word in the vocabulary, and softmax turns the scores into probabilities. In autoregression, the output at position i predicts token i plus one. In masked modelling, the output at a mask position predicts the hidden token.

### 72 · One architecture, trained two ways

**Say:** This is the central figure of the notebook. We trained one tiny architecture twice on a fictional corpus in which each symptom determines the test that gets ordered. Each row shows on the left what a position read, and on the right what it predicted. Top row: the autoregressive model after "reports" gives 0.25 to each of the four symptoms. Nothing on the left tells it which one is coming, and the grey bars show that everything on the right is hidden. That even split is the honest answer. Middle row: after "a", the same model predicts "chest" with confidence, and its attention went almost entirely to "cough", eight positions back. Attention is the mechanism that carried that fact forward. Bottom row: the masked model recovers the hidden symptom by reading "chest" and "xray" on the right.

**Notebook:** Tutorial 2 Part 2, sections 7 and 8. Training takes a few seconds.

**Ask:** Before I run it: what should the top-right panel look like, and why?

### 73 · Testing the mask: can the future leak?

**Say:** A causal mask makes a guarantee, and guarantees can be tested. Change two late tokens, "chest xray" to "blood test", and measure how far the model's scores move at the earlier positions. For the autoregressive model the largest earlier change is exactly 0.0. The future cannot influence the past, by construction, and that is what allows the model to write token 10 before token 11 exists. For the masked model the change is 12.5, and that is the intended behaviour. Here is the link back to Tutorial 1. If an autoregressive model ever shows a non-zero number in this test, it has been reading its own answers during training, and its low loss is worthless. That is data leakage in sequence form.

**Notebook:** Tutorial 2 Part 2, section 9.

### 74 · The same contrast in pretrained models

**Say:** Does this hold in real models? The notebook loads two small pretrained models that run on a laptop. On the left, distilgpt2, an autoregressive model, reads "The patient went to the" and spreads its probability over destinations, with "hospital" first at 0.24. In the middle and on the right, distilbert, a masked model, gets the same five words on the left plus the rest of the sentence. With "to pick up the prescription" on the right it answers "pharmacy" at 0.80. With "to have the surgery" it answers "hospital" at 0.53. The left context is identical in all three panels, so the change comes entirely from the words on the right.

**Notebook:** Tutorial 2 Part 2, section 10. The first run downloads about 600 MB, and the section skips itself cleanly when offline.

### 75 · The masks are visible in real attention maps

**Say:** And here are the masks themselves, visible in the models' first-layer attention. distilgpt2 on the left is a lower triangle, and the largest weight above the diagonal is exactly zero, the same guarantee we just tested. distilbert on the right fills the whole square, with 38 percent of its weight above the diagonal. Two cautions. These are general-purpose models with no clinical training. And an attention map shows where weight went, which is different from explaining why the model gave its answer.

**Transition:** Look once more at the left panel of the previous slide: "hospital" 0.24, "doctor" 0.07, "emergency" 0.06. The model hands over a probability for every token. Something still has to choose one. That choice is decoding.

## Decoding (slides only)

### 76 · Slides only · Decoding — How the next word is chosen

**Say:** This section has no notebook this year. The question: why does the same model, asked the same question, give a different answer each time? You will set these controls yourself in the Lab 2 chatbot, so it is worth ten minutes. An archived notebook with runnable code is available for anyone who wants it.

### 77 · The model outputs a score for every token

**Say:** Back to the sentence from the autoregression slide: "An urgent result is communicated by the blank". The model's output is a probability for every token in its vocabulary. Here are the top eight: "clinician" at 0.58, "nurse" at 0.19, "physician" at 0.14, and a long tail. The model's work ends there. Decoding is a separate rule, chosen by you, that picks one token from this distribution.

### 78 · Temperature reshapes the same scores

**Say:** Temperature divides the scores before the softmax. At a temperature of 1.0, the blue bars, you get the model's own distribution. At 0.2, the dark bars, "clinician" takes 0.995 and the output becomes almost deterministic. At 2.0, the gold bars, the distribution flattens, and "fax" becomes a live option. Temperature changes how the dice are weighted. The model's underlying scores, and its knowledge, stay the same. A lower temperature buys repeatability.

### 79 · Top-k and top-p remove candidates

**Say:** Two more controls remove candidates before sampling. Top-k keeps a fixed number, here three, sets the rest to exactly zero, and renormalises. Top-p keeps the smallest set of tokens whose probabilities add up to p. The table shows why people like top-p: at p of 0.9 it keeps two tokens when the model is confident, three at the model's own temperature, and six when the model is uncertain. Top-k keeps a fixed count, and top-p adapts to the model's confidence.

### 80 · Same request, five runs

**Say:** Put it together. Send the same request five times. With greedy decoding, which always takes the top token, you get one distinct output. With sampling at temperature 1.0 you get five. When a colleague says the model is inconsistent because it summarised the same note two different ways, this is what happened. The variation comes from the sampling rule. The remedy is to judge each output against the source text, since agreement between two outputs proves nothing about either.

### 81 · Choosing settings for health work

**Say:** Some starting points. Extracting a value from a document: temperature zero, because there is one right answer and you want it on every run. Answering from retrieved policy text, which is what your chatbot will do: zero to 0.3, so the wording follows the evidence. Drafting text that a person will edit: around 0.7 with top-p of 0.9. Anything that will be audited: temperature zero with the model version pinned, so the run can be reproduced. And read the takeaway, because it sets up the last section. Decoding chooses among what the model already has. New evidence has to come in another way.

## Part 3 · Retrieval-augmented generation

### 82 · Part 3 · Retrieval-augmented generation — Getting evidence in

**Say:** The last notebook, and the one Lab 2 is built on. Two questions: what changes when the same model receives retrieved evidence, and how do we check what it did with that evidence? The second question gets most of our time. Everything uses three fictional policy documents from an invented clinic called Northstar.

### 83 · The RAG pipeline

**Say:** Six stages. Documents are split into chunks with metadata. Chunks become embedding vectors. A question retrieves the top-k chunks. Those chunks go into a grounded prompt, and the model generates a response. The two braces are the idea to hold on to: retrieval ranks text that already exists, and generation writes new text. Keeping them apart lets you find the stage where a failure happened. And notice what stays fixed. RAG changes what is in the prompt, and the model's weights are untouched.

**Notebook:** Tutorial 2 Part 3, section 1.

### 84 · Same model, two conditions

**Say:** The experiment is a controlled comparison. In the direct condition the question goes straight to the model and you get a general answer. In the RAG condition the same question goes in with the top-k chunks, to the same model. The RAG prompt adds three rules: use only the supplied context, cite the chunk you relied on, and abstain if the answer is absent. Those three rules are the core of the prompt you will write for Lab 2.

**Notebook:** Tutorial 2 Part 3, section 5.

### 85 · Chunking decides what can be retrieved

**Say:** Before anything can be retrieved, the documents are cut into pieces, and the way you cut them decides what can ever be found. Cutting by heading keeps each policy section whole: portal registration, appointment changes, accessibility. Cutting into fixed windows of words is simpler and works on any text, but look at window 2, which starts in the middle of a sentence. A fact split across two windows may be retrieved as half a fact. Neither strategy wins everywhere, so test them on your own questions. Lab 2 asks you to do exactly that.

**Notebook:** Tutorial 2 Part 3, section 3b.

### 86 · Nine chunks, one per heading

**Say:** Heading-based chunking turns the three documents into nine chunks, shown here with their word counts and coloured by source document. Each chunk has a stable ID, such as `access_guide::000`. Remember that format. Every retrieval score, every citation and every audit from here on refers to these nine IDs. Stable identifiers are what make a RAG system checkable.

**Notebook:** Tutorial 2 Part 3, section 3.

### 87 · Four questions, four behaviours to test

**Say:** The notebook evaluates with four questions, and each one tests a different behaviour. Single chunk: how long is a registration code valid? A good system cites one passage. Multi chunk: who sends urgent results, and who interprets them? A good system combines two. Unanswerable: what parking fee does the clinic charge? The documents never mention parking, so a good system says the information is absent. Partial evidence: can a parent see a teenager's results? The corpus covers proxy access and says nothing about minors, so a good system answers the covered part and names the gap. That last case is the dangerous one, because relevant but incomplete evidence invites the model to fill the gap.

**Notebook:** Tutorial 2 Part 3, section 4.

### 88 · Two ways to score a passage

**Say:** Retrieval needs a score for each question and chunk, and there are two families. Lexical scoring counts shared words: the overlap between query words and chunk words divided by their union. Every score can be explained by pointing at words, and it is free and instant. Its weakness is the one from the tokenization notebook: "bloodwork" and "laboratory result" share nothing. Embedding scoring uses the cosine similarity from the Word2Vec notebook, applied to whole passages. It matches paraphrases and synonyms, it usually needs a paid service, and it gives a ranking with no stated reason. The notebook starts with the transparent method so that every decision can be inspected.

**Notebook:** Tutorial 2 Part 3, section 6.

### 89 · Every question against every chunk

**Say:** Here is every question scored against every chunk with the lexical method. A star marks the evidence we expect. Q1 and Q2 find their starred chunks. Now read the Q3 row, the parking question: every score is zero, and asking for the top four would still return four chunks. Then Q4: the expected chunk scores 0.042 and sits below three off-topic chunks, because the question shares more words with the results policy than with the privacy policy. So we can already see two retrieval problems, and no language model has been involved yet.

**Notebook:** Tutorial 2 Part 3, section 7.

### 90 · Top-k always returns k

**Say:** Here is the same lesson with embedding retrieval, taken from the recorded run. On the left, the registration-code question, which has an answer: the top chunk scores 0.67 and the rest fall away quickly. On the right, the parking question, which has none: the scores are 0.34, 0.32, 0.30 and 0.21. It still returned four chunks, and they look like a plausible ranking. Top-k means highest ranked. Whether the top-ranked chunk is relevant is a separate check. A flat profile with a low top score is a useful warning sign, and your chatbot can use it.

**Notebook:** Tutorial 2 Part 3, section 11.

### 91 · A terminology connects patient words to document words

**Say:** Now we fix the vocabulary gap deliberately. A patient asks, "Who tells me about my bloodwork?" The documents say "laboratory result". A terminology is a file stating that these phrases name one concept, here a fictional code, NC-00412. We match the question against the terminology and add the documents' own vocabulary to the query. The table shows the effect on lexical scores: the routine-results chunk goes from 0.000 to 0.125, and the correct policy now ranks first. Real systems use SNOMED CT, LOINC or ICD-10 for this, and those bring licensing and maintenance work that our small file leaves out.

**Notebook:** Tutorial 2 Part 3, section 8.

### 92 · A knowledge graph adds relationships — and provenance

**Say:** A terminology says two phrases mean the same thing. A knowledge graph goes further and says how concepts relate. Start at "laboratory result". One hop reaches "urgent lab result", "interpretation question" and "patient portal". Two hops reach "designated clinician", "ordering team" and, in red, "code expires in 48 hours", which has drifted off topic. More hops reach more material, and relevance falls as you go. The valuable part is written under each node: every edge carries the chunk it was derived from. A reviewer who asks "why was this passage shown?" gets a traceable path, which is more useful than a similarity number.

**Notebook:** Tutorial 2 Part 3, section 9.

### 93 · Beyond the notebook: GraphRAG builds the graph with an LLM

**Say:** The graph on the last slide has seven nodes, and I drew them by hand. GraphRAG builds that kind of graph automatically, from the whole corpus. Compare the two rows. Ordinary RAG indexes passages: it embeds each chunk and later returns the chunks most similar to the question. GraphRAG adds a step in which a language model reads every chunk and writes down the entities it mentions and the relations between them. Those become a knowledge graph, and each edge keeps the chunk it came from, as ours did. The graph is then split into communities of closely linked entities, and the model writes a summary of each community. At question time there are two ways in. Local search finds the entities named in the question and follows their edges. Global search matches the question against the community summaries. All of those model calls happen before anyone asks anything, and the graph inherits every mistake the extracting model makes.

**Point at:** the pink extraction stage, then the two search boxes.

### 94 · GraphRAG is a family of designs

**Say:** "GraphRAG" is the name of a family, and the paper on the next slides compares four members. KG-based systems extract triples, such as "urgent lab result, communicated by, designated clinician", and retrieve those triples, with or without the text they came from. Community-based systems are the Microsoft design from the last slide. Graph-guided text systems, such as HippoRAG 2, use the graph only to choose chunks, so the model still reads the original passages. Summary trees, such as RAPTOR, cluster chunks and summarise them level by level without naming any entities. Read the right-hand column, because it predicts the results we are about to see. Original text keeps the details a question asks about. Triples and summaries keep the connections and lose some of the details.

**Ask:** Which of these four designs is closest to the graph we used in section 9? (Listen for graph-guided text: our graph led us to original chunks. The difference is that we built it by hand.)

### 95 · RAG or GraphRAG? It depends on the question

**Say:** Han and colleagues compared RAG and GraphRAG under one protocol, holding the chunk size, the embedding model, the number of retrieved items and the generator fixed. Blue is RAG and orange is GraphRAG, here the community design with local search. On single facts RAG is slightly ahead, and on multi-hop and comparison questions GraphRAG is slightly ahead. The two large gaps are lower down. On questions about the order of events in time, GraphRAG scores 50.6 against 30.7. On questions whose answer is absent from the corpus, where the right response is "insufficient information", RAG scores 96.0 and GraphRAG 80.1. The global-search version abstained correctly only 19 percent of the time. The right-hand column gives the reason. RAG passes the model original passages, so the details survive, including the detail that the answer is missing. GraphRAG passes entities, relations and summaries, so the links between documents survive.

One aside about evaluation. Earlier reports that GraphRAG writes better summaries relied on a language model as the judge. This paper found that swapping the order in which two summaries were shown could reverse the judge's verdict, so an LLM judge needs the same scrutiny as any other measurement.

**Point at:** the "Order in time" and "Answer absent" rows.

### 96 · They get different questions right

**Say:** Averages hide which questions each method gets right. On the left, every MultiHop-RAG question falls into one of four cells. Both methods answer 55 percent correctly and both fail on 19 percent. The coloured cells matter most: 11.6 percent are answered only by RAG and 13.6 percent only by GraphRAG. That invites combining them, and the paper tried two ways. Routing uses a model to classify each question as a fact question or a reasoning question and sends it to one method; that gained 1.1 points. Combining runs both and gives the generator both sets of evidence, and overall accuracy rose from 71.2 to 77.6. Now read the last column. With combined evidence, correct abstentions on no-answer questions fell from 91.4 to 59.5. More retrieved text gave the model more material to build an answer from when the corpus had none. This is the parking-fee question at benchmark scale, and it is why your Lab 2 test set needs unanswerable questions scored on their own line.

**Point at:** the two coloured cells, then 59.5.

### 97 · What the graph costs

**Say:** GraphRAG costs more to build. On this benchmark the RAG index took 135 seconds to build and the two graph indexes took 7,702 and 5,560 seconds, because a model has to read every chunk. Retrieval time depends on the design. The knowledge-graph version was slowest, because it expands entities with a model and walks several hops, and the community version was fastest, because it matches summaries directly. Storage is similar for all three. The row to watch is tokens per question: GraphRAG sent the generator 9,770 tokens against RAG's 3,631, and from the tokenization notebook you know what that does to cost. When the authors gave RAG the same token budget, the overall scores matched, 69.3 against 69.0, and GraphRAG kept a clear lead only on time-ordering questions. The graph can also miss facts: only 65.8 percent of the HotpotQA answer entities made it into the extracted knowledge graph, and a fact missing from the graph cannot be retrieved through it. So start with RAG. Add a graph when your own test questions need facts linked across documents, and check that the gain survives a matched token budget.

**Ask:** Which questions about Northstar's policies would need two documents to answer?

### 98 · What the evidence changed

**Say:** Now the payoff, from a real recorded run of a commercial model. The question is how long a registration code remains valid. Without context the model says codes "might remain valid for anywhere from a few hours to several days". That is fluent, reasonable and useless. With retrieved context it says 48 hours and cites `access_guide::000`. The answer became specific and traceable. Look at the token counts too: 169 against 312. Grounding roughly doubled the cost, which connects back to the tokenization notebook.

**Notebook:** Tutorial 2 Part 3, section 11. The notebook replays this recorded run by default, so it needs no key and costs nothing.

### 99 · Evaluate in two gates

**Say:** Evaluate a RAG system in two stages. The retrieval gate asks whether the retrieved set contained every chunk the answer needs. When it fails, fix the chunking, the value of k, or the query. The generation gate asks whether the answer cites only retrieved chunks and abstains when evidence is missing. When it fails, fix the prompt or check the model. The takeaway gives the test case: the correct chunk was retrieved and the answer contains an invented fee. Retrieval passed, so that is a generation failure. Lab 2 asks you to report these two gates separately.

**Notebook:** Tutorial 2 Part 3, section 12.

### 100 · A citation is a pointer for a human check

**Say:** Here is the exercise I most want you to remember. Five hand-written answers, all with citations. The first is correct. The second adds an invented fifteen-dollar fee and cites a real chunk. The third cites the wrong chunk. The fourth invents a policy about patients aged fourteen and older. The fifth cites a chunk ID that does not exist. Now read the right-hand column. The automated checker flags one of the four faulty answers, the fabricated ID. The checker verifies that the citation label exists. Verifying the claim takes a person who reads the cited passage.

**Notebook:** Tutorial 2 Part 3, section 13. Try the audit yourself before you open the answer key.

### 101 · The partial-evidence trap

**Say:** One of those cases deserves its own slide. The question: can a parent see a teenager's results? Retrieval does its job and returns a relevant passage saying proxy access requires documented consent. The corpus says nothing about minors, ages or sensitive results. The generated answer then continues, "For patients aged 14 and older, withhold…", which is confident, plausible and invented. Retrieval succeeded, and generation overreached into the gap. The safe answer covers the supported part and then names the gap. Put that instruction in your prompt, and put a question like this one in your test set.

### 102 · From notebook to chat interface

**Say:** So far everything ran in notebook cells, and the people who will use your system will see a chat box. Streamlit turns a Python script into a web page. Three ideas are enough to read the code. The script reruns from top to bottom on every interaction. So anything that must survive, such as the conversation, lives in session state, and anything expensive, such as the corpus and the embeddings, is cached. On the right is the flow: the notebook writes the app file, Streamlit serves it on localhost, and the app calls the same retrieve function you studied earlier and calls generate only when a key is present. The interface adds no intelligence. It gives someone else a way to use what you built.

**Notebook:** Tutorial 2 Part 3, section 14. Run the launch cell, open the printed address, and run the stop cell before you close the notebook.

### 103 · The chat app, asked a patient's question

**Say:** Here is the app answering the bloodwork question in retrieval-only mode, with no key and no cost. It reports that no model was called, names the best-matching passage, and shows that the terminology added "laboratory result". The evidence panel lists the same ranking we saw in the terminology section, because it is the same code. Now look at rank 4: a score of 0.000, and it was shown anyway. That is "top-k always returns k" inside a friendly interface. A chat box looks finished and authoritative, so the limits have to be put on the screen. The notebook ends with that exercise: make the app decline when the top score is too low.

**Ask:** What would you want this screen to tell a patient that it currently leaves out?

### 104 · Limits to carry forward

**Say:** Six limits to take with you. Nearest neighbours still need a relevance check. Retrieved sources can be incomplete or wrong, and the model will repeat them faithfully. A cited answer still needs its claim checked. Our terminology and graph are tiny, fictional and hand-built, and they inherit their author's mistakes. The replayed responses are one run on one date. And live services raise cost, privacy and retention questions that must be settled before any real document goes near them. In one line: treat retrieved text as data to check.

---

# Close

### 105 · Your turn: Lab 2

**Say:** Lab 2 asks you to do all of this on a new corpus. Five steps. Know the corpus: what it can answer fully, in part, and not at all. Design retrieval, and compare at least two configurations on the same questions. Build the chatbot so that it cites, abstains and shows its evidence. Evaluate it with the two gates and a citation audit done by hand. Then plan the path to production by brainstorming with an AI tool and applying your own judgement to what it suggests. You also submit an AI use record and a one-page reflection, as in Lab 1. My strongest advice is on the slide: write your test questions before you tune the system, and include questions the corpus cannot answer. That is "split first, evaluate once", carried over from Tutorial 1.

### 106 · Seven things to keep

**Say:** Seven things to keep. From Tutorial 1: compare with a baseline before anything else, put every step that learns inside the pipeline, and treat a surprisingly good result as a reason to look for leakage. From Tutorial 2: clinical text is expensive in tokens, the mask decides what a model may read, temperature zero buys repeatability, and a person still has to check each cited claim. The first three tell you how to judge a number. The last four tell you how a language model produced its sentence, and why that sentence deserves the same scrutiny. Thank you. Open Tutorial 2 Part 3 and start with your corpus.
