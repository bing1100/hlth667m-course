# Tutorial Setup: VS Code, Cline, and Google Colab

This guide prepares you to work through the tutorial in **Visual Studio Code (VS Code)**. You will edit notebooks and files in VS Code, connect the notebook to a **Google Colab** computer to run code, and use **Cline** as an optional AI assistant.

> **You do not need to use the Google Colab web interface.** We will connect to a Colab runtime from inside VS Code.

## In This Tutorial

### The instructor will demonstrate

- Setting up VS Code and the supporting Python tools.
- Setting up an LLM workspace with Cline and an Azure-hosted model.
- Connecting a VS Code notebook to Google Colab.
- Running a synthetic-data pipeline.

### We will do collaboratively

- Practice prompting an AI assistant using **Plan** and **Act** modes.
- Conduct and synthesize a literature search.
- Analyze a dataset.
- Generate a synthetic-data pipeline.

## Before You Begin

Please have the following ready:

- A laptop with a reliable internet connection.
- A current version of [VS Code](https://code.visualstudio.com/).
- A Google account that can access Google Colab.
- The downloaded course/tutorial folder.
- The Azure access details provided by the instructor or your organization, if you will configure Cline during the session.

Allow approximately **15–20 minutes** for installation and sign-in.

### Working Safely with Health Data

Use only the synthetic or de-identified course materials supplied for this tutorial.

- Do **not** upload, paste, or prompt with identifiable patient, clinical, employee, or research-participant information.
- Do **not** place passwords, API keys, access tokens, or other secrets in notebooks, chat prompts, or files that you share.
- Treat AI-generated code, summaries, and literature findings as drafts: check the source, test the code, and make sure you understand the result.

## 1. Install and Open VS Code

[Download and install VS Code](https://code.visualstudio.com/download) for your operating system. Use the standard installer options unless your organization provides different instructions.

**What is VS Code?** VS Code is the application where you will open course files, read and edit code, run notebooks, and connect to Colab.

After installing it:

1. Open VS Code.
2. Select **File → Open Folder**.
3. Choose the downloaded course/tutorial folder.
4. If VS Code asks whether you trust the authors of the folder, select **Trust** only after confirming that it is the course material you received.

## 2. Install the VS Code Extensions

Open the **Extensions** view in VS Code:

- Windows/Linux: `Ctrl+Shift+X`
- macOS: `Cmd+Shift+X`

Search for and install the following extensions.

| Extension | Required? | What it does |
| --- | --- | --- |
| [Colab](https://marketplace.visualstudio.com/items?itemName=Google.colab) by Google | Yes | Connects a notebook in VS Code to a Google Colab runtime. |
| [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) by Microsoft | Yes | Opens, displays, and runs Jupyter notebook (`.ipynb`) files in VS Code. |
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) by Microsoft | Recommended | Provides Python editing support and is useful if you later run code locally. |
| [Cline](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) | For the AI-assisted activities | Adds an AI assistant to VS Code for planning, explanation, and guided file/code work. |

The Python extension may install supporting tools such as **Pylance** automatically. You do not need to install them separately for this tutorial.

### Optional: Data Wrangler

[Data Wrangler](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.datawrangler) is an optional Microsoft extension that gives you a visual way to inspect and clean tabular data, such as CSV files. It can also generate the corresponding Pandas code. Install it if you would find a point-and-click view of the dataset helpful.

## 3. Connect a Notebook to Google Colab from VS Code

For this tutorial, **VS Code is your workspace** and **Colab is the online computer that runs your notebook code**. This means you do not need a local Python installation to follow the main tutorial workflow.

1. In the VS Code Explorer, open:

   ```text
   Part-1-Setup_&_Colab/SDV_Synthesize_a_table_(CTGAN).ipynb
   ```

2. In the upper-right corner of the notebook, select **Select Kernel**.
3. Select **Colab → Auto Connect**.
4. When prompted, sign in with your Google account in the browser window.
5. Return to VS Code and wait for the Colab connection to complete.
6. Run the first code cell using its **Run** (play) button.

### A Few Useful Terms

| Term | Meaning |
| --- | --- |
| **Notebook** | A document that combines notes, code, and results. Notebook files end in `.ipynb`. |
| **Cell** | One small section of code or text in a notebook. Run code cells one at a time or from top to bottom. |
| **Kernel / runtime** | The temporary computer session that runs the notebook code. In this tutorial, it is provided by Colab. |

### First-Run Note

The first code cell installs the **SDV** package used for synthetic-data generation. This may take a few minutes. If the Colab runtime disconnects or resets, reconnect and rerun the setup cells from the top.

The notebook includes a later CTGAN example that uses `epochs=1000`. This training step can take longer than the earlier examples, so treat it as **optional unless the instructor asks you to run it**.

## 4. Set Up Your Cline LLM Workspace

**Cline** is an AI assistant inside VS Code. During this tutorial, it can help you think through tasks, explain code, organize a literature search, and draft or revise work under your direction.

> Cline is an assistant, not an authority. You remain responsible for checking its claims, reviewing changes, and understanding submitted work.

### Open Cline

1. Select the **Cline** icon in VS Code's Activity Bar.
2. Follow the sign-in or provider-configuration prompts.
3. Choose the Azure-hosted model/provider configuration supplied by the instructor or your organization.

Azure configurations vary by organization. Use only the approved endpoint, deployment/model name, and authentication method provided for this course. Do not guess values or share credentials in the chat.

### Plan Before Act

We will use two complementary ways of working with Cline:

| Mode | Use it for | Example prompt |
| --- | --- | --- |
| **Plan** | Understanding the task, reviewing files, identifying assumptions, and proposing steps before changing anything. | `Read this notebook and propose a high-level plan to explore the dataset. Do not change files.` |
| **Act** | Making a specific, reviewed change after the plan is clear. | `Add a short Markdown explanation of this chart. Preserve the existing code.` |

### Good Prompting Habits

- State your goal, audience, and constraints.
- Ask for a short plan before requesting broad changes.
- Provide the relevant file or selected text rather than assuming the assistant can infer your context.
- Ask the assistant to explain unfamiliar code in plain language.
- Review every proposed change before accepting it.
- Verify results by running the notebook cells and checking the output.

For literature work, use AI to help refine search concepts, organize notes, or draft a synthesis. Always read and cite the original sources; do not treat an AI-generated citation as verified.

## 5. Confirm That Your Setup Works

You are ready when you can complete all of the following:

- [ ] Open the course folder in VS Code.
- [ ] Open the CTGAN notebook.
- [ ] Select **Colab → Auto Connect** as the notebook kernel.
- [ ] Run the first notebook cell without an error.
- [ ] See a dataset preview after running the early notebook cells.
- [ ] Open Cline and, if applicable, see the approved Azure configuration.
- [ ] Explain the difference between Plan mode and Act mode.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| **Colab is not shown in Select Kernel.** | Confirm that the Colab and Jupyter extensions are installed. Reload VS Code, reopen the notebook, and try again. |
| **Google sign-in does not complete.** | Allow the browser sign-in window or pop-up, then try again with the Google account approved for the course. If your organization blocks Colab, notify the instructor. |
| **`No module named 'sdv'` appears.** | Rerun the first `%pip install sdv` cell, then rerun the cell that failed. |
| **The Colab connection disconnects.** | Select **Kernel → Colab → Auto Connect** again, then rerun cells from the beginning. Colab runtimes are temporary. |
| **Model training is taking too long.** | Stop or skip the optional 1,000-epoch CTGAN example and continue with the earlier cells. |
| **Cline cannot connect to Azure.** | Check that you selected the instructor-provided configuration and that you are signed in through the approved method. Do not paste credentials into the chat; ask the instructor for help. |
| **An AI answer looks plausible but you are unsure.** | Pause, inspect the source/code/output, and ask the instructor. Do not submit work you cannot explain. |

## Official Resources

- [VS Code setup](https://code.visualstudio.com/docs/setup/setup-overview)
- [Jupyter notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Google Colab VS Code extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)
- [Cline installation documentation](https://docs.cline.bot/getting-started/installing-cline)
- [SDV documentation](https://docs.sdv.dev/sdv)

---

## Appendix: Optional Local Python Setup with Anaconda

You **do not need Anaconda or a local Python installation** for the main tutorial workflow because the notebook runs on Colab. This appendix is for students who would like to run Python and Jupyter notebooks on their own computer later.

### What Is Anaconda?

[Anaconda Distribution](https://www.anaconda.com/download) is a Python distribution that includes Python and tools for managing separate project environments. A local environment can be useful for independent work, but it requires more setup and maintenance than the Colab route.

### Optional Installation Steps

1. Download Anaconda Distribution for your operating system from the [official Anaconda download page](https://www.anaconda.com/download).
2. Follow Anaconda's [official installation guide](https://docs.anaconda.com/anaconda/install/).
3. Open VS Code and install the **Python** and **Jupyter** extensions if you have not already done so.
4. Open a notebook in VS Code and select **Select Kernel**.
5. Choose the local Python/Anaconda environment you created.

When working locally, packages must be installed into the selected local environment. For this tutorial, the needed package is installed with:

```python
%pip install sdv
```

Use a separate environment for course work when possible. This helps prevent package conflicts with other projects. If you are unsure which environment or kernel to select, use the Colab connection for the tutorial and ask the instructor for support before changing local settings.
