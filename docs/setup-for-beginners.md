# Setting up your computer for the course (complete beginner)

This page takes you from a computer with nothing installed to a working notebook that can run kusim. You do not need any programming experience. Allow **30 to 45 minutes**, mostly waiting for downloads, and an internet connection.

You will install and use four things:

| Tool | What it is for |
|---|---|
| **A terminal** | A window where you type commands. Already on your computer. |
| **git** | Downloads the course and kusim from the internet (GitHub). |
| **uv** | Installs Python and every package kusim needs, in one private folder. |
| **JupyterLab** | The notebook program you will work in every day. |

You do **not** install Python yourself. `uv` does it for you.

Anything written in a grey box below is a **command**: type it (or copy and paste it) into the terminal and press **Enter**. Lines starting with `#` are comments for you to read; leave them out.

---

## Step 1. Open a terminal

- **Windows:** press the **Windows key**, type `PowerShell`, and open **Windows PowerShell** (the blue window). Use this window for everything below.
- **macOS:** press **Cmd + Space**, type `Terminal`, press Enter.
- **Linux:** open the **Terminal** application.

Keep this window open, or reopen it when a step says so.

## Step 2. Install git

**Check first:** type

```
git --version
```

If it prints something like `git version 2.45.0`, git is already installed: go to Step 3. If it says the command is not recognised or not found, install it:

- **Windows:** download **Git for Windows** from <https://git-scm.com/download/win>, run the installer, and click **Next** through every page, keeping the default choices. Then **close the PowerShell window and open a new one**.
- **macOS:** type `git --version` and, if a window offers to install the "command line developer tools", click **Install**. Wait until it finishes.
- **Linux (Ubuntu or Debian):** `sudo apt install git`

Check again with `git --version`.

## Step 3. Install uv

- **Windows (PowerShell):**

  ```
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- **macOS and Linux:**

  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

**Close the terminal and open a new one** (this is needed so that the computer finds the new command). Then check:

```
uv --version
```

It should print a version number such as `uv 0.9.7`. The exact number does not matter.

## Step 4. Download kusim and the course

Choose a folder where the course will live. This example uses your **Documents** folder; any folder you can find again is fine.

```
cd Documents
git clone https://github.com/MElfeel/kusim.git
cd kusim
```

What happened: `cd` means "go into this folder", and `git clone` copied the whole project into a new folder called `kusim`. From now on, **every command is typed inside this `kusim` folder**. If you ever open a new terminal, go back with `cd Documents/kusim` (on Windows, `cd Documents\kusim` works too).

> Tip for Windows: if your Documents folder is synced to OneDrive and you get odd errors later, use a folder outside OneDrive, for example `cd C:\` and then `mkdir work` and `cd work` before `git clone`.

## Step 5. Install Python and all the packages

Still inside the `kusim` folder:

```
uv sync --extra dev
```

This one command:

1. downloads **Python 3.12** if you do not have it yet (it never touches any other Python on your computer),
2. creates a private folder called `.venv` inside `kusim` that holds everything (so nothing else on your computer is changed),
3. installs **kusim** and the packages it uses (NumPy, SciPy, pandas, matplotlib, plotly, pytest, and others), using the exact versions the course was written with.

It takes a few minutes the first time and prints a lot of text. That is normal. When it stops and you see your prompt again without a red error message, it worked.

## Step 6. Start JupyterLab

```
uv run --with jupyterlab jupyter lab
```

The first time, uv downloads JupyterLab (about a minute). Then a **browser tab opens** with JupyterLab. If no tab opens, look in the terminal for a line starting with `http://localhost:8888/lab?token=...`, copy it, and paste it into your browser's address bar.

In the file list on the left, open the folder `course`, then `day01`, then double-click `day01_simulation_models.ipynb`.

Two things to know:

- **Leave the terminal window open** while you work: it is the engine behind the notebook. Minimise it, do not close it.
- **To stop** JupyterLab when you finish: save your notebook (**Ctrl+S**), then click in the terminal and press **Ctrl+C**. If it asks, type `y` and press Enter.

## Step 7. Check that everything works

In JupyterLab, open a new notebook (**File, New, Notebook**; if asked for a kernel choose **Python 3**) and type this into the first cell:

```python
import kusim, numpy, pandas, matplotlib
print("kusim version", kusim.__version__)
```

Press **Shift + Enter** to run the cell. You should see a line starting with `kusim version`. If you see a red error (`ModuleNotFoundError`), see the troubleshooting list below.

A second, optional check, in the terminal (open a second terminal window, go to the `kusim` folder, and type):

```
uv run kusim version
```

It prints a short block of text that includes `"kusim": "0.1.0"` (or a newer number).

---

## Each time you study (the short version)

1. Open a terminal and go to the folder: `cd Documents/kusim`
2. *(Optional, once a week or when told)* get updates: `git pull` then `uv sync --extra dev`
3. Start the notebook program: `uv run --with jupyterlab jupyter lab`
4. When you finish, **Ctrl+S** to save and **Ctrl+C** in the terminal to stop.

Your own changes to a notebook are saved in your copy. If `git pull` complains about a notebook you changed, copy that notebook under a new name first (in JupyterLab, right-click it and choose **Duplicate**), and ask for help.

## Troubleshooting

| What you see | What to do |
|---|---|
| `uv` (or `git`) is "not recognized" / "command not found" | Close the terminal and open a new one. If it still fails, repeat the install step for that tool. |
| PowerShell says "running scripts is disabled" | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and try again. |
| The uv install command is blocked or fails with a certificate or network error (common on company or university networks) | Try a different network (for example a phone hotspot) or ask your IT office. Adding `--native-tls` after `uv` in the `uv sync` and `uv run` commands often helps (`uv sync --native-tls --extra dev`). |
| `git clone` says "destination path 'kusim' already exists" | You already downloaded it. Run `cd kusim` and carry on from Step 5. |
| `uv sync` says it cannot find a file `pyproject.toml` | You are not inside the `kusim` folder. Type `cd kusim` (or the full path) and try again. |
| The notebook says `ModuleNotFoundError: No module named 'kusim'` | JupyterLab was not started with `uv run` inside the `kusim` folder. Stop it (Ctrl+C), `cd` into `kusim`, and start it again exactly as in Step 6. |
| The browser tab does not open | Copy the `http://localhost:8888/...` address from the terminal into your browser. |
| "Port 8888 is already in use" | JupyterLab is already running in another terminal window; use that one, or close it first. |
| You pasted several commands and one of them failed | Run them one at a time and read the message after each. The last lines of an error usually say what is wrong. |
| Anything else | Copy the whole error message and the command that caused it, and send both to the course team. |

## What you installed, in one line each

- `git clone` made a copy of the project in the `kusim` folder.
- `uv sync` made the private `.venv` folder with Python and the packages inside it.
- `uv run ...` runs a command inside that private environment, so you never have to "activate" anything.
- JupyterLab is the notebook program; each `.ipynb` file is a notebook.

To remove everything later, delete the `kusim` folder (and, if you wish, uninstall git and uv like any other program). Nothing else on your computer was changed.
