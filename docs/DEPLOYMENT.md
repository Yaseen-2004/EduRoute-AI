# Deploying EduRoute AI on Streamlit Community Cloud

The app (retrieval, routing, verification, interface) runs on Streamlit
Community Cloud for free. Streamlit Cloud has no GPU, so the language and
vision models run on a Google Colab GPU during a demo and are reached over a
temporary HTTPS tunnel. Without the models, the app still works and shows
clearly labelled source excerpts in place of generated answers.

```
Student browser ──> Streamlit Cloud (app.py, FAISS index, MiniLM)
                          │  EDUROUTE_LLM_BASE_URL
                          ▼
                    Colab T4 (Ollama: qwen2.5:3b, qwen2.5vl:3b) via Cloudflare tunnel
```

## One-time setup (about 20 minutes)

1. **Create a GitHub repository.** On github.com, choose *New repository*,
   for example `eduroute-ai`. Private is fine. Do **not** add a README or a
   .gitignore; the project already has both.
2. **Push the project.** In a terminal in the project folder (the repository and
   its first commit already exist):

   ```
   git remote add origin https://github.com/<your-account>/eduroute-ai.git
   git push -u origin main
   ```

   The push is about 220 MB. It includes the 55 MB engineering index, and GitHub
   may warn about files over 50 MB; that is fine because each is under the
   100 MB limit.

   Excluded on purpose: `.env`, voice recordings, logs, grading keys (so raters
   stay blind), raw caches and the embedding copies.
3. **Create the app.** Go to <https://share.streamlit.io> and choose *Create app*,
   then *Deploy a public app from GitHub*:
   - Repository: `<your-account>/eduroute-ai`, branch `main`, main file `app.py`.
   - *Advanced settings*: Python **3.12**. In *Secrets*, paste the contents of
     `.streamlit/secrets.toml.example`; the model URLs are filled in step 2 below.
   - Deploy. The first build installs dependencies (5–10 minutes, CPU-only PyTorch)
     and downloads the MiniLM embedding model on first use.
4. **Check the app** without the models: ask "Explain the Rankine cycle". It
   should answer with labelled source excerpts and show the Rankine cycle
   sources.

The app needs about 0.9 GB of memory (measured: 813 MB loaded, 862 MB after
a question). The NLI verification model and the evaluation studio are switched
off on the hosted app through the secrets.

## Before each demo (about 10 minutes)

1. Open `colab/EduRoute_Colab_Model_Server.ipynb` in Colab, choose
   *Runtime → Change runtime type → T4 GPU*, then *Run all*.
2. Step 4 of the notebook prints a block of secrets. In Streamlit, open
   *Manage app → Settings → Secrets*, replace the four `EDUROUTE_*_URL/MODEL`
   lines with the printed ones, and save. The app restarts and reconnects.
3. In the sidebar, *System status* should now say **Language model – ready**
   (and **Image questions – ready**). Ask one question to warm up.
4. After the demo, stop step 5 of the notebook and disconnect the runtime.

The tunnel URL changes every Colab session, so repeat step 2 each time. Anyone
who has the URL can use the models while the session runs: do not share it, and
stop the session after the demo.

## Updating the app

Commit and push. Streamlit Cloud redeploys automatically:

```
git add -A
git commit -m "Describe the change"
git push
```

## Privacy on the hosted app

- Logs record a hash of each question, never its text, and are deleted after 30
  days. On Streamlit Cloud the file system is temporary, so logs are lost at every
  restart anyway.
- Questions are sent to the Colab model server while it runs. The privacy notice
  in the app sidebar says so.
- Do not deploy a real `.streamlit/secrets.toml` or `.env`; both are git-ignored.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Build fails while installing torch | Make sure the first line of `requirements.txt` (the PyTorch CPU index) is present, and that Python 3.12 is selected |
| App restarts with "resource limits" | Memory. Confirm `EDUROUTE_NLI_MODEL = ""` in the secrets; the app itself needs about 0.9 GB |
| Answers are excerpts, with "Language model unavailable" | The Colab session ended or the URL changed: re-run the server notebook and update the secrets |
| HTTP 403 from the tunnel | The tunnel must forward the host header: step 3 of the notebook passes `--http-host-header localhost:11434` |
| Image questions say "I could not analyse the image" | The vision model is not reachable: check `EDUROUTE_VISION_*` in the secrets |
| "The evaluation studio is disabled" | Expected on the hosted app; run the studio locally (`streamlit run app.py`) |
