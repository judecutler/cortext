# Where recordings are stored (decision) and what you need to do

## Decision

| What | Where | Why |
|---|---|---|
| Code, docs | **GitHub** | Small text files, reviewed by pull request. |
| Recordings (EEG, trials, events) | **A private Hugging Face dataset repo** | Built for large files, versioned, access control per person/org, `huggingface_hub` downloads only what you ask for. |
| Your working copy | `data/cortext-p300/` on your machine (git-ignored) | The app, pipeline and `load_epochs()` read from here. |

Why not GitHub for data: every recording permanently grows the repo (git never forgets), the 100 MB file limit is close
for raw EEG, and a deleted session would stay in history. Why not a shared drive: no versioning or per-file access,
and "who has the latest" problems. Hugging Face stays simple: **one folder per session**, so two laptops uploading at once
never touch the same file, and there is no shared index file to conflict on.

Raw EEG is large, so teammates normally download only the small `epochs.npz` files (`pull`) and fetch everything with `pull --all`.

## What you need to do (once)

1. Make a Hugging Face account (free) at huggingface.co. For a team, create an **organization** and invite teammates (write access for recorders, read for others).
2. Create an **access token** (Settings → Access Tokens) with *write* permission to your repo.
3. On each machine that records or downloads: `pip install huggingface_hub`, then `hf auth login` (paste the token). Or set the `HF_TOKEN` environment variable. The token is never stored in this repo.
4. Edit `datasets/cortext-p300/dataset.json`: set `"hf_repo": "your-name-or-org/cortext-p300"`. Commit that one line so teammates share it.
5. Create the repo (private): `python datasets/cortext-p300/cortext_store.py init` (or the **Create repo** button on the Setup tab).
6. Check: `python datasets/cortext-p300/cortext_store.py status` (or **Check connection** on the Setup tab).
7. Have old sessions from the in-repo folder? `python datasets/cortext-p300/cortext_store.py migrate`, then `push`.

After that: the Setup tab shows what is uploaded; tick **Upload automatically** to upload each session when it ends, or press **Upload pending**.

## Deleting

Delete on the **Experiment → Recorded sessions** page. You choose which copies go: the browser, the bridge logs, the dataset
folder on this computer, and (separately, with a typed confirmation) the Hugging Face copy including its stored history.
Sessions that are still recording cannot be deleted. Copies teammates already downloaded stay with them.

## Buckets

A bucket is just a label on a session (`pilot`, `final`, `bad-electrodes`…) stored in its `meta.json`. It is not a folder, so
moving a session between buckets is instant and cannot lose data. Use `load_epochs(bucket="final")` to train on one bucket.
Changing a bucket marks the session as "not uploaded" so the next upload carries the new label.

## Superseded sessions (redone in the app)

When the operator presses **Redo this session** (or **Restart session**), the first recording is kept and its
`meta.json` gets `superseded_by: <id of the redo>`; the redo has `redo_of`. `load_epochs`, `load_imagery`,
`list_sessions` and `train_p300.py` leave superseded sessions out by default. Pass `include_superseded=True` /
`--include-superseded`, or name the session in `sessions=[...]` / `--sessions`, to use it anyway. The operator can
remove the mark from the session's summary page (*Use it after all*); upload again to share the change.

## Privacy

Keep the repo **private**. Use participant codes, never names; session notes are included in `meta.json`.

## Honest status

The Hugging Face code was tested only against a faked `HfApi` (the real service needs your account), so run `status` and
a first `push` of one test session yourself before relying on it.
