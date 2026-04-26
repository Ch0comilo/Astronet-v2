# Helper setup guide — k-fold ensemble training

Thanks for helping! This guide walks you through cloning the repo, setting up
Python, and running your assigned training jobs. The whole pipeline trains 10
CNN models (5 folds × 2 random seeds) for a stacked ensemble. You'll be
running a subset of those 10 jobs on your machine.

Pick the section below that matches your OS.

---

## What you'll need before starting

1. **Python 3.10** (other 3.x versions may break TensorFlow 2.15).
2. **The 5 fold tfrecord files.** They're not in the repo because GitHub
   rejects files over 100 MB. I'll send a Google Drive (or similar) link with
   `fold_0.tfrecord` … `fold_4.tfrecord` (~360 MB each, ~1.8 GB total).
3. **Your assigned job IDs** (I'll tell you, e.g. `0,1,2`).
4. **~10 GB free disk space** (data + checkpoints).

---

## Option A — Windows with NVIDIA GPU

TensorFlow GPU on native Windows is broken past TF 2.10, and we need 2.15.
**Use WSL2 Ubuntu** — it gives you a Linux environment that talks directly to
your Windows GPU. Setup takes ~15 min.

### 1. Install WSL2 + Ubuntu

Open PowerShell **as Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot when prompted. On first launch Ubuntu will ask you to create a
username and password. From here on, **all commands go in the Ubuntu
terminal** (not PowerShell).

### 2. Verify GPU is visible from WSL

```bash
nvidia-smi
```

Should print a table showing your GPU. If it doesn't, install the latest
NVIDIA Game Ready / Studio driver on **Windows** (not inside WSL) — WSL
inherits the Windows driver.

### 3. Install Python 3.10 and project deps

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip git

# Clone the repo into your home directory
cd ~
git clone https://github.com/Ch0comilo/Astronet-v2.git Astronet-Triage
cd Astronet-Triage

# Create isolated Python environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify TensorFlow sees the GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should print `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.
If it prints `[]`, the GPU isn't being detected — message me, don't proceed.

### 5. Drop the fold data into place

Download the 5 fold tfrecords from the link I sent. From inside Ubuntu/WSL,
your Windows Downloads folder is at `/mnt/c/Users/<your-windows-username>/Downloads/`.

```bash
mkdir -p data/folds
cp /mnt/c/Users/<your-windows-username>/Downloads/fold_*.tfrecord data/folds/
ls -lh data/folds/   # should list 5 files, ~360 MB each
```

### 6. Run your assigned jobs

Replace `<jobs>` with the comma-separated list I gave you (e.g. `0,1,2`):

```bash
bash astronet/ensemble_train_kfold_distributed.sh <jobs>
```

Each job takes ~8 min on an RTX 4060. Expect roughly 8 min × number of jobs.

---

## Option B — MacBook Pro

Macs have no NVIDIA GPU, so we use either Apple's Metal backend (Apple
Silicon M1/M2/M3/M4 — much faster) or plain CPU (Intel Macs — slow).

### 1. Install Homebrew + Python 3.10

If you don't have Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then:

```bash
brew install python@3.10 git
```

### 2. Clone the repo

```bash
cd ~
git clone https://github.com/Ch0comilo/Astronet-v2.git Astronet-Triage
cd Astronet-Triage
```

### 3. Create the Python env

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Apple Silicon only) Add the Metal GPU plugin

Skip this step on Intel Macs — Metal isn't supported.

```bash
pip install tensorflow-metal
```

Verify:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

On Apple Silicon you should see a `GPU` device. On Intel, `[]` is fine — it
will fall back to CPU.

### 5. Drop the fold data into place

Download the 5 fold tfrecords from the link I sent (default download
location is `~/Downloads`):

```bash
mkdir -p data/folds
cp ~/Downloads/fold_*.tfrecord data/folds/
ls -lh data/folds/   # should list 5 files, ~360 MB each
```

### 6. Run your assigned jobs

```bash
bash astronet/ensemble_train_kfold_distributed.sh <jobs>
```

**Speed expectations on Mac:**
- Apple Silicon (M1/M2/M3/M4) with `tensorflow-metal`: ~15–25 min per job.
- Intel Mac (CPU only): could be 1–2 hours per job. Probably only worth it
  if you take 1–2 jobs at most.

If your Mac is slow, take fewer jobs (1 or 2) and let the Windows machine
handle the rest.

---

## What gets produced

Each completed job writes a checkpoint directory:

```
checkpoints/fa1t_38_kfold/fold<k>_seed<s>/AstroCNNModel_final_alpha_1_tuned_<timestamp>/
```

For example, if you ran jobs `0,1,2` you'll have:

```
checkpoints/fa1t_38_kfold/fold0_seed1/...
checkpoints/fa1t_38_kfold/fold0_seed2/...
checkpoints/fa1t_38_kfold/fold1_seed1/...
```

---

## Sending results back

Once your jobs finish, zip the checkpoints and send them back:

```bash
cd checkpoints/fa1t_38_kfold
tar -czf my_share.tar.gz fold*_seed*/
ls -lh my_share.tar.gz
```

Upload `my_share.tar.gz` to Google Drive / WeTransfer / Dropbox and share
the link with me. Each checkpoint dir is ~50–100 MB, so total upload size
depends on how many jobs you ran.

---

## Job ID reference

| Job ID | Fold | Seed | Output dir                                       |
|--------|------|------|--------------------------------------------------|
| 0      | 0    | 1    | `checkpoints/fa1t_38_kfold/fold0_seed1`          |
| 1      | 0    | 2    | `checkpoints/fa1t_38_kfold/fold0_seed2`          |
| 2      | 1    | 1    | `checkpoints/fa1t_38_kfold/fold1_seed1`          |
| 3      | 1    | 2    | `checkpoints/fa1t_38_kfold/fold1_seed2`          |
| 4      | 2    | 1    | `checkpoints/fa1t_38_kfold/fold2_seed1`          |
| 5      | 2    | 2    | `checkpoints/fa1t_38_kfold/fold2_seed2`          |
| 6      | 3    | 1    | `checkpoints/fa1t_38_kfold/fold3_seed1`          |
| 7      | 3    | 2    | `checkpoints/fa1t_38_kfold/fold3_seed2`          |
| 8      | 4    | 1    | `checkpoints/fa1t_38_kfold/fold4_seed1`          |
| 9      | 4    | 2    | `checkpoints/fa1t_38_kfold/fold4_seed2`          |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'astronet'`**
You forgot to activate the venv, or you ran the script from outside the
repo root. Run `source .venv/bin/activate` and `cd` into `Astronet-Triage/`
before launching the script.

**`Your input ran out of data`**
The fold tfrecord files aren't in `data/folds/`. Re-check step 5.

**Training is using CPU instead of GPU**
- Windows/WSL: re-run `nvidia-smi` inside WSL. If empty, update the Windows
  NVIDIA driver.
- Mac Apple Silicon: confirm `pip install tensorflow-metal` succeeded.

**Out of GPU memory**
Edit `astronet/astro_cnn_model/configurations.py` and lower `batch_size`
(currently it's set in the `final_alpha_1_tuned` config). Halve it and try
again.

**Anything else** — message me with the full error output.
