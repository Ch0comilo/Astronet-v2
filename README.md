
🚀 Getting Started with Astronet-Triage
Follow these steps to set up your environment and prepare the Predict-fa1t-38-test notebook for inference.

1. Create the Virtual Environment
Open your terminal in the project root folder and run:

Bash
python -m venv venv
2. Activate the Environment
Depending on which terminal you are using, choose the appropriate command:

Powershell:

PowerShell
.\venv\Scripts\Activate.ps1
Command Prompt (CMD):

DOS
venv\Scripts\activate
macOS / Linux / WSL:

Bash
source venv/bin/activate
3. Install Dependencies
Once the environment is active (you should see (venv) in your prompt), upgrade pip and install the required libraries:

Bash
pip install --upgrade pip
pip install -r requirements.txt
4. Configure the Notebook
To ensure your notebook uses the correct environment, follow these steps inside your IDE (VS Code or Jupyter):

Open Predict-fa1t-38-test.ipynb.

In the top right corner, click on Select Kernel.

Choose Python Environments...

Select the interpreter located at ./venv/Scripts/python.exe.

✨ You’re ready!
You can now run the cells in your notebook to start predicting exoplanet candidates using the fa1t_38 model weights.