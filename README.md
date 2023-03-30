# MediBuddy

MediBuddy is an AI Medical Chat tool designed to provide users with human-like and empathetic responses to their medical queries. This project consists of a Jupyter notebook, and includes a comparative analysis of different natural language processing solutions, such as BERT and GPT-2. The AI system is trained with a comprehensive medical knowledge base to ensure accurate responses, and is optimized for efficient and timely performance.

## Getting Started - Colab Notebook

To run MediBuddy in a Colab notebook, follow these steps:

1. Open the following link in your browser: https://colab.research.google.com/github/martinteoharov/MediBuddy/blob/main/main.ipynb
2. Click the "Copy to Drive" to make your own copy of the notebook.
3. Follow the instructions in the notebook to run MediBuddy.

## Pushing to GitHub from Colab

1. Open the Colab notebook that you want to push changes to.
2. Click on "File" in the top left corner, then "Save a copy in GitHub." This will open the GitHub integration panel.
3. In the GitHub integration panel, select the repository you want to push changes to, enter a commit message, and click "OK."
4. Wait for the changes to be pushed to GitHub. You can check the status of the push in the output panel at the bottom of the screen.

## Pulling from GitHub to Colab

1. Open the Colab notebook that you want to pull changes to.
2. Click on "File" in the top left corner, then "Open notebook."
3. In the "GitHub" tab of the "Open notebook" dialog, select the repository and branch you want to pull changes from.
4. Click "Open." Colab will download the latest version of the notebook from GitHub and open it in a new tab.

Note that if you have made any changes to the Colab notebook that you haven't saved to GitHub, you will need to save a copy

## Getting Started - Locally

To run MediBuddy locally on your computer, follow these steps:

1. Clone this repository to your local machine
```
git clone https://github.com/martinteoharov/MediBuddy.git
```

2. Install Pyenv by running the following commands in your terminal or command prompt:
```
curl https://pyenv.run | bash
```

3. Install Pipenv by running the following command in your terminal or command prompt:
```
pip install pipenv
```

4. Navigate to the project directory and run the following command to install the required packages:
```
cd MediBuddy/
pipenv install
```

5. Run the following command to start the Jupyter notebook:
```pipenv run jupyter notebook```

6. Open `main.ipynb` in your browser by visiting http://localhost:8888/notebooks/main.ipynb

## Requirements

This project requires Python 3.7, Pyenv, and Pipenv. It has been tested on Python 3.7.

## Contributing

If you'd like to contribute to MediBuddy, please open an issue or submit a pull request. We welcome contributions of all kinds, including bug fixes, new features, and documentation updates.

## License

This project is licensed under the MIT License - see the LICENSE file for details.







