#./CONTRIBUTING.md
# Contributing to Matrix to Discourse Bot

## Prerequisites
1. Fork the repository.https://github.com/irregularchat/matrix-to-discourse/tree/main
3. Read the [ROADMAP.md](./ROADMAP.md) to understand the current status of the project and the planned features.
Needed for Full Testeding but not for error testing.
1. Request access to the maubot dashboard. https://matrix.irregularchat.com/_matrix/maubot
2. Request api access for Discourse and OpenAI from sac.

## Environment Setup
git clone the forked repository you made earlier.

``` bash
python3 -m venv env  # Create the virtual environment
source env/bin/activate  # Activate the virtual environment (on Linux/Mac)
python3 -m pip install -r requirements.txt
```

## Switch to the right branch
``` bash
git checkout <branch-name>
```

## Updating
1. Review the code and make updates with detailed commit messages.
2. Test the changes by running the bot locally.
3. Push your changes to your forked repository.
4. Create a pull request to the main repository.

## Running the bot locally
1. Make sure you have the correct branch checked out.
1. Update maubot.yaml with the correct test version of the plugin. 
1. Create .mbp file in the plugins directory with the following:
```bash
# These are the current files in the repository needed to run the bot.
zip -9r matrix-to-discourse-testing.mbp bot.py maubot.yaml config.yaml requirements.txt
```
1. Initial Testing: `python3 -m maubot` This will tell you if there are any errors in the plugin that need to be fixed.
1. Upload the matrix-to-discourse-testing.mbp file to the maubot dashboard, plugin section.
1. Create a new instance selecting the testing version of the plugin and irregularchatbot as the bot.
1. Allow testing bot only in the testing room. 
```copy
!EzjSxyqeDljGcDDcFQ:irregularchat.com
```
1. Test the bot by sending it a message in the Bot Testing room it is in.