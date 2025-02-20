# Prerequisites

## Setup virtual environment
1. Create a virtual environment
```bash
python3 -m venv venv
```

2. Activate the virtual environment
```bash
source venv/bin/activate
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

When opening a new terminal, just activate the virtual environment by running `source venv/bin/activate`.


# Contributing

Create a new branch from `main` branch, and create a pull request to merge back to `main` branch. 
```
git checkout main
git pull origin main
git checkout -b github_username/<branch-name>
```

To simplify the merge you can rebase your branch on top of the main branch before creating the pull request.
```
git checkout main
git pull origin main
git checkout github_username/<branch-name>
git rebase main
```

After resolving any conflicts, you can push your branch to the remote repository.
```
git push origin github_username/<branch-name>
```

If push is rejected, you can force push your branch to the remote repository.
```
git push --force origin github_username/<branch-name>
```

