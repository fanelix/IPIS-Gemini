# IPI Dashboard Deployment Guide
## GitHub + Streamlit Cloud Step-by-Step

This guide walks you through deploying the IPI Cumulative Displacement Dashboard from your local machine to a live web application using GitHub and Streamlit Cloud (free tier).

---

## Prerequisites

Before starting, ensure you have:
- [ ] A GitHub account (free) → [github.com](https://github.com)
- [ ] A Streamlit Cloud account (free) → [streamlit.io/cloud](https://streamlit.io/cloud)
- [ ] Git installed on your computer
- [ ] The `app.py` file downloaded

---

## Part 1: Prepare Your Project Files

### Step 1.1: Create Project Folder

Create a new folder on your computer for the project:

```bash
# Windows (Command Prompt)
mkdir ipi-dashboard
cd ipi-dashboard

# Mac/Linux (Terminal)
mkdir ipi-dashboard
cd ipi-dashboard
```

### Step 1.2: Add Required Files

Your project folder needs these 3 files:

```
ipi-dashboard/
├── app.py              # Main application (already created)
├── requirements.txt    # Python dependencies
└── README.md           # Project description (optional but recommended)
```

### Step 1.3: Create requirements.txt

Create a file named `requirements.txt` with these contents:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
```

### Step 1.4: Create README.md (Optional)

Create a file named `README.md`:

```markdown
# IPI Cumulative Displacement Dashboard

A Streamlit web application for visualizing In-Place Inclinometer (IPI) monitoring data.

## Features
- Auto-detection of Campbell Scientific TOA5 format
- Configurable gauge length and sensor depths
- Base reading correction (initial reading subtraction)
- Bottom-up cumulative displacement calculation
- Interactive Plotly charts

## Usage
1. Upload your IPI data file (.dat or .csv)
2. Configure sensor parameters in the sidebar
3. Explore displacement profiles and trends

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
```

---

## Part 2: Create GitHub Repository

### Step 2.1: Log in to GitHub

1. Go to [github.com](https://github.com)
2. Sign in to your account (or create one if needed)

### Step 2.2: Create New Repository

1. Click the **+** icon in the top-right corner
2. Select **"New repository"**

![New Repo Button](https://docs.github.com/assets/images/help/repository/repo-create.png)

3. Fill in the repository details:

| Field | Value |
|-------|-------|
| Repository name | `ipi-dashboard` |
| Description | `IPI Cumulative Displacement Dashboard` |
| Visibility | **Public** (required for free Streamlit hosting) |
| Initialize | ☐ Do NOT check any boxes |

4. Click **"Create repository"**

### Step 2.3: Note Your Repository URL

After creation, you'll see a page with setup instructions. Copy your repository URL:
```
https://github.com/YOUR_USERNAME/ipi-dashboard.git
```

---

## Part 3: Upload Code to GitHub

You have two options: **Command Line (Recommended)** or **Web Upload**.

### Option A: Using Git Command Line (Recommended)

#### Step 3A.1: Initialize Git Repository

Open Terminal/Command Prompt in your project folder:

```bash
# Navigate to your project folder
cd ipi-dashboard

# Initialize git
git init
```

#### Step 3A.2: Add Files

```bash
# Add all files to staging
git add .

# Verify files are staged
git status
```

You should see:
```
new file:   app.py
new file:   requirements.txt
new file:   README.md
```

#### Step 3A.3: Commit Files

```bash
git commit -m "Initial commit: IPI Dashboard"
```

#### Step 3A.4: Connect to GitHub

```bash
# Add remote repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ipi-dashboard.git

# Rename branch to main
git branch -M main
```

#### Step 3A.5: Push to GitHub

```bash
git push -u origin main
```

If prompted, enter your GitHub credentials. 

> **Note:** If you have 2FA enabled, you'll need a Personal Access Token instead of password.
> Go to: GitHub → Settings → Developer settings → Personal access tokens → Generate new token

#### Step 3A.6: Verify Upload

1. Go to `https://github.com/YOUR_USERNAME/ipi-dashboard`
2. You should see all 3 files listed

---

### Option B: Using GitHub Web Interface (Easier)

#### Step 3B.1: Upload Files via Browser

1. Go to your repository: `https://github.com/YOUR_USERNAME/ipi-dashboard`
2. Click **"uploading an existing file"** link
3. Drag and drop all 3 files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Scroll down and click **"Commit changes"**

---

## Part 4: Deploy to Streamlit Cloud

### Step 4.1: Access Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in"** → **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub

### Step 4.2: Create New App

1. Click **"New app"** button (top-right)

### Step 4.3: Configure Deployment

Fill in the deployment form:

| Field | Value |
|-------|-------|
| Repository | `YOUR_USERNAME/ipi-dashboard` |
| Branch | `main` |
| Main file path | `app.py` |

![Streamlit Deploy Form](https://docs.streamlit.io/images/streamlit-community-cloud/deploy-an-app.png)

### Step 4.4: Advanced Settings (Optional)

Click **"Advanced settings"** if you need to:
- Set Python version (3.9, 3.10, or 3.11 recommended)
- Add secrets (not needed for this app)

### Step 4.5: Deploy

1. Click **"Deploy!"** button
2. Wait 2-5 minutes for deployment
3. You'll see a build log showing progress

### Step 4.6: Access Your Live App

Once deployed, your app will be available at:
```
https://YOUR_USERNAME-ipi-dashboard-app-XXXXX.streamlit.app
```

You can customize this URL in settings later.

---

## Part 5: Managing Your Deployment

### Updating Your App

When you make changes to your code:

#### Via Command Line:
```bash
# Make changes to app.py
git add .
git commit -m "Update: description of changes"
git push
```

#### Via GitHub Web:
1. Go to your repository
2. Click on `app.py`
3. Click pencil icon (Edit)
4. Make changes
5. Click "Commit changes"

**Streamlit Cloud automatically redeploys when you push to GitHub!**

### Viewing Logs

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. Click **"Manage app"** (bottom-right corner)
4. Select **"Logs"** to view deployment/error logs

### Rebooting App

If your app becomes unresponsive:
1. Go to Streamlit Cloud dashboard
2. Click **"Manage app"**
3. Click **"Reboot app"**

### Deleting App

1. Go to Streamlit Cloud dashboard
2. Click the **⋮** menu on your app
3. Select **"Delete"**

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError"
**Solution:** Ensure all packages are in `requirements.txt`

#### 2. "No module named 'streamlit'"
**Solution:** Check `requirements.txt` exists and contains `streamlit`

#### 3. App crashes on file upload
**Solution:** Streamlit Cloud has a 200MB memory limit on free tier. Large files may cause issues.

#### 4. "Permission denied" during git push
**Solution:** Use Personal Access Token instead of password:
```bash
git remote set-url origin https://TOKEN@github.com/YOUR_USERNAME/ipi-dashboard.git
```

#### 5. App shows old version
**Solution:** Clear browser cache or force refresh (Ctrl+Shift+R)

### Resource Limits (Free Tier)

| Resource | Limit |
|----------|-------|
| Apps | 3 private apps |
| Memory | 1 GB |
| CPU | Shared |
| Sleep | After 7 days of inactivity |

---

## Quick Reference Commands

```bash
# Clone existing repo
git clone https://github.com/YOUR_USERNAME/ipi-dashboard.git

# Check status
git status

# Add all changes
git add .

# Commit with message
git commit -m "Your message"

# Push to GitHub
git push

# Pull latest changes
git pull
```

---

## Summary Checklist

- [ ] Created project folder with 3 files
- [ ] Created GitHub repository (public)
- [ ] Pushed code to GitHub
- [ ] Connected Streamlit Cloud to GitHub
- [ ] Deployed app
- [ ] Tested live URL

---

## Your App URLs

After deployment, fill in your URLs:

| Item | URL |
|------|-----|
| GitHub Repo | `https://github.com/___________/ipi-dashboard` |
| Live App | `https://___________-ipi-dashboard-app-_____.streamlit.app` |

---

## Need Help?

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Community:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Docs:** [docs.github.com](https://docs.github.com)
