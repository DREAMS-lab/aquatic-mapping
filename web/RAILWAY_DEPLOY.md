# Deploy to Railway

## Step 1: Sign Up for Railway
1. Go to https://railway.app
2. Click "Start a New Project"
3. Sign up with GitHub (free account)

## Step 2: Install Railway CLI (Optional)
```bash
npm install -g @railway/cli
```

Or deploy via GitHub (easier):

## Step 3: Push to GitHub
```bash
cd ~/workspaces/aquatic-mapping/web
git init
git add .
git commit -m "Initial commit for Railway deployment"
git branch -M main
# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/aquatic-mapping-web.git
git push -u origin main
```

## Step 4: Deploy on Railway
1. Go to https://railway.app/new
2. Select "Deploy from GitHub repo"
3. Choose your aquatic-mapping-web repo
4. Railway will auto-detect and deploy!

## Step 5: Set Environment Variables
In Railway dashboard, go to Variables and add:
- `SIM_USERNAME`: your username (optional, defaults to "bakin")
- `SIM_PASSWORD`: your password (optional, defaults to "ozhugu")

## Step 6: Get Your URL
Railway will give you a URL like: `https://your-app.railway.app`

## Step 7: Connect Your PC (When You Want to Run Simulations)
On your PC, keep the Cloudflare tunnel running to allow Railway to communicate with your local Docker.

## Cost
- Free tier: $5 credit/month
- After that: ~$5/month for this app size

## Features
- ✅ Website always online
- ✅ View reconstruction results
- ✅ Generate comparison heatmaps
- ❌ Can't start Docker containers (only when PC is connected)
- ❌ No GPU access (stays on your PC)
