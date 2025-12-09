# Chapter 4: Make It Build Things

Here's a secret: you don't need to know how to code to have code.

You need to know what you want. The AI writes the code. You test it. It works or it doesn't. If it doesn't, you describe what went wrong. The AI fixes it. Eventually, it works.

This isn't a hypothetical. I build tools this way constantly. Automations that check things automatically. Scripts that process files. Mini-apps that solve specific problems. I don't write the code—I direct someone who does.

That someone just happens to be an AI that works for free.

This chapter shows you how.

---

## The Mental Model: You're the Director

Think of yourself as a film director. You don't operate the camera, arrange the lighting, or edit the footage. You tell the people who do those things what you want.

"I need this scene to feel tense. The lighting should be dim, shadows in the corner. When the actor opens the door, I want a slow push-in on their face."

The cinematographer and editor figure out how to achieve that.

You're doing the same thing with code.

"I need a tool that checks a list of URLs every hour and alerts me if any of them go down. It should send me a text message when something breaks."

The AI figures out how to achieve that.

Your job: know what you want, describe it clearly, test the result, provide feedback.

Not your job: understand how the code works, write syntax, debug compiler errors.

---

## The Basic Loop

Every AI-assisted build follows this pattern:

1. **Describe** what you want
2. **Receive** code from the AI
3. **Run** the code
4. **Report** what happened (success or failure)
5. **Iterate** until it works

Let's walk through each step.

### Step 1: Describe What You Want

Be specific about:

- **What it should do** (the action)
- **When it should do it** (trigger or schedule)
- **What it needs** (inputs)
- **What it produces** (outputs)
- **Where it runs** (your computer, a server, a browser)

**Bad description:**
> "Make me a script that deals with my emails."

**Good description:**
> "Make me a Python script that:
> - Connects to my Gmail inbox
> - Finds emails with 'Invoice' in the subject line from the past 7 days
> - Downloads any PDF attachments
> - Saves them to a folder called 'invoices' on my desktop
> - Prints a summary of what it downloaded"

The second version gives the AI everything it needs.

### Step 2: Receive Code

The AI will produce code. It might look like gibberish to you. That's fine.

What to look for:
- Does the AI explain what the code does in plain English?
- Does it tell you how to run it?
- Does it warn you about any setup needed?

If it doesn't explain these things, ask: "Walk me through what this does and how to use it."

### Step 3: Run the Code

Here's where many people freeze. "I don't know how to run code."

The basics:

**For Python scripts:**
1. Save the code as a file ending in `.py` (like `my_script.py`)
2. Open a terminal/command prompt
3. Type `python my_script.py` and hit enter

**For web stuff (HTML/JavaScript):**
1. Save as a file ending in `.html`
2. Double-click to open in your browser

**For Google Sheets/Excel automations:**
1. The AI will usually tell you where to paste it (Script Editor for Sheets)
2. Paste, save, run

If you don't know how to run something, just ask: "I'm on a Mac/Windows/etc. How do I run this?"

### Step 4: Report What Happened

Three possible outcomes:

**It worked:** Great. Tell the AI what worked and what you want to add or change.

**It crashed (error message):** Copy the entire error message and paste it to the AI. Don't try to interpret it—just paste the whole thing.

> "I got this error: [paste error]"

**It ran but did the wrong thing:** Describe the difference between expected and actual behavior.

> "It ran without errors, but it only found 2 emails when I know there are 10 with 'Invoice' in the subject. The ones it missed have 'INVOICE' in caps—maybe it's case sensitive?"

### Step 5: Iterate

Rarely works perfectly the first time. Usually takes 2-5 rounds. This is normal.

Each round, you give feedback, the AI adjusts, you test again.

The AI learns your specific environment as you go: "Oh, your Gmail needs an app password not your regular password." "Your Python version needs this different syntax." 

By round 3-4, it's dialing in.

---

## Real Example: Building a Link Checker

Let me walk through an actual build session.

### Round 1

**Me:**
> "I have a website with about 200 pages. I want to find broken links—links that go to pages that don't exist anymore. Build me a tool to crawl my site and report dead links."

**AI:**
```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

def check_links(start_url, max_pages=200):
    visited = set()
    broken_links = []
    to_visit = [start_url]
    base_domain = urlparse(start_url).netloc
    
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
            
        visited.add(url)
        print(f"Checking: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code >= 400:
                broken_links.append((url, response.status_code))
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == base_domain:
                    if full_url not in visited:
                        to_visit.append(full_url)
                        
        except Exception as e:
            broken_links.append((url, str(e)))
            
        time.sleep(0.5)  # Be polite to the server
        
    return broken_links

# Usage
start = "https://yourwebsite.com"
broken = check_links(start)
print("\n=== BROKEN LINKS ===")
for link, status in broken:
    print(f"{link} - {status}")
```

**AI also says:** "You'll need to install requests and beautifulsoup4. Run: `pip install requests beautifulsoup4`"

### Round 2

**Me:**
> "I ran it and got this error: ModuleNotFoundError: No module named 'requests'"

**AI:**
> "You need to install the required packages. Open your terminal and run:
> `pip install requests beautifulsoup4`
> Then try running the script again."

**Me:**
> "Ok that worked. But it found 500 links and is still running. I only want to check MY site, not external links."

**AI adjusts code to skip external links**

### Round 3

**Me:**
> "Better. But it's reporting links to PDFs as broken when they actually work."

**AI:**
> "PDFs might need different handling. Let me add a check to skip media files or test them differently."

**AI adjusts code to handle PDFs**

### Round 4

**Me:**
> "Perfect. Can you save the results to a CSV file so I can open it in Excel?"

**AI adds CSV export**

Final tool works. Took about 15 minutes and 4 rounds of feedback.

---

## What You Can Build

Here's what's within reach—no coding knowledge required:

### Data Processing

- Combine multiple spreadsheets into one
- Clean up messy data (fix formatting, remove duplicates)
- Convert between file formats (CSV to JSON, etc.)
- Extract specific information from documents

### Automation

- Download files from websites automatically
- Rename/organize files based on rules
- Send scheduled emails or messages
- Monitor websites for changes

### Simple Web Tools

- Forms that collect and store data
- Calculators for specific purposes
- Dashboards showing information
- Simple interactive pages

### API Integrations

- Connect different services (more on this below)
- Pull data from one place, push to another
- Create reports from multiple sources

### Analysis

- Generate summaries and reports
- Find patterns in data
- Compare information across files
- Track changes over time

---

## APIs: Connecting Services

APIs are how different software talks to each other. They're simpler than they sound.

Think of an API as a waiter at a restaurant. You tell the waiter what you want (your order). The waiter takes it to the kitchen (the service). The kitchen prepares it and gives it back to the waiter. The waiter brings it to you (the response).

You don't need to know how the kitchen works. You just need to know how to place an order.

### API Keys: Your Permission Slip

Most APIs require an "API key"—basically a password that identifies you and grants permission.

Getting an API key usually looks like:
1. Create an account on the service
2. Go to "Developer" or "API" settings
3. Click "Generate API key" or "Create new key"
4. Copy the long string of letters and numbers

**Important:** API keys are sensitive. Don't share them, post them publicly, or paste them where others can see.

### Example: Building a Weather Alert Tool

**What I want:** A tool that checks the weather every morning and texts me if it's going to rain.

**Services needed:**
- Weather API (to get forecast) — OpenWeatherMap is free
- SMS API (to send texts) — Twilio has free tier

**Me:**
> "Build a Python script that:
> 1. Gets the weather forecast for [my city] from OpenWeatherMap
> 2. Checks if rain is predicted in the next 12 hours
> 3. If yes, sends me a text via Twilio
> 4. I want to run this every morning at 7 AM
> 
> I have API keys for both services. Where do I put them?"

The AI produces code with placeholders:
```python
OPENWEATHER_KEY = "your-key-here"
TWILIO_KEY = "your-key-here"
```

You fill in your keys, test it, iterate until it works.

**Setting up the schedule** (running every morning):
- On Mac: Ask AI to help set up a "cron job" or "launchd"
- On Windows: Ask AI to help set up "Task Scheduler"

---

## Testing Without Breaking Things

A reasonable fear: "What if the code does something bad?"

Some safeguards:

### Start with Dry Runs

Ask for a "dry run" version first.

> "Before it actually sends emails, can it just print what it WOULD send so I can verify?"

### Use Test Data

Don't run on your real data first.

> "Create a small test file I can use to verify this works before running it on my actual 5,000-row spreadsheet."

### Add Confirmations

> "Before it deletes anything, make it ask for confirmation."

### Back Up First

Before running anything that modifies files:
> "What files will this change? Should I back anything up first?"

### Sandboxed Testing

If you're working with services (email, SMS, etc.), many have "sandbox" or "test" modes.

> "Can we test this in Twilio's test mode first so it doesn't actually send texts?"

---

## When You Get Stuck

Sometimes things don't work and you don't know why.

### The Magic Phrase

> "I'm not sure how to debug this. Can you explain what might be going wrong and give me specific things to check?"

### Useful Information to Provide

When stuck, tell the AI:

1. **What you tried:** "I ran the script by typing python script.py"
2. **What happened:** "The terminal showed [this error/nothing/something unexpected]"
3. **Your environment:** "I'm on Windows 11, using Python 3.9"
4. **What you expected:** "I expected it to create a file called output.csv"

The more specific you are, the faster the fix.

### When to Try a Different Approach

If you're on round 10 and still stuck, sometimes it's the approach, not the implementation.

> "We've been going back and forth on this. Is there a simpler way to achieve [original goal] that might work better?"

Often there's an easier path.

---

## Building Reusable Tools

Once you build something that works, save it for reuse.

### Document What It Does

Add a comment at the top:
```python
# Tool: Invoice Downloader
# What it does: Downloads PDF invoices from Gmail
# How to run: python download_invoices.py
# Requires: Gmail app password in gmail_creds.txt
# Last updated: [date]
```

### Create a "Tools" Folder

Organize your scripts by function:
```
/my-tools
  /email
    download_invoices.py
    send_weekly_report.py
  /data
    clean_csv.py
    merge_spreadsheets.py
  /web
    check_links.py
    monitor_prices.py
```

### Keep a Simple Log

A text file or spreadsheet:
- Tool name
- What it does
- When you last used it
- Any quirks or notes

---

## Leveling Up

As you build more tools, you'll notice patterns:

**The 80/20 rule:** 80% of the work is describing clearly what you want. 20% is the back-and-forth debugging.

**Compound skills:** Each build teaches you something. "Oh, that's what a JSON file is." "Oh, that's how environment variables work." You'll recognize patterns.

**Faster iteration:** Your round count drops. First tools take 5-10 rounds. Later tools take 2-3.

**Bigger ambition:** What started as simple scripts becomes integrations, automations, even small apps.

---

## Summary

- **You don't need to code** to have code. You describe, AI builds, you test, AI fixes.

- **Follow the loop:** Describe → Receive → Run → Report → Iterate.

- **Be specific** about what, when, inputs, outputs, and where.

- **APIs connect services.** API keys are just permission slips. Don't share them.

- **Test safely:** Dry runs, test data, confirmations, backups.

- **Save what works.** Organize your tools. Document them. Reuse them.

The next chapter tackles the elephant in the room: AI gets things wrong. Confidently, fluently wrong. How do you know when to trust it and when to verify?
