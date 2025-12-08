# API Key Testing Guide

This guide explains how to test your Anthropic, XAI, and OpenAI API keys.

## Quick Start

1. **Install required packages:**
   ```bash
   pip install anthropic openai python-dotenv
   ```

2. **Run the test script:**
   ```bash
   python test_api_keys.py
   ```

## Ways to Provide API Keys

### Option 1: Environment Variables (Recommended)

Set the keys as environment variables before running the script:

**Windows PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY='your-key-here'
$env:XAI_API_KEY='your-key-here'
$env:OPENAI_API_KEY='your-key-here'
python test_api_keys.py
```

**Windows CMD:**
```cmd
set ANTHROPIC_API_KEY=your-key-here
set XAI_API_KEY=your-key-here
set OPENAI_API_KEY=your-key-here
python test_api_keys.py
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY='your-key-here'
export XAI_API_KEY='your-key-here'
export OPENAI_API_KEY='your-key-here'
python test_api_keys.py
```

### Option 2: Command-Line Arguments

Pass keys directly as arguments:

```bash
python test_api_keys.py --anthropic-key YOUR_KEY --xai-key YOUR_KEY --openai-key YOUR_KEY
```

### Option 3: .env File

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your-anthropic-key-here
XAI_API_KEY=your-xai-key-here
OPENAI_API_KEY=your-openai-key-here
```

Then run:
```bash
python test_api_keys.py
```

**Note:** Make sure `.env` is in your `.gitignore` to avoid committing secrets!

## Cursor Secrets

If you've added secrets to Cursor's account settings (`xai-hydra`, `openai-hydra`, `ANTHROPIC_API_KEY`), they may not be automatically available as environment variables. 

To use Cursor secrets:
1. Check Cursor's documentation on how secrets are exposed
2. Or manually set them as environment variables using one of the methods above
3. Or use command-line arguments to pass them to the test script

## What the Test Does

The script will:
- Test each API key with a simple request
- Show success/error status for each provider
- Display token usage information
- Provide a summary at the end

## Expected Output

```
Testing API Keys...
============================================================

Testing Anthropic API...
✓ Anthropic: SUCCESS
  Model: claude-3-haiku-20240307
  Response: test successful
  Usage: 15 input tokens, 3 output tokens

Testing XAI API...
✓ XAI: SUCCESS
  Model: grok-beta
  Response: test successful
  Usage: 12 prompt tokens, 3 completion tokens

Testing OpenAI API...
✓ OpenAI: SUCCESS
  Model: gpt-3.5-turbo
  Response: test successful
  Usage: 12 prompt tokens, 3 completion tokens

🎉 All API keys are working correctly!
```

## Troubleshooting

### "Package not installed" error
Install missing packages:
```bash
pip install anthropic openai
```

### "API key not set" error
Make sure you've set the environment variables or passed them as arguments.

### Authentication errors
- Verify your API keys are correct
- Check that your API keys haven't expired
- Ensure you have sufficient credits/quota

### Network errors
- Check your internet connection
- Verify firewall settings aren't blocking API requests

