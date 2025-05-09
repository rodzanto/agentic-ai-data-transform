# Install Claude Code
#npm install -g @anthropic-ai/claude-code

# Configure for Bedrock with Claude 3.7 Sonnet
export CLAUDE_CODE_USE_BEDROCK=1
#export ANTHROPIC_MODEL='us.anthropic.claude-3-7-sonnet-20250219-v1:0'
export ANTHROPIC_MODEL='us.anthropic.claude-3-5-sonnet-20241022-v2:0'

# Control prompt caching - set to 1 to disable (see note below)
export DISABLE_PROMPT_CACHING=1

# Launch Claude Code
claude