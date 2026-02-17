---
title: Getting Started
description: Create your first AI agent in Orbiter
section: getting-started
order: 1
---

# Getting Started with Orbiter

Welcome to Orbiter — a platform for building, testing, and deploying AI agents. This guide walks you through creating your first agent.

## Step 1: Create a Project

1. Navigate to **Projects** in the sidebar
2. Click **New Project**
3. Give it a name and description
4. Click **Create**

Projects organize your agents, workflows, and resources into logical groups.

## Step 2: Configure a Model Provider

Before creating agents, you need at least one model provider configured.

1. Go to **Settings** from the sidebar
2. Open the **Providers** tab
3. Click **Add Provider** and choose your provider (OpenAI, Anthropic, Gemini, etc.)
4. Enter your API key and click **Save**

## Step 3: Create Your First Agent

1. Navigate to **Agents** in the sidebar
2. Click **New Agent**
3. Fill in the basics:
   - **Name**: Give your agent a descriptive name
   - **Model**: Select a model from your configured providers
   - **System Prompt**: Define the agent's behavior and personality
4. Click **Create**

## Step 4: Test in the Playground

1. Go to **Playground** in the sidebar
2. Select your agent from the dropdown
3. Type a message and press Enter
4. Watch your agent respond in real-time with streaming

## Step 5: Add Tools (Optional)

Tools give your agent the ability to take actions — search the web, execute code, query databases, and more.

1. Open your agent's edit page
2. Scroll to the **Tools** section
3. Click **Add Tool** and select from the built-in catalog, or create a custom tool

## Next Steps

- **[Workflows](/docs/workflows)** — Build multi-step automation with a visual canvas
- **[Knowledge Base](/docs/knowledge-base)** — Give agents access to your documents via RAG
- **[Deployments](/docs/deployments)** — Deploy agents as API endpoints or embeddable widgets
- **[Monitoring](/docs/monitoring)** — Track runs, costs, and performance
