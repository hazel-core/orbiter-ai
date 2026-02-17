---
title: API Reference
description: REST API endpoint documentation
section: api-reference
order: 1
---

# API Reference

All API endpoints are prefixed with `/api/v1/`. Requests require authentication via session cookie or API key header.

## Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | Log in with email and password |
| POST | `/api/v1/auth/register` | Create a new account |
| GET | `/api/v1/auth/me` | Get current user |
| POST | `/api/v1/auth/logout` | Log out |
| GET | `/api/v1/auth/csrf` | Get CSRF token |

## Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/projects` | List projects |
| POST | `/api/v1/projects` | Create a project |
| GET | `/api/v1/projects/:id` | Get project |
| PUT | `/api/v1/projects/:id` | Update project |
| DELETE | `/api/v1/projects/:id` | Delete project |

## Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/agents` | List agents |
| POST | `/api/v1/agents` | Create an agent |
| GET | `/api/v1/agents/:id` | Get agent |
| PUT | `/api/v1/agents/:id` | Update agent |
| DELETE | `/api/v1/agents/:id` | Delete agent |
| POST | `/api/v1/agents/:id/duplicate` | Duplicate an agent |

## Workflows

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/workflows` | List workflows |
| POST | `/api/v1/workflows` | Create a workflow |
| GET | `/api/v1/workflows/:id` | Get workflow with canvas data |
| PUT | `/api/v1/workflows/:id` | Update workflow and canvas |
| DELETE | `/api/v1/workflows/:id` | Delete workflow |
| POST | `/api/v1/workflows/:id/run` | Execute a workflow |

## Tools

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tools` | List all tools (built-in + custom) |
| POST | `/api/v1/tools` | Create a custom tool |
| GET | `/api/v1/tools/:id` | Get tool |
| PUT | `/api/v1/tools/:id` | Update tool |
| DELETE | `/api/v1/tools/:id` | Delete tool |

## Knowledge Bases

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/knowledge-bases` | List knowledge bases |
| POST | `/api/v1/knowledge-bases` | Create a knowledge base |
| GET | `/api/v1/knowledge-bases/:id` | Get knowledge base |
| DELETE | `/api/v1/knowledge-bases/:id` | Delete knowledge base |
| POST | `/api/v1/knowledge-bases/:id/documents` | Upload a document |

## Deployments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/deployments` | List deployments |
| POST | `/api/v1/deployments` | Create a deployment |
| GET | `/api/v1/deployments/:id` | Get deployment |
| PUT | `/api/v1/deployments/:id` | Update deployment |
| DELETE | `/api/v1/deployments/:id` | Delete deployment |

## Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/runs` | List runs with filtering |
| GET | `/api/v1/runs/:id` | Get run details with trace |
| GET | `/api/v1/monitoring/costs` | Get cost aggregation |
| GET | `/api/v1/monitoring/health` | Get system health metrics |
| GET | `/api/v1/alerts` | List alert rules |
| POST | `/api/v1/alerts` | Create an alert rule |

## Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/conversations` | List conversations |
| GET | `/api/v1/conversations/:id` | Get conversation |
| DELETE | `/api/v1/conversations/:id` | Delete conversation |
| GET | `/api/v1/conversations/:id/messages` | Get conversation messages |

## Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/search?q=query` | Global search across entities |

## Providers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/providers` | List configured providers |
| POST | `/api/v1/providers` | Add a provider |
| PUT | `/api/v1/providers/:id` | Update provider |
| DELETE | `/api/v1/providers/:id` | Delete provider |
| POST | `/api/v1/providers/:id/test` | Test provider connectivity |

## Pagination

List endpoints return paginated results:

```json
{
  "data": [...],
  "pagination": {
    "next_cursor": "abc123",
    "has_more": true,
    "total": 42
  }
}
```

Pass `?cursor=abc123` to fetch the next page.

## Error Format

Errors return a JSON body with a `detail` field:

```json
{
  "detail": "Resource not found"
}
```

Common status codes: `400` (validation), `401` (unauthorized), `403` (forbidden), `404` (not found), `429` (rate limited).
