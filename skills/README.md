# User skills

This directory holds **optional skills** that GEAK discovers at runtime. Each skill is a small package of instructions the agent can load when a task matches the skill’s description.

## Where to put a skill

1. Create a **new folder** under this directory:

   `GEAK/skills/<your-skill-folder>/`

2. Add a file named **`SKILL.md`** inside that folder (exact name, case-sensitive).

GEAK scans **immediate subdirectories** of `skills/` and only picks up folders that contain `SKILL.md`. Files at the top level of `skills/` (not inside a folder) are ignored for discovery.

## Required shape of `SKILL.md`

### YAML frontmatter (mandatory)

The file **must** start with YAML between `---` delimiters. The runtime parses this with a YAML loader; **`name`** and **`description`** are required.

```yaml
---
name: my-skill-id
description: One line explaining when this skill should be used (shown in the agent’s skill list).
---
```

- **`name`**: Stable identifier for the skill. It must be **unique** among all skills in this folder. The agent selects a skill by this string (see below). Use lowercase, hyphens, and short words (for example `silu-optimization`).
- **`description`**: Written for the **model**. It should say *when* to use the skill (task type, domain, hardware, library, etc.). This text is injected into the system prompt inside `<available_skills>`.

If frontmatter is missing or invalid, that skill folder may be skipped during discovery.

### Markdown body (recommended)

Everything after the closing `---` is free-form Markdown. When the agent loads the skill, the **full file** (including frontmatter) is provided as context, so you can structure the body however you like.

Good practices:

- Start with a short title and **overview** of what the skill covers.
- Use clear sections (workflows, checklists, constraints, examples).
- Prefer **concrete** guidance: commands, file patterns, APIs, or patterns the agent should follow.
- If the skill is domain-specific (kernels, build systems, tests), spell out **assumptions** (data types, targets, repo layout).

See the built-in example: `examples/skills/silu-optimization/SKILL.md` — frontmatter plus structured sections (context, functionality, numbered optimizations with code excerpts).

## How the agent uses skills

1. **Discovery**: At startup, GEAK reads each `skills/<folder>/SKILL.md` and registers the skill by the **`name`** from frontmatter (not necessarily the folder name).
2. **Listing**: Only **`name`** and **`description`** from frontmatter are advertised in the system prompt.
3. **Loading**: When relevant, the model is instructed to request a skill; the full `SKILL.md` contents are then injected for that turn/session logic.

Keep **`description`** accurate so the model knows when to load the skill; put the detailed procedure in the Markdown body.

## Checklist for a new skill

| Step | Action |
|------|--------|
| 1 | Create `GEAK/skills/<folder>/` |
| 2 | Add `SKILL.md` with `---` YAML block containing `name` and `description` |
| 3 | Ensure `name` is unique across all skills in `skills/` |
| 4 | Write the body with task-specific guidance and examples |

After adding or editing skills, restart or reload the agent path that constructs `SkillRuntime` so discovery runs again.
