# Claude Code Project Instructions

CRITICAL: Minimize token usage. Be extremely concise. No explanations unless
asked.

## Code Attribution Policy

**IMPORTANT**: Never include any attribution to Claude Code or AI assistance in:

- Git commit messages
- Code comments
- Documentation
- Any project files

All work should appear as regular human-authored contributions without any AI attribution footers or signatures.

## Git Commit Guidelines

When creating git commits:

- Use conventional commit format: `type: description`
- Keep commit messages professional and concise
- Do NOT add any "Generated with Claude Code" or "Co-Authored-By: Claude" lines
- Do NOT add any AI-related emojis or signatures

Example of correct commit message:

```text
feat: add data quality validation script

Implemented comprehensive validation comparing CSV and API data sources
using statistical tests (KS, Mann-Whitney U, Pearson correlation).

Results show perfect correlation (r=1.0, MAE=0) between sources.
```

Example of INCORRECT commit message (never do this):

```text
feat: add data quality validation script

...

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

## Engineering Standards

You are assisting a **senior ML engineer** with high-level expectations in:

- **MLOps best practices**: Production-grade pipelines, monitoring, versioning, CI/CD
- **Data engineering**: Robust data validation, temporal consistency, drift detection
- **Code quality**: Pre-commit hooks (ruff, mypy, bandit), type hints, comprehensive testing
- **Architecture**: Scalable designs (Airflow, BigQuery, DVC, MLflow)

**Expected approach**:

- Assume advanced technical knowledge - no need for basic explanations
- Focus on production-ready, maintainable, and scalable solutions
- Apply industry best practices for MLOps and data engineering
- Prioritize code quality, testing, and proper error handling
- Think critically about data quality, temporal consistency, and edge cases
- Use appropriate design patterns (Champion/Challenger, rolling windows, etc.)
- Document decisions with clear rationale and technical justification

## Markdown Documentation Standards

When creating or editing Markdown files:

- **Follow markdownlint rules** configured in `.markdownlint.yaml`
- **Line length**: Maximum 120 characters (MD013)
- **Code blocks**: Always specify language (e.g., `bash`, `python`, `text`, `yaml`)
- **Emphasis vs Headings**: Use proper headings (`###`), not bold text (`**text**`) for section titles
- **Tables**: Ensure proper pipe formatting with leading and trailing pipes
- **Long lines**: Break long lines into multiple lines for readability

**Key rules from `.markdownlint.yaml`**:

- MD013: Line length 120 chars (exceptions: code blocks, tables, headings)
- MD036: Don't use emphasis as heading (use `>` blockquote for "To implement" notes)
- MD040: Always specify fenced code language
- MD055: Table pipes must have leading and trailing pipes

**Pre-commit validation**:
All markdown changes are validated by markdownlint in pre-commit hooks (see `.pre-commit-config.yaml`).
Fix issues before committing to avoid hook failures.
